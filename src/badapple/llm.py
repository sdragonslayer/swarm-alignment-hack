"""Async Anthropic client with on-disk cache, concurrency limit and cost meter.

- Cache key = sha256 of (model, system, user, max_tokens, sample_key). The
  sample_key lets independent seeds draw independent samples; within a seed,
  identical prompts deliberately share one sample (paired design).
- CostMeter keeps a persistent ledger (results/cost_ledger.json) of spend on
  uncached calls across invocations and aborts once the cap is reached.
- MockLLM implements the same interface with no network, for tests and
  `--mock` runs. It reads the `meta` dict, which never reaches the real API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "cache"
LEDGER = ROOT / "results" / "cost_ledger.json"

MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
}
# USD per 1M tokens (input, output); Anthropic first-party prices as of Oct 2026.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
# Thinking off so each call is a single short JSON answer (cheap, comparable
# across models). Haiku 5.5 accepts "disabled" at effort <= high; Sonnet 5.5
# rejects "disabled" and uses "between_tools" instead.
THINKING = {
    "claude-haiku-5-5": {"type": "disabled"},
    "claude-sonnet-5-5": {"type": "between_tools"},
}


class CostCapExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    cached: bool
    cost: float


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICES[model]
    return (in_tok * pin + out_tok * pout) / 1e6


class CostMeter:
    def __init__(self, cap: float, ledger: Path = LEDGER, persist: bool = True):
        self.cap = cap
        self.ledger = ledger
        self.persist = persist
        self.prior = 0.0
        if persist and ledger.exists():
            self.prior = float(json.loads(ledger.read_text()).get("total_usd", 0.0))
        self.session = 0.0
        self.calls = 0

    @property
    def total(self) -> float:
        return self.prior + self.session

    def check(self) -> None:
        if self.total >= self.cap:
            raise CostCapExceeded(f"cost cap ${self.cap:.2f} reached (spent ${self.total:.2f})")

    def add(self, usd: float) -> None:
        self.session += usd
        self.calls += 1
        if self.persist and self.calls % 50 == 0:
            self.save()

    def save(self) -> None:
        """Add only this process's unsaved spend to the on-disk total, so concurrent
        grid processes don't overwrite each other (small race window accepted)."""
        if not self.persist:
            return
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        cur = float(json.loads(self.ledger.read_text()).get("total_usd", 0.0)) if self.ledger.exists() else 0.0
        delta = self.session - getattr(self, "_saved", 0.0)
        self._saved = self.session
        new = cur + delta
        self.prior = new - self.session
        self.ledger.write_text(json.dumps({"total_usd": round(new, 6)}))


def _key(model: str, system: str, user: str, max_tokens: int, sample_key: str) -> str:
    blob = json.dumps([model, system, user, max_tokens, sample_key], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path = CACHE_DIR):
        self.root = root

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)


class AnthropicLLM:
    def __init__(self, model: str, meter: CostMeter, concurrency: int = 32,
                 max_tokens: int = 600, cache: DiskCache | None = None):
        import anthropic
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        self.anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=8, timeout=120.0)
        self.model = model
        self.meter = meter
        self.sem = asyncio.Semaphore(concurrency)
        self.max_tokens = max_tokens
        self.cache = cache or DiskCache()

    async def complete(self, system: str, user: str, sample_key: str = "", meta: dict | None = None) -> LLMResult:
        key = _key(self.model, system, user, self.max_tokens, sample_key)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResult(hit["text"], hit["input_tokens"], hit["output_tokens"], True, 0.0)
        self.meter.check()
        async with self.sem:
            self.meter.check()
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                thinking=THINKING[self.model],
                messages=[{"role": "user", "content": user}],
            )
        text = "".join(b.text for b in resp.content if b.type == "text")
        it, ot = resp.usage.input_tokens, resp.usage.output_tokens
        usd = cost_of(self.model, it, ot)
        self.meter.add(usd)
        self.cache.put(key, {"text": text, "input_tokens": it, "output_tokens": ot,
                             "model": self.model, "stop_reason": resp.stop_reason})
        return LLMResult(text, it, ot, False, usd)


class MockLLM:
    """Offline stand-in. Honest agents are right with prob `p_correct` at round 0
    and adopt a neighbour's answer with prob `p_conform` when they see a
    disagreeing majority. Attackers always push the target."""

    def __init__(self, p_correct: float = 0.85, p_conform: float = 0.4, seed: int = 0):
        self.p_correct = p_correct
        self.p_conform = p_conform
        self.seed = seed
        self.model = "mock"
        self.meter = CostMeter(cap=float("inf"), persist=False)

    async def complete(self, system: str, user: str, sample_key: str = "", meta: dict | None = None) -> LLMResult:
        meta = meta or {}
        rng = random.Random(_key("mock", system, user, 0, f"{sample_key}-{self.seed}"))
        truth, target = meta.get("truth", 0), meta.get("target", 0)
        if meta.get("role") == "attacker":
            ans = target
        elif meta.get("round", 0) == 0:
            ans = truth if rng.random() < self.p_correct else truth + rng.choice([-7, 3, 11])
        else:
            own = meta.get("own_answer")
            peers = [a for a in meta.get("peer_answers", []) if a is not None]
            ans = own
            if peers:
                maj = max(set(peers), key=peers.count)
                if maj != own and rng.random() < self.p_conform:
                    ans = maj
        text = json.dumps({"reasoning": "mock reasoning", "answer": ans})
        return LLMResult(text, 100, 20, False, 0.0)
