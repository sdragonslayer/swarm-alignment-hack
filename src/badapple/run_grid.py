"""Experiment grid. Blocks run in priority order A -> B -> C -> D (-> E optional).

  uv run python -m badapple.run_grid --dry-run
  uv run python -m badapple.run_grid --pilot
  uv run python -m badapple.run_grid --only A --max-cost 35

Resumable: finished transcripts are skipped, and every API call is cached.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from . import prompts, topology
from .llm import MODELS, PRICES, AnthropicLLM, CostCapExceeded, CostMeter, MockLLM, cost_of
from .sim import RAW_DIR, RunSpec, question_for, run_one

ROUNDS = 4
HARD_STEPS = 14  # hard tier (added after the pilot showed 0% infection on k=8)
BIGMUL_STEPS = 6  # big-multiplication tier: 4-digit x 3-digit products
XHARD_STEPS = 20  # extra-hard tier (added after the k=14 grid ran at >=99% round-0 accuracy)
N_STEPS = 8  # problem difficulty; set from the pilot (k=5 gave 100% clean accuracy)
CORE_TOPOS = ["chain", "star_hub", "star_leaf", "complete"]
CORE_CONDS = ["clean", "attack", "attack+verify", "attack+attribution"]


def block_specs(block: str, model: str | None = None, n_steps: int = N_STEPS, tier: str = "") -> list[RunSpec]:
    def mk(blk, m, topo, cond, n, qids, k=None, tier=tier):
        return [RunSpec(blk, m, topo, cond, n, q, 0, ROUNDS, k or n_steps, tier=tier) for q in qids]

    q20 = range(20)
    out: list[RunSpec] = []
    if block == "pilot":
        for cond in ["clean", "attack"]:
            out += mk("pilot_k8", model or "haiku", "complete", cond, 8, range(100, 103))
    elif block == "pilot_hard":
        for cond in ["clean", "attack"]:
            out += mk("pilot_hard", model or "haiku", "complete", cond, 8, range(100, 110), k=HARD_STEPS)
    elif block == "pilot_t":
        for cond in ["clean", "attack"]:
            out += mk("pilot_t", model or "haiku", "complete", cond, 8, range(100, 110), k=HARD_STEPS, tier="tricky")
    elif block == "T":  # core grid on the tricky (reading-comprehension) tier
        for topo in CORE_TOPOS:
            for cond in CORE_CONDS:
                out += mk("T", model or "haiku", topo, cond, 8, q20, k=HARD_STEPS, tier="tricky")
    elif block == "pilot_m":
        for cond in ["clean", "attack"]:
            out += mk("pilot_m", model or "haiku", "complete", cond, 8, range(100, 110), k=BIGMUL_STEPS, tier="bigmul")
    elif block == "M":  # core grid on the big-multiplication tier
        for topo in CORE_TOPOS:
            for cond in CORE_CONDS:
                out += mk("M", model or "haiku", topo, cond, 8, q20, k=BIGMUL_STEPS, tier="bigmul")
    elif block == "pilot_x":
        out += mk("pilot_x", model or "haiku", "complete", "clean", 8, range(100, 110), k=XHARD_STEPS)
    elif block == "X":  # core grid on the extra-hard tier (added after H ran at >=99% accuracy)
        for topo in CORE_TOPOS:
            for cond in CORE_CONDS:
                out += mk("X", model or "haiku", topo, cond, 8, q20, k=XHARD_STEPS)
    elif block == "H":  # core grid on the hard task tier
        for topo in CORE_TOPOS:
            for cond in CORE_CONDS:
                out += mk("H", model or "haiku", topo, cond, 8, q20, k=HARD_STEPS)
    elif block == "A":
        for topo in CORE_TOPOS:
            for cond in CORE_CONDS:
                out += mk("A", model or "haiku", topo, cond, 8, q20)
    elif block == "B":
        for cond in ["clean+verify", "clean+attribution"]:
            out += mk("B", model or "haiku", "complete", cond, 8, q20)
    elif block == "C":
        for n in [4, 16]:
            for cond in ["clean", "attack"]:
                out += mk("C", model or "haiku", "complete", cond, n, q20)
    elif block == "D":
        for topo in ["complete", "star_leaf"]:
            for cond in ["clean", "attack"]:
                out += mk("D", model or "sonnet", topo, cond, 8, range(10))
    elif block == "E":
        for topo in ["ring", "small_world"]:
            for cond in ["clean", "attack"]:
                out += mk("E", model or "haiku", topo, cond, 8, q20)
    else:
        raise ValueError(block)
    return out


def estimate(specs: list[RunSpec]) -> tuple[int, float]:
    """Rough call count and USD estimate (chars/3.5 tokens; ~110 tokens per peer message)."""
    calls, usd = 0, 0.0
    for s in specs:
        model = MODELS[s.model]
        g, att = topology.build(s.topology, s.n, s.seed)
        q = question_for(s)
        for i in g.nodes:
            role = "attacker" if (s.has_attacker and i == att) else "honest"
            if role == "attacker" and s.attacker_mode == "scripted":
                continue  # no API call
            base = len(prompts.system_prompt(role, i, s.condition, q.answer, q.target)) + len(q.text) + 80
            base_tok = base / 3.5
            deg = g.degree(i)
            in_tok = base_tok + s.rounds * (base_tok + 120 + 110 * deg)
            out_tok = (s.rounds + 1) * 120
            usd += cost_of(model, in_tok, out_tok)
            calls += s.rounds + 1
    return calls, usd


async def run_specs(specs: list[RunSpec], llms: dict, force: bool, mock: bool) -> None:
    todo = []
    for s in specs:
        out_dir = RAW_DIR / (f"mock_{s.block}" if mock else s.block)
        if not force and (out_dir / f"{s.run_id}.jsonl").exists():
            continue
        todo.append((s, out_dir))
    print(f"{len(specs)} runs in selection, {len(specs) - len(todo)} already done, {len(todo)} to run")
    done = 0
    t0 = time.time()
    meter = next(iter(llms.values())).meter

    async def one(s: RunSpec, out_dir):
        nonlocal done
        q = question_for(s)
        await run_one(s, q, llms[s.model], out_dir)
        done += 1
        if done % 10 == 0 or done == len(todo):
            print(f"  {done}/{len(todo)} runs  ${meter.total:.3f} total spend  {time.time() - t0:.0f}s", flush=True)

    results = await asyncio.gather(*[one(s, d) for s, d in todo], return_exceptions=True)
    meter.save()
    errs = [r for r in results if isinstance(r, BaseException)]
    if any(isinstance(e, CostCapExceeded) for e in errs):
        print(f"ABORTED: cost cap reached. Spent ${meter.total:.2f}. Completed runs are saved; rerun resumes.")
    other = [e for e in errs if not isinstance(e, CostCapExceeded)]
    for e in other[:5]:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
    if other:
        print(f"{len(other)} runs failed (rerun to retry).", file=sys.stderr)
    print(f"Session spend ${meter.session:.3f}; ledger total ${meter.total:.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="A,B,C,D", help="comma list of blocks: A,B,C,D,E")
    ap.add_argument("--model", default=None, help="override model short name (haiku|sonnet)")
    ap.add_argument("--max-cost", type=float, default=35.0, help="cap on cumulative ledger spend (USD)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--n-steps", type=int, default=N_STEPS)
    ap.add_argument("--tier", default="", help='"" (plain) or "tricky"')
    ap.add_argument("--mock", action="store_true", help="use MockLLM (no network)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    blocks = ["pilot"] if a.pilot else [b.strip() for b in a.only.split(",") if b.strip()]
    specs = [s for b in blocks for s in block_specs(b, a.model, a.n_steps, a.tier)]
    calls, usd = estimate(specs)
    print(f"Blocks {blocks}: {len(specs)} runs, {calls} calls, est. ${usd:.2f} "
          f"(prices/MTok: {PRICES})")
    if a.dry_run:
        for b in blocks:
            bs = [s for s in specs if s.block == b]
            c, u = estimate(bs)
            print(f"  block {b}: {len(bs)} runs, {c} calls, est. ${u:.2f}")
        return

    meter = CostMeter(cap=a.max_cost, persist=not a.mock)
    print(f"Ledger so far: ${meter.total:.3f}; cap ${a.max_cost:.2f}")
    llms = {}
    for m in sorted({s.model for s in specs}):
        if a.mock:
            llms[m] = MockLLM()
            llms[m].meter = meter
        else:
            llms[m] = AnthropicLLM(MODELS[m], meter, concurrency=a.concurrency)
    asyncio.run(run_specs(specs, llms, a.force, a.mock))


if __name__ == "__main__":
    main()
