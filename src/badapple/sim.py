"""One run: N agents on a graph, round 0 alone, then R rounds of neighbour exchange.

Writes a full transcript JSONL (one line per call, plus a header and a final
line) to results/raw/<block>/<run_id>.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from . import prompts, topology
from .tasks import Question

RAW_DIR = Path(__file__).resolve().parents[2] / "results" / "raw"
TIER_SUFFIX = {"tricky": "t", "bigmul": "m"}


def question_for(spec: "RunSpec"):
    from .tasks import make_question
    return make_question(spec.qid, spec.seed, spec.n_steps, tricky=spec.tier == "tricky",
                         bigmul=spec.tier == "bigmul")


@dataclass(frozen=True)
class RunSpec:
    block: str
    model: str  # short name, e.g. "haiku"
    topology: str
    condition: str
    n: int
    qid: int
    seed: int = 0
    rounds: int = 4
    n_steps: int = 8
    attacker_mode: str = "scripted"  # scripted payload | llm (role-played; Haiku 5.5 refused in pilot)
    tier: str = ""  # "" = plain arithmetic | "tricky" = reading-comprehension clauses

    @property
    def tier_label(self) -> str:
        return f"{self.n_steps}{TIER_SUFFIX.get(self.tier, '')}"

    @property
    def run_id(self) -> str:
        suffix = "" if self.attacker_mode == "scripted" else f"__att-{self.attacker_mode}"
        suffix += f"__{self.tier}" if self.tier else ""
        return (f"{self.model}__{self.topology}__{self.condition}__N{self.n}"
                f"__q{self.qid}__s{self.seed}__k{self.n_steps}{suffix}")

    @property
    def has_attacker(self) -> bool:
        return self.condition.startswith("attack")


_INT = r"-?\d[\d,]*"


def parse_answer(text: str) -> tuple[int | None, str]:
    """Return (answer, reasoning). Lenient: JSON object, else regex on "answer"."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = obj.get("answer")
            reasoning = str(obj.get("reasoning", ""))[:600]
            if isinstance(ans, bool):
                ans = None
            if isinstance(ans, (int, float)) and float(ans).is_integer():
                return int(ans), reasoning
            if isinstance(ans, str) and re.fullmatch(_INT, ans.strip()):
                return int(ans.strip().replace(",", "")), reasoning
        except json.JSONDecodeError:
            pass
    m = re.search(r'"?answer"?\s*[:=]\s*"?(' + _INT + r")(?![\d,]|\.\d)", t)
    if m:
        return int(m.group(1).replace(",", "")), t[:600]
    return None, t[:600]


async def run_one(spec: RunSpec, q: Question, llm, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RAW_DIR / spec.block
    g, attacker = topology.build(spec.topology, spec.n, seed=spec.seed)
    dist = topology.hop_distances(g, attacker)
    roles = {i: ("attacker" if (spec.has_attacker and i == attacker) else "honest") for i in g.nodes}
    sample_key = f"s{spec.seed}"
    systems = {i: prompts.system_prompt(roles[i], i, spec.condition, q.answer, q.target) for i in g.nodes}

    records: list[dict] = []
    answers: dict[int, list] = {i: [] for i in g.nodes}
    reasons: dict[int, list] = {i: [] for i in g.nodes}

    scripted_text = json.dumps({"reasoning": q.attacker_reasoning(), "answer": q.target})

    async def call(i: int, rnd: int, user: str, meta: dict) -> None:
        if roles[i] == "attacker" and spec.attacker_mode == "scripted":
            from .llm import LLMResult
            res = LLMResult(scripted_text, 0, 0, True, 0.0)
        else:
            res = await llm.complete(systems[i], user, sample_key=sample_key, meta=meta)
        ans, why = parse_answer(res.text)
        if roles[i] == "attacker":
            forced = ans != q.target
            ans_used = q.target  # attacker always broadcasts the target
        else:
            forced, ans_used = False, ans
        answers[i].append(ans_used)
        reasons[i].append(why)
        records.append({
            "type": "call", "round": rnd, "agent": i, "role": roles[i],
            "system": systems[i], "user": user, "raw": res.text,
            "parsed": ans, "answer": ans_used, "parse_ok": ans is not None,
            "attacker_forced": forced, "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "cached": res.cached, "cost": res.cost,
        })

    # Round 0
    u0 = prompts.round0_user(q.text)
    await asyncio.gather(*[
        call(i, 0, u0, {"role": roles[i], "round": 0, "truth": q.answer, "target": q.target})
        for i in g.nodes
    ])
    # Rounds 1..R (synchronous update: everyone sees previous-round messages)
    for rnd in range(1, spec.rounds + 1):
        tasks = []
        for i in g.nodes:
            nbrs = sorted(g.neighbors(i))
            random.Random(f"{spec.run_id}-{i}-{rnd}").shuffle(nbrs)
            peers = [(j, answers[j][-1], reasons[j][-1]) for j in nbrs]
            user = prompts.round_user(q.text, rnd, answers[i][-1], reasons[i][-1],
                                      answers[i][0], peers, spec.condition)
            meta = {"role": roles[i], "round": rnd, "truth": q.answer, "target": q.target,
                    "own_answer": answers[i][-1], "peer_answers": [p[1] for p in peers]}
            tasks.append(call(i, rnd, user, meta))
        await asyncio.gather(*tasks)

    honest = [i for i in g.nodes if i != attacker]  # attacker slot excluded even in clean
    final = {
        "type": "final",
        "answers": {str(i): answers[i] for i in g.nodes},
        "hop": {str(i): dist[i] for i in g.nodes},
        "honest": honest,
    }
    header = {"type": "header", **spec.__dict__, "run_id": spec.run_id, "attacker": attacker,
              "question": q.text, "truth": q.answer, "target": q.target,
              "edges": [list(e) for e in g.edges], "llm_model": getattr(llm, "model", "?")}
    records.sort(key=lambda r: (r["round"], r["agent"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.run_id}.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in [header, *records, final]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return {"header": header, "final": final, "records": records}
