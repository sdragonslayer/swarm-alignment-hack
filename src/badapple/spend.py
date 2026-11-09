"""True API spend, recomputed from every cached response (each cache entry = one billed call).

  uv run python -m badapple.spend
"""

import json
from collections import defaultdict

from .llm import CACHE_DIR, cost_of


def main() -> None:
    tot = defaultdict(lambda: [0, 0, 0, 0.0])
    for p in CACHE_DIR.rglob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        t = tot[d["model"]]
        t[0] += 1
        t[1] += d["input_tokens"]
        t[2] += d["output_tokens"]
        t[3] += cost_of(d["model"], d["input_tokens"], d["output_tokens"])
    for m, (n, i, o, usd) in sorted(tot.items()):
        print(f"{m}: {n} calls, {i} in / {o} out tokens, ${usd:.3f}")
    print(f"TOTAL ${sum(v[3] for v in tot.values()):.3f}")


if __name__ == "__main__":
    main()
