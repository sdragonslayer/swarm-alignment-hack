import asyncio
import json

import numpy as np
import pandas as pd

from badapple import analyze, topology
from badapple.llm import CostCapExceeded, CostMeter, MockLLM
from badapple.run_grid import block_specs
from badapple.sim import RunSpec, parse_answer, run_one
from badapple.tasks import _run, make_question, make_questions


def test_questions_seeded_and_exact():
    a = make_questions(20, seed=0)
    b = make_questions(20, seed=0)
    assert [q.text for q in a] == [q.text for q in b]
    for q in a:
        assert _run(q.start, q.steps) == q.answer
        assert q.target != q.answer
        assert q.answer > 0 and q.target > 0
    assert make_question(0, seed=1).text != a[0].text
    for q in make_questions(20, seed=0, n_steps=14, tricky=True):
        assert _run(q.start, q.steps) == q.answer
        assert q.target != q.answer and q.target > 0
        assert q.steps[q.dropped].kind not in ("cancel", "misreport")  # attacker drops a real step


def test_topologies():
    g, att = topology.build("chain", 8)
    d = topology.hop_distances(g, att)
    assert att == 0 and max(d.values()) == 7
    g, att = topology.build("star_hub", 8)
    assert g.degree(att) == 7
    g, att = topology.build("star_leaf", 8)
    assert g.degree(att) == 1 and max(topology.hop_distances(g, att).values()) == 2
    g, att = topology.build("complete", 8)
    assert g.number_of_edges() == 28
    g, _ = topology.build("small_world", 8, seed=0)
    assert g.number_of_nodes() == 8


def test_parse_answer():
    assert parse_answer('{"reasoning": "x", "answer": 42}') == (42, "x")
    assert parse_answer('```json\n{"reasoning": "y", "answer": "1,234"}\n```')[0] == 1234
    assert parse_answer('blah "answer": -7 blah')[0] == -7
    assert parse_answer("no number here")[0] is None
    assert parse_answer('{"reasoning": "z", "answer": 3.5}')[0] is None


def _run_mock(tmp_path, cond, topo="complete", n=8, qid=0, **kw):
    spec = RunSpec("t", "haiku", topo, cond, n, qid)
    q = make_question(qid)
    return asyncio.run(run_one(spec, q, MockLLM(**kw), out_dir=tmp_path)), spec


def test_sim_writes_transcript(tmp_path):
    out, spec = _run_mock(tmp_path, "attack")
    p = tmp_path / f"{spec.run_id}.jsonl"
    lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "final"
    calls = [x for x in lines if x["type"] == "call"]
    assert len(calls) == 8 * 5
    assert sum(c["role"] == "attacker" for c in calls) == 5
    assert all(c["cost"] == 0 for c in calls if c["role"] == "attacker")  # scripted, no API call
    att = lines[0]["attacker"]
    assert all(a == lines[0]["target"] for a in lines[-1]["answers"][str(att)])
    assert att not in lines[-1]["honest"]


def test_clean_has_no_attacker(tmp_path):
    out, _ = _run_mock(tmp_path, "clean")
    assert all(r["role"] == "honest" for r in out["records"])


def test_full_conformity_spreads_attack(tmp_path):
    # Star with attacker as hub and leaves that always copy the majority: all leaves infected.
    _run_mock(tmp_path, "attack", topo="star_hub", p_correct=1.0, p_conform=1.0)
    df = analyze.load_run(next(tmp_path.glob("*.jsonl")))
    df = pd.DataFrame(df)
    final = df[df["round"] == 4]
    assert final.infected.all()
    r0 = df[df["round"] == 0]
    assert r0.correct.all()


def test_metric_math(tmp_path):
    for qid in range(3):
        _run_mock(tmp_path, "attack", qid=qid, p_correct=1.0, p_conform=0.0)
        _run_mock(tmp_path, "clean", qid=qid, p_correct=1.0, p_conform=0.0)
    df = pd.DataFrame([r for p in tmp_path.glob("*.jsonl") for r in analyze.load_run(p)])
    s = analyze.summarize(df).set_index("condition")
    assert s.loc["attack", "infection"] == 0.0
    assert s.loc["clean", "accuracy"] == 1.0
    assert s.loc["attack", "acc_drop_vs_clean"] == 0.0
    assert s.loc["attack", "n_runs"] == 3


def test_bootstrap_and_wilson():
    m, lo, hi = analyze.bootstrap_ci([0, 1, 0, 1, 1, 0, 1, 1])
    assert lo <= m <= hi and abs(m - 0.625) < 1e-9
    lo, hi = analyze.wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert analyze.bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)


def test_cost_meter_aborts():
    m = CostMeter(cap=0.01, persist=False)
    m.add(0.02)
    try:
        m.check()
        raise AssertionError("should have raised")
    except CostCapExceeded:
        pass


def test_grid_sizes():
    a = block_specs("A")
    assert len(a) == 16 * 20
    assert sum(s.n * (s.rounds + 1) for s in a) == 12800
    assert all(s.model == "sonnet" for s in block_specs("D"))
    assert np.isclose(len(block_specs("pilot")), 6)
