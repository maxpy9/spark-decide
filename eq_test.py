"""Test di equivalenza: modello di riferimento (bf16, codice originale) vs 4-bit (decide e decide_fast).
Uso:  python eq_test.py collect | ref | q4 | report
"""
import json, os, sys, time, random
from snake_env import SnakeEnv, SNAKE_QUESTIONS, policy_greedy, DIRECTIONS

MODES = ["sensors_and_grid", "sensors_only", "grid_only"]
N_POS = 20
OUT = "eq_results"
os.makedirs(OUT, exist_ok=True)


def collect():
    positions = []
    seed = 2000
    checkpoints = {3, 8, 15, 25, 40, 60, 80}
    while len(positions) < N_POS:
        env = SnakeEnv(10, 10, seed=seed)
        rnd = random.Random(seed)
        step = 0
        while env.alive and step < 90 and len(positions) < N_POS:
            if step in checkpoints:
                positions.append({
                    "seed": seed, "step": step,
                    "safe_moves": env.get_safe_moves(),
                    "states": {m: env.get_state(m) for m in MODES},
                })
            safe = env.get_safe_moves()
            mv = policy_greedy(env) if (rnd.random() < 0.75 or not safe) else rnd.choice(safe)
            env.step(mv)
            step += 1
        seed += 1
    json.dump(positions, open(f"{OUT}/positions.json", "w"))
    print("positions:", len(positions))


def run(tag, fn_names):
    import spark_decide as sd
    positions = json.load(open(f"{OUT}/positions.json"))
    w = positions[0]["states"]["sensors_and_grid"]
    for _ in range(3):
        sd.decide(w, SNAKE_QUESTIONS)
    sd.sync()
    for fn_name in fn_names:
        fn = getattr(sd, fn_name)
        rows = []
        for i, pos in enumerate(positions):
            for m in MODES:
                sd.sync(); t0 = time.perf_counter()
                r = fn(pos["states"][m], SNAKE_QUESTIONS)
                sd.sync(); ms = (time.perf_counter() - t0) * 1000
                rows.append({"i": i, "mode": m, "ms": ms, "res": r})
            print(f"[{tag}/{fn_name}] {i+1}/{len(positions)}  last {ms:.0f} ms", flush=True)
        json.dump(rows, open(f"{OUT}/{tag}_{fn_name}.json", "w"))


def report():
    import statistics as st
    positions = json.load(open(f"{OUT}/positions.json"))
    ref = json.load(open(f"{OUT}/ref_decide.json"))
    out = {}
    for name in ["q4_decide", "q4_decide_fast"]:
        path = f"{OUT}/{name}.json"
        if not os.path.exists(path):
            continue
        cand = json.load(open(path))
        per_mode = {}
        for m in MODES + ["ALL"]:
            pairs = [(a, b) for a, b in zip(ref, cand) if m == "ALL" or a["mode"] == m]
            mv = sum(a["res"]["move"]["answer"] == b["res"]["move"]["answer"] for a, b in pairs) / len(pairs)
            dg = sum(a["res"]["danger"]["answer"] == b["res"]["danger"]["answer"] for a, b in pairs) / len(pairs)
            rk = sum(a["res"]["risk"]["answer"] == b["res"]["risk"]["answer"] for a, b in pairs) / len(pairs)
            diffs = [abs(a["res"]["move"]["probabilities"][k] - b["res"]["move"]["probabilities"][k])
                     for a, b in pairs for k in DIRECTIONS]
            safe_ref = sum(a["res"]["move"]["answer"] in positions[a["i"]]["safe_moves"] for a, _ in pairs) / len(pairs)
            safe_c = sum(b["res"]["move"]["answer"] in positions[b["i"]]["safe_moves"] for _, b in pairs) / len(pairs)
            per_mode[m] = dict(n=len(pairs), move_agree=round(mv * 100, 1), danger_agree=round(dg * 100, 1),
                               risk_agree=round(rk * 100, 1), move_prob_mean_absdiff=round(st.mean(diffs), 4),
                               move_prob_max_absdiff=round(max(diffs), 4),
                               safe_pct_ref=round(safe_ref * 100, 1), safe_pct_cand=round(safe_c * 100, 1),
                               p50_ms_ref=round(st.median(a["ms"] for a, _ in pairs)),
                               p50_ms_cand=round(st.median(b["ms"] for _, b in pairs)))
        out[name] = per_mode
    # decide vs decide_fast (stessi pesi 4-bit): deve essere ~identico
    if os.path.exists(f"{OUT}/q4_decide.json") and os.path.exists(f"{OUT}/q4_decide_fast.json"):
        a_rows = json.load(open(f"{OUT}/q4_decide.json")); b_rows = json.load(open(f"{OUT}/q4_decide_fast.json"))
        d = [abs(a["res"][q]["probabilities"][k] - b["res"][q]["probabilities"][k])
             for a, b in zip(a_rows, b_rows) for q in a["res"] for k in a["res"][q]["probabilities"]]
        same = sum(all(a["res"][q]["answer"] == b["res"][q]["answer"] for q in a["res"]) for a, b in zip(a_rows, b_rows))
        out["decide_vs_decide_fast_4bit"] = dict(max_absdiff=round(max(d), 5), all_answers_equal=f"{same}/{len(a_rows)}")
    json.dump(out, open(f"{OUT}/report.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "collect": collect()
    elif ph == "ref": run("ref", ["decide"])
    elif ph == "q4": run("q4", ["decide", "decide_fast"])
    elif ph == "report": report()
