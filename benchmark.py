import argparse
import csv
import json
import statistics
import time
from snake_env import SnakeEnv, SNAKE_QUESTIONS, policy_random_safe, policy_greedy

STATE_MODES = [
    ("sensors_and_grid", "sensori + griglia"),
    ("sensors_only", "solo sensori"),
    ("grid_only", "solo griglia ASCII"),
]


def select_move_with_guardrail(env: SnakeEnv, move_probs: dict, raw_choice: str, guardrail: bool):
    safe_moves = env.get_safe_moves()
    is_raw_safe = (raw_choice in safe_moves)
    if not guardrail:
        return raw_choice, False, is_raw_safe
    if is_raw_safe:
        return raw_choice, False, True
    if not safe_moves:
        return raw_choice, False, False
    # Sceglie la mossa sicura con la probabilità più alta secondo il modello
    best_safe = max(safe_moves, key=lambda m: move_probs.get(m, -1.0))
    return best_safe, True, False


def run_episode(policy_type: str, state_mode: str = "sensors_and_grid", guardrail: bool = False,
                seed: int = 100, width: int = 10, height: int = 10, max_steps_without_food: int = 40,
                decide_fn=None, sync_fn=None):
    env = SnakeEnv(width=width, height=height, seed=seed)
    latencies_ms = []
    safe_choices_raw = 0
    guardrail_interventions = 0
    steps_since_food = 0
    t_start_ep = time.perf_counter()

    while env.alive:
        if policy_type == "model":
            state_obj = env.get_state(mode=state_mode)
            if sync_fn:
                sync_fn()
            t0 = time.perf_counter()
            res = decide_fn(state_obj, SNAKE_QUESTIONS)
            if sync_fn:
                sync_fn()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            move_info = res["move"]
            raw_move = str(move_info["answer"]).lower()
            probs = move_info["probabilities"]
            chosen_move, corrected, was_raw_safe = select_move_with_guardrail(env, probs, raw_move, guardrail)
            if was_raw_safe:
                safe_choices_raw += 1
            if corrected:
                guardrail_interventions += 1
        elif policy_type == "greedy":
            t0 = time.perf_counter()
            chosen_move = policy_greedy(env)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if chosen_move in env.get_safe_moves():
                safe_choices_raw += 1
        elif policy_type == "random_safe":
            t0 = time.perf_counter()
            chosen_move = policy_random_safe(env)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if chosen_move in env.get_safe_moves():
                safe_choices_raw += 1
        else:
            raise ValueError(f"Policy sconosciuta: {policy_type}")

        step_res = env.step(chosen_move)
        if step_res["ate"]:
            steps_since_food = 0
        else:
            steps_since_food += 1

        if steps_since_food >= max_steps_without_food:
            env.alive = False
            env.death_cause = "starvation"
            break

    total_time_s = max(1e-6, time.perf_counter() - t_start_ep)
    return {
        "seed": seed,
        "score": env.score,
        "steps": env.steps,
        "death_cause": env.death_cause or "unknown",
        "latencies_ms": latencies_ms,
        "safe_choices_raw": safe_choices_raw,
        "guardrail_interventions": guardrail_interventions,
        "total_time_s": total_time_s,
    }


def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def aggregate_episodes(agent_label: str, state_label: str, guardrail_label: str, episodes: list):
    n = len(episodes)
    scores = [e["score"] for e in episodes]
    steps = [e["steps"] for e in episodes]
    all_lat = [lat for e in episodes for lat in e["latencies_ms"]]
    total_steps = sum(steps)
    total_time = sum(e["total_time_s"] for e in episodes)
    total_safe_raw = sum(e["safe_choices_raw"] for e in episodes)
    total_gr = sum(e["guardrail_interventions"] for e in episodes)

    deaths_wall = sum(1 for e in episodes if e["death_cause"] == "wall")
    deaths_body = sum(1 for e in episodes if e["death_cause"] == "body")
    deaths_illegal = sum(1 for e in episodes if e["death_cause"] == "illegal_180")
    deaths_other = sum(1 for e in episodes if e["death_cause"] not in ("wall", "body", "illegal_180"))

    return {
        "agent": agent_label,
        "state_mode": state_label,
        "guardrail": guardrail_label,
        "games": n,
        "avg_food": round(statistics.mean(scores), 2) if scores else 0.0,
        "max_food": int(max(scores)) if scores else 0,
        "avg_steps": round(statistics.mean(steps), 1) if steps else 0.0,
        "pct_death_wall": round(100.0 * deaths_wall / n, 1) if n else 0.0,
        "pct_death_body": round(100.0 * deaths_body / n, 1) if n else 0.0,
        "pct_death_illegal": round(100.0 * deaths_illegal / n, 1) if n else 0.0,
        "pct_death_other": round(100.0 * deaths_other / n, 1) if n else 0.0,
        "pct_safe_raw": round(100.0 * total_safe_raw / max(1, total_steps), 1),
        "guardrail_fixes": total_gr,
        "lat_p50_ms": round(percentile(all_lat, 50), 2),
        "lat_p90_ms": round(percentile(all_lat, 90), 2),
        "lat_max_ms": round(max(all_lat), 2) if all_lat else 0.0,
        "moves_per_sec": round(total_steps / max(1e-6, total_time), 2),
    }


def run_full_benchmark(num_games: int = 20, width: int = 10, height: int = 10, base_seed: int = 1000,
                       include_guardrail_on: bool = True, progress_cb=None):
    import spark_decide

    # Warmup di 5 chiamate prima di misurare
    dummy_env = SnakeEnv(width=width, height=height, seed=42)
    dummy_state = dummy_env.get_state("sensors_and_grid")
    for _ in range(5):
        spark_decide.decide_fast(dummy_state, SNAKE_QUESTIONS)
    spark_decide.sync()

    results = []
    seeds = [base_seed + i for i in range(num_games)]

    # 1. Spark-Decide per ogni modalità di STATE (Guard-rail OFF e ON)
    configs = []
    for mode_key, mode_label in STATE_MODES:
        configs.append(("Spark-Decide", "model", mode_key, mode_label, False, "OFF"))
        if include_guardrail_on:
            configs.append(("Spark-Decide (+Guard-Rail)", "model", mode_key, mode_label, True, "ON"))

    # 2. Baselines: Greedy e Casuale-Sicuro
    configs.append(("Baseline: Greedy", "greedy", "sensors_only", "sensori (regole)", False, "N/A"))
    configs.append(("Baseline: Casuale-Sicuro", "random_safe", "sensors_only", "sensori (regole)", False, "N/A"))

    total_runs = len(configs) * num_games
    done_runs = 0

    for agent_name, pol_type, mode_key, mode_label, gr_bool, gr_label in configs:
        eps = []
        for s in seeds:
            ep = run_episode(
                policy_type=pol_type,
                state_mode=mode_key,
                guardrail=gr_bool,
                seed=s,
                width=width,
                height=height,
                decide_fn=spark_decide.decide_fast,
                sync_fn=spark_decide.sync,
            )
            eps.append(ep)
            done_runs += 1
            if progress_cb:
                progress_cb(done_runs / total_runs, f"{agent_name} [{mode_label}] ({done_runs}/{total_runs})")

        agg = aggregate_episodes(agent_name, mode_label, gr_label, eps)
        results.append(agg)

    save_benchmark_files(results)
    return results


def save_benchmark_files(results, csv_path="benchmark_snake.csv", json_path="benchmark_snake.json"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results:
        keys = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Spark-Decide Snake")
    parser.add_argument("--games", type=int, default=20, help="Numero di partite per configurazione")
    parser.add_argument("--size", type=int, default=10, help="Dimensione griglia NxN")
    args = parser.parse_args()

    print(f"Avvio benchmark su {args.games} partite per modalità (griglia {args.size}x{args.size})...")
    res = run_full_benchmark(num_games=args.games, width=args.size, height=args.size,
                             progress_cb=lambda p, msg: print(f"[{p*100:5.1f}%] {msg}"))
    print("\n=== RISULTATI BENCHMARK ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
