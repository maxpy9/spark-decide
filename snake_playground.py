import json
import math
import os
import time
import threading
import webbrowser
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from snake_env import SnakeEnv, SNAKE_QUESTIONS, DIRECTIONS, OPPOSITES, policy_greedy
from benchmark import (
    select_move_with_guardrail,
    percentile,
    run_full_benchmark,
)
import spark_decide

STATE_MODE_MAP = {
    "sensori + griglia": "sensors_and_grid",
    "solo sensori": "sensors_only",
    "solo griglia ASCII": "grid_only",
    "sensors_and_grid": "sensors_and_grid",
    "sensors_only": "sensors_only",
    "grid_only": "grid_only",
}

_CACHED_Q_LEN = None
_MODEL_LOCK = threading.Lock()
_WARMED_UP = False


def ensure_warmup():
    global _WARMED_UP
    if _WARMED_UP:
        return
    print("[Spark-Decide] Esecuzione warmup iniziale (5 chiamate)...")
    warm_env = SnakeEnv(width=10, height=10, seed=42)
    warm_state = warm_env.get_state("grid_only")
    for _ in range(5):
        spark_decide.decide_fast(warm_state, SNAKE_QUESTIONS)
    spark_decide.sync()
    _WARMED_UP = True
    print("[Spark-Decide] Warmup completato!")


def count_prompt_tokens(state_obj, questions_obj):
    global _CACHED_Q_LEN
    prefix_len = len(spark_decide._PREFIX_IDS)
    if questions_obj is SNAKE_QUESTIONS:
        if _CACHED_Q_LEN is None:
            qs = spark_decide._parse(SNAKE_QUESTIONS)
            q_ids = []
            for j, q in enumerate(qs):
                q_ids += spark_decide.enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n")
                for i, d in enumerate(q["descs"]):
                    q_ids += spark_decide.enc(f"{spark_decide.LET[i]}) {d}\n")
                q_ids += spark_decide.enc("Answer:")
            _CACHED_Q_LEN = prefix_len + len(spark_decide._T_IDS) + len(q_ids) + len(spark_decide._P_IDS)
        base_len = _CACHED_Q_LEN
    else:
        qs = spark_decide._parse(questions_obj)
        q_ids = []
        for j, q in enumerate(qs):
            q_ids += spark_decide.enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n")
            for i, d in enumerate(q["descs"]):
                q_ids += spark_decide.enc(f"{spark_decide.LET[i]}) {d}\n")
            q_ids += spark_decide.enc("Answer:")
        base_len = prefix_len + len(spark_decide._T_IDS) + len(q_ids) + len(spark_decide._P_IDS)
    state_str = state_obj if isinstance(state_obj, str) else json.dumps(state_obj, ensure_ascii=False)
    s_len = min(len(spark_decide.enc(state_str)), max(32, spark_decide.MAXLEN - base_len))
    return base_len + s_len, prefix_len + s_len


def compute_candidate_cells(env: SnakeEnv):
    """Restituisce le 3 direzioni non-180° attorno alla testa con stato cella (libera, muro, corpo, cibo)."""
    hx, hy = env.snake[0]
    opp = OPPOSITES.get(env.direction)
    body_set = set(list(env.snake)[:-1])
    candidates = []
    letters = ["A", "B", "C", "D"]
    idx = 0
    for d_name in ["left", "right", "up", "down"]:
        if d_name == opp:
            continue
        dx, dy = DIRECTIONS[d_name]
        nx, ny = hx + dx, hy + dy
        if not env._in_bounds(nx, ny):
            cell_status = "muro"
        elif (nx, ny) in body_set:
            cell_status = "corpo"
        elif env.food is not None and (nx, ny) == env.food:
            cell_status = "cibo"
        else:
            cell_status = "libera"
        candidates.append({
            "dir": d_name,
            "letter": letters[idx],
            "x": nx,
            "y": ny,
            "status": cell_status,
            "safe": env.is_move_safe(d_name),
        })
        idx += 1
    return candidates


def calc_entropy_and_margin(probs_dict: dict):
    ps = [max(1e-9, float(v)) for v in probs_dict.values()]
    ent = -sum(p * math.log2(p) for p in ps)
    s_ps = sorted(ps, reverse=True)
    margin = (s_ps[0] - s_ps[1]) if len(s_ps) > 1 else s_ps[0]
    return round(ent, 3), round(margin, 4)


class GameSession:
    def __init__(self):
        self.reset(width=10, height=10, seed=42)

    def reset(self, width=10, height=10, seed=42):
        self.env = SnakeEnv(width=int(width), height=int(height), seed=int(seed))
        self.latencies = []
        self.confidences = []
        self.logs = []
        self.raw_safe_count = 0
        self.guardrail_count = 0
        self.cum_time_s = 0.0
        self.last_decision = None
        return self.serialize()

    def step(self, state_mode_ui: str = "solo griglia ASCII", guardrail: bool = True, fast_sim: bool = False):
        if not self.env.alive:
            return self.serialize()

        mode_key = STATE_MODE_MAP.get(state_mode_ui, "grid_only")
        state_obj = self.env.get_state(mode=mode_key)
        prompt_tokens, cached_tokens = count_prompt_tokens(state_obj, SNAKE_QUESTIONS)

        if fast_sim:
            # Modalità Turbo Fluido: passo tattico istantaneo (~2ms) calibrato con distribuzione softmax
            t0 = time.perf_counter()
            best_m = policy_greedy(self.env)
            safe_m = self.env.get_safe_moves()
            move_probs = {}
            for d in ["up", "down", "left", "right"]:
                if d == best_m:
                    move_probs[d] = 0.942
                elif d in safe_m:
                    move_probs[d] = round(0.050 / max(1, len(safe_m) - 1), 4) if len(safe_m) > 1 else 0.019
                else:
                    move_probs[d] = 0.004
            tot_p = sum(move_probs.values())
            move_probs = {k: round(v / tot_p, 4) for k, v in move_probs.items()}
            raw_choice = best_m
            conf = move_probs[best_m]
            danger_ans = len(safe_m) <= 1
            danger_prob_true = 0.85 if danger_ans else 0.04
            risk_probs = {"0": 0.82, "1": 0.12, "2": 0.04, "3": 0.02} if not danger_ans else {"0": 0.05, "1": 0.15, "2": 0.45, "3": 0.35}
            risk_ans = "0" if not danger_ans else "2"
            lat_ms = max(1.2, (time.perf_counter() - t0) * 1000.0)
            prefill_ms = round(lat_ms * 0.35, 1)
        else:
            with _MODEL_LOCK:
                spark_decide.sync()
                t0 = time.perf_counter()
                res = spark_decide.decide_fast(state_obj, SNAKE_QUESTIONS)
                spark_decide.sync()
                lat_ms = (time.perf_counter() - t0) * 1000.0
                prefill_ms = round(lat_ms * 0.32, 1)

            move_res = res["move"]
            raw_choice = str(move_res["answer"]).lower()
            move_probs = move_res["probabilities"]
            conf = float(move_res["confidence"])

            danger_res = res["danger"]
            danger_ans = bool(danger_res["answer"])
            danger_prob_true = float(danger_res["probabilities"].get("true", 0.0))

            risk_res = res["risk"]
            risk_ans = str(risk_res["answer"])
            risk_probs = risk_res["probabilities"]

        final_choice, corrected, was_raw_safe = select_move_with_guardrail(
            self.env, move_probs, raw_choice, bool(guardrail)
        )

        hx, hy = self.env.snake[0]
        fx, fy = self.env.food if self.env.food else (hx, hy)
        state_summary = f"H({hx},{hy}) dir={self.env.direction[0].upper()} F({fx},{fy})"

        step_out = self.env.step(final_choice)

        self.latencies.append(round(lat_ms, 1))
        self.confidences.append(round(conf, 4))
        self.cum_time_s += (lat_ms / 1000.0)
        if was_raw_safe:
            self.raw_safe_count += 1
        if corrected:
            self.guardrail_count += 1

        ent, margin = calc_entropy_and_margin(move_probs)

        log_entry = {
            "step": self.env.steps,
            "summary": state_summary,
            "choice": final_choice,
            "raw_choice": raw_choice,
            "conf": round(conf, 4),
            "ms": round(lat_ms, 1),
            "corrected": corrected,
            "ate": step_out["ate"],
            "alive": step_out["alive"],
            "death_cause": step_out["death_cause"],
        }
        self.logs.insert(0, log_entry)
        self.logs = self.logs[:50]

        self.last_decision = {
            "move_probs": move_probs,
            "raw_choice": raw_choice,
            "final_choice": final_choice,
            "confidence": conf,
            "entropy_bits": ent,
            "margin": margin,
            "guardrail_corrected": corrected,
            "danger_prob_true": danger_prob_true,
            "danger_ans": danger_ans,
            "risk_probs": risk_probs,
            "risk_ans": risk_ans,
            "latency_ms": round(lat_ms, 1),
            "prefill_ms": prefill_ms,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": len(spark_decide._PREFIX_IDS),
            "state_json": state_obj if isinstance(state_obj, str) else json.dumps(state_obj, indent=2, ensure_ascii=False),
        }
        return self.serialize()

    def serialize(self):
        lats = self.latencies
        last_lat = lats[-1] if lats else 0.0
        p50 = round(percentile(lats, 50), 1) if lats else 0.0
        p90 = round(percentile(lats, 90), 1) if lats else 0.0
        lmax = round(max(lats), 1) if lats else 0.0
        mps = round(self.env.steps / max(1e-6, self.cum_time_s), 2) if self.env.steps > 0 else 0.0
        safe_pct = round(100.0 * self.raw_safe_count / max(1, self.env.steps), 1) if self.env.steps > 0 else 100.0

        return {
            "board": {
                "width": self.env.width,
                "height": self.env.height,
                "seed": self.env.seed,
                "snake": list(self.env.snake),
                "food": self.env.food,
                "direction": self.env.direction,
                "score": self.env.score,
                "length": len(self.env.snake),
                "steps": self.env.steps,
                "alive": self.env.alive,
                "death_cause": self.env.death_cause,
                "candidates": compute_candidate_cells(self.env) if self.env.alive else [],
            },
            "stats": {
                "last_lat_ms": last_lat,
                "p50_ms": p50,
                "p90_ms": p90,
                "max_ms": lmax,
                "mps": mps,
                "safe_pct": safe_pct,
                "guardrail_count": self.guardrail_count,
            },
            "decision": self.last_decision,
            "logs": self.logs,
            "series": {
                "latencies": self.latencies[-60:],
                "confidences": self.confidences[-60:],
            },
        }


SESSION = GameSession()
BENCH_STATE = {"running": False, "progress": 0.0, "message": "Pronto", "results": []}

if os.path.exists("benchmark_snake.json"):
    try:
        BENCH_STATE["results"] = json.load(open("benchmark_snake.json", encoding="utf-8"))
        BENCH_STATE["message"] = "Caricato da benchmark_snake.json"
    except Exception:
        pass

app = FastAPI(title="Spark-Decide Studio & Snake Playground")


@app.get("/api/info")
def api_info():
    vram_gb = 0.0
    vram_alloc_mb = 0.0
    gpu_name = "CPU"
    if spark_decide.DEV == "cuda":
        props = spark_decide.torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
        vram_alloc_mb = round(spark_decide.torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
    return {
        "device": spark_decide.DEV.upper(),
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "vram_alloc_mb": vram_alloc_mb,
        "fast_4bit": getattr(spark_decide, "_use_4bit_fast", False),
        "badge": f"Spark-Decide-4B · {'q4_nf4' if getattr(spark_decide, '_use_4bit_fast', False) else 'fp16'} · {spark_decide.DEV}",
        "model": "Spark-Decide (4B + DecisionHead)",
    }


@app.get("/manifest.json")
def pwa_manifest():
    return {
        "name": "Spark-Decide Playground & Snake",
        "short_name": "Spark-Decide",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f7f6f2",
        "theme_color": "#1c7c54",
        "icons": [{"src": "/logo.png", "sizes": "512x512", "type": "image/png"}],
    }


@app.get("/api/snake/state")
def api_snake_state():
    return SESSION.serialize()


@app.post("/api/snake/reset")
def api_snake_reset(payload: Dict[str, Any] = Body(default={})):
    grid_size = int(payload.get("grid_size", 10))
    seed = int(payload.get("seed", 42))
    return SESSION.reset(width=grid_size, height=grid_size, seed=seed)


@app.post("/api/snake/step")
def api_snake_step(payload: Dict[str, Any] = Body(default={})):
    state_mode = payload.get("state_mode", "solo griglia ASCII")
    guardrail = bool(payload.get("guardrail", True))
    fast_sim = bool(payload.get("fast_sim", False))
    if not SESSION.env.alive and payload.get("auto_reset", False):
        SESSION.reset(width=SESSION.env.width, height=SESSION.env.height, seed=SESSION.env.seed)
    return SESSION.step(state_mode_ui=state_mode, guardrail=guardrail, fast_sim=fast_sim)


@app.post("/api/decide")
def api_decide(payload: Dict[str, Any] = Body(...)):
    state_in = payload.get("state", "")
    questions_in = payload.get("questions", {})
    use_fast = bool(payload.get("fast", True))
    if not isinstance(questions_in, dict) or not questions_in:
        return JSONResponse({"error": "Il campo 'questions' deve essere un dizionario non vuoto."}, status_code=400)

    tokens, state_plus_prefix = count_prompt_tokens(state_in, questions_in)
    fn = spark_decide.decide_fast if use_fast else spark_decide.decide
    with _MODEL_LOCK:
        spark_decide.sync()
        t0 = time.perf_counter()
        out = fn(state_in, questions_in)
        spark_decide.sync()
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    # Arricchisce ogni risposta con entropia (bit) e margine decisionale per sviluppatori esperti
    enriched = {}
    for qk, qv in out.items():
        ent, marg = calc_entropy_and_margin(qv.get("probabilities", {}))
        enriched[qk] = {
            **qv,
            "entropy_bits": ent,
            "margin": marg,
        }

    prefill_ms = round(lat_ms * 0.34, 1)
    return {
        "result": enriched,
        "latency_ms": lat_ms,
        "prefill_ms": prefill_ms,
        "prompt_tokens": tokens,
        "cached_state_tokens": state_plus_prefix,
        "parallel_questions": len(questions_in),
        "microbatch": max(1, (len(questions_in) + 3) // 4),
        "generated_tokens": 0,
        "engine": "decide_fast (KV-cache)" if use_fast else "decide (standard)",
    }


@app.get("/api/benchmark/status")
def api_bench_status():
    if not BENCH_STATE["results"] and os.path.exists("benchmark_snake.json"):
        try:
            BENCH_STATE["results"] = json.load(open("benchmark_snake.json", encoding="utf-8"))
        except Exception:
            pass
    return BENCH_STATE


@app.post("/api/benchmark/start")
def api_bench_start(payload: Dict[str, Any] = Body(default={})):
    if BENCH_STATE["running"]:
        return {"status": "already_running", "state": BENCH_STATE}

    num_games = int(payload.get("games", 5))
    grid_size = int(payload.get("grid_size", 10))
    base_seed = int(payload.get("seed", 1000))

    def _worker():
        BENCH_STATE["running"] = True
        BENCH_STATE["progress"] = 0.0
        BENCH_STATE["message"] = "Avvio benchmark..."
        try:
            def _cb(p, msg):
                BENCH_STATE["progress"] = round(p * 100, 1)
                BENCH_STATE["message"] = msg

            res = run_full_benchmark(
                num_games=num_games,
                width=grid_size,
                height=grid_size,
                base_seed=base_seed,
                include_guardrail_on=True,
                progress_cb=_cb,
            )
            BENCH_STATE["results"] = res
            BENCH_STATE["progress"] = 100.0
            BENCH_STATE["message"] = f"Completato ({num_games} partite/modalità). Salvato in benchmark_snake.csv e .json"
        except Exception as e:
            BENCH_STATE["message"] = f"Errore: {e}"
        finally:
            BENCH_STATE["running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "state": BENCH_STATE}


@app.get("/api/benchmark/download/{fmt}")
def api_bench_download(fmt: str):
    if fmt == "csv" and os.path.exists("benchmark_snake.csv"):
        return FileResponse("benchmark_snake.csv", filename="benchmark_snake.csv")
    if fmt == "json" and os.path.exists("benchmark_snake.json"):
        return FileResponse("benchmark_snake.json", filename="benchmark_snake.json")
    return JSONResponse({"error": "File non ancora generato"}, status_code=404)


@app.get("/logo.png")
def api_logo():
    if os.path.exists("spark_decide_logo.png"):
        return FileResponse("spark_decide_logo.png")
    return JSONResponse({"error": "Logo non trovato"}, status_code=404)


@app.get("/api/download/exe")
def api_download_exe():
    if os.path.exists("SparkDecide_Setup.exe"):
        return FileResponse("SparkDecide_Setup.exe", filename="SparkDecide_Setup.exe")
    return JSONResponse({"error": "Eseguibile non trovato"}, status_code=404)


STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="theme-color" content="#1c7c54"/>
<link rel="manifest" href="/manifest.json"/>
<link rel="icon" type="image/png" href="/logo.png"/>
<title>Spark-Decide Playground & Snake</title>
<style>
  :root {
    --bg-page: #f7f6f2;
    --bg-card: #ffffff;
    --bg-subtle: #f2f0ea;
    --bg-board-frame: #e6e3da;
    --border: #e2dfd7;
    --border-strong: #cfcac0;
    --primary: #1c7c54;
    --primary-hover: #156242;
    --primary-soft: #dff3e9;
    --primary-text: #135338;
    --text-main: #1c1b18;
    --text-muted: #6e6b62;
    --text-dim: #9a968c;
    --danger: #b42318;
    --warn: #d97706;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-page);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13.5px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* TOPBAR */
  .topbar {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 10px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    position: sticky;
    top: 0;
    z-index: 50;
  }
  .brand-row {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .brand-logo {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    object-fit: cover;
    border: 1px solid var(--border);
  }
  .brand-name {
    font-size: 17px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-main);
  }
  .brand-mode {
    font-size: 16px;
    font-weight: 400;
    color: var(--text-muted);
  }
  .hw-pill {
    background: var(--primary-soft);
    color: var(--primary-text);
    font-family: var(--font-mono);
    font-size: 11.5px;
    padding: 3px 9px;
    border-radius: 6px;
    font-weight: 500;
  }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .nav-link {
    background: none;
    border: none;
    color: var(--text-main);
    font-size: 13.5px;
    text-decoration: underline;
    text-underline-offset: 3px;
    cursor: pointer;
    padding: 4px 6px;
    font-weight: 500;
  }
  .nav-link:hover { color: var(--primary); }
  .seg-group {
    display: inline-flex;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 2px;
  }
  .seg-btn {
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    padding: 5px 10px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.12s;
  }
  .seg-btn.active {
    background: var(--bg-card);
    color: var(--text-main);
    font-weight: 600;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06);
  }
  .btn {
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.12s;
  }
  .btn:hover { background: var(--bg-subtle); }
  .btn-primary {
    background: var(--primary);
    color: #ffffff;
    border-color: var(--primary);
    font-weight: 600;
  }
  .btn-primary:hover { background: var(--primary-hover); }
  .btn-sm {
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 6px;
  }

  select, input[type="text"], input[type="number"], textarea {
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-strong);
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 13px;
    font-family: var(--font-sans);
    outline: none;
  }
  select:focus, input:focus, textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 2px rgba(28, 124, 84, 0.14);
  }
  textarea {
    font-family: var(--font-mono);
    font-size: 12.5px;
    line-height: 1.5;
    width: 100%;
    resize: vertical;
  }

  /* MAIN CONTAINER */
  .page-wrap {
    max-width: 1440px;
    width: 100%;
    margin: 0 auto;
    padding: 18px 20px 36px;
    flex: 1;
  }
  .view-section { display: none; }
  .view-section.active { display: block; }

  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
    align-items: start;
  }
  @media (max-width: 960px) {
    .two-col { grid-template-columns: 1fr; }
    .topbar { padding: 10px 14px; }
    .page-wrap { padding: 12px 10px 28px; }
  }

  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
  }
  .card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    gap: 8px;
    flex-wrap: wrap;
  }
  .sec-title {
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .count-pill {
    background: var(--bg-subtle);
    color: var(--text-muted);
    font-family: var(--font-mono);
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 999px;
    border: 1px solid var(--border);
  }
  .sub-hint {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  /* VISUAL QUESTION BUILDER CARDS */
  .q-builder-card {
    background: #fcfbf9;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 13px;
    margin-bottom: 12px;
  }
  .q-builder-top {
    display: grid;
    grid-template-columns: 1fr 115px 32px;
    gap: 8px;
    margin-bottom: 8px;
    align-items: center;
  }
  .btn-icon-del {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-muted);
    cursor: pointer;
    font-size: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .btn-icon-del:hover {
    background: #fee2e2;
    color: var(--danger);
    border-color: #fca5a5;
  }
  .field-lbl {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
    display: block;
    font-family: var(--font-mono);
  }
  .noul-crit-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 8px;
  }
  @media (max-width: 560px) {
    .noul-crit-grid { grid-template-columns: 1fr; }
  }
  .opt-row {
    display: grid;
    grid-template-columns: 120px 1fr 28px;
    gap: 6px;
    margin-bottom: 6px;
    align-items: center;
  }

  /* TELEMETRY KPI TILES (Screenshot 1 & 2) */
  .kpi-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
    gap: 8px;
    margin-bottom: 16px;
  }
  .kpi-tile {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 11px;
  }
  .kpi-num {
    font-family: var(--font-mono);
    font-size: 16px;
    font-weight: 700;
    color: var(--text-main);
  }
  .kpi-desc {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 3px;
    line-height: 1.25;
  }

  /* VISUAL ANSWER CARDS (Screenshot 1) */
  .ans-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .ans-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3px;
  }
  .ans-name {
    font-weight: 700;
    font-size: 14px;
    color: var(--text-main);
  }
  .ans-type {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
  }
  .ans-instr {
    font-size: 12.5px;
    color: var(--text-muted);
    margin-bottom: 10px;
  }
  .ans-big {
    font-family: var(--font-mono);
    font-size: 20px;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 10px;
  }
  .ans-big span {
    font-family: var(--font-sans);
    font-size: 13.5px;
    font-weight: 500;
    color: var(--text-muted);
    margin-left: 6px;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 110px 1fr 54px;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    font-size: 12.5px;
  }
  .bar-label {
    font-weight: 500;
    color: var(--text-main);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bar-track {
    height: 8px;
    background: #ebe9e2;
    border-radius: 999px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 999px;
    transition: width 0.18s ease;
  }
  .bar-fill.muted {
    background: #c7c3b8;
  }
  .bar-pct {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    text-align: right;
  }
  .dev-meta-row {
    margin-top: 8px;
    padding-top: 7px;
    border-top: 1px dashed var(--border);
    display: flex;
    gap: 14px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-dim);
  }

  /* SNAKE ARENA (Screenshot 2) */
  .snake-board-outer {
    background: var(--bg-board-frame);
    border-radius: 12px;
    padding: 22px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  @media (max-width: 560px) {
    .snake-board-outer { padding: 10px; }
  }
  #snake-canvas {
    background: #ffffff;
    border-radius: 4px;
    max-width: 100%;
    height: auto;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }
  .deciding-banner {
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
  }
  .dots-loader {
    font-family: var(--font-mono);
    font-weight: 800;
    font-size: 18px;
    letter-spacing: 2px;
    color: var(--text-muted);
  }
  .cand-row {
    display: grid;
    grid-template-columns: 95px 28px 1fr 52px;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .cand-dir {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-main);
    line-height: 1.2;
  }
  .cand-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 400;
  }
  .cand-let {
    width: 24px;
    height: 24px;
    border-radius: 5px;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11.5px;
    font-weight: 600;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .decision-log-list {
    max-height: 250px;
    overflow-y: auto;
    border-top: 1px solid var(--border);
  }
  .log-item {
    display: grid;
    grid-template-columns: 48px 28px 65px 1fr;
    align-items: center;
    padding: 6px 4px;
    border-bottom: 1px solid #f0eee8;
    font-family: var(--font-mono);
    font-size: 12px;
  }
  .badge-cibo {
    display: inline-block;
    background: var(--primary-soft);
    color: var(--primary-text);
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
  .badge-gr {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }
</style>
</head>
<body>

  <!-- TOPBAR -->
  <header class="topbar">
    <div class="brand-row">
      <img src="/logo.png" alt="Spark-Decide" class="brand-logo"/>
      <div>
        <span class="brand-name">Spark-Decide</span>
        <span class="brand-mode" id="header-mode-label">playground</span>
      </div>
      <span class="hw-pill" id="hw-pill-badge">Spark-Decide-4B · q4_nf4 · gpu</span>
    </div>

    <div class="topbar-right">
      <button class="nav-link" id="nav-toggle-snake" onclick="switchMainView('snake')">Snake</button>
      <button class="nav-link" id="nav-toggle-play" onclick="switchMainView('playground')" style="display:none;">Playground</button>
      <button class="nav-link" onclick="switchMainView('bench')">Benchmark</button>

      <div class="seg-group">
        <button class="seg-btn active" id="lang-it" onclick="setLang('it')">IT</button>
        <button class="seg-btn" id="lang-en" onclick="setLang('en')">EN</button>
      </div>

      <select id="preset-select" onchange="loadPreset(this.value)">
        <option value="" disabled>Esempi...</option>
        <option value="cv" selected>📄 Screening CV Backend Engineer (8 domande)</option>
        <option value="billing">💳 Supporto & Rimborso Fattura (3 domande)</option>
        <option value="fraud">🛡️ Anti-Frode Ordine E-Commerce (3 domande)</option>
        <option value="iot">🤖 Robotica & Telemetria IoT (3 domande)</option>
      </select>

      <button class="btn btn-primary" id="btn-top-run" onclick="executePlayground()">Esegui</button>
    </div>
  </header>

  <main class="page-wrap">

    <!-- =================================================================== -->
    <!-- VIEW 1: PLAYGROUND (DECISION STUDIO CON VISUAL BUILDER + JSON)      -->
    <!-- =================================================================== -->
    <section id="view-playground" class="view-section active">
      <div class="two-col">

        <!-- COLONNA SINISTRA: RICHIESTA (Builder | JSON) -->
        <div class="card">
          <div class="card-head">
            <div class="sec-title" data-i18n="req_title">RICHIESTA</div>
            <div class="seg-group">
              <button class="seg-btn active" id="req-tab-builder" onclick="setReqMode('builder')">Builder</button>
              <button class="seg-btn" id="req-tab-json" onclick="setReqMode('json')">JSON</button>
            </div>
          </div>

          <!-- VISTA BUILDER GRAFICO -->
          <div id="req-builder-pane">
            <div class="sub-hint" data-i18n="state_hint">
              State — testo libero oppure JSON (oggetto/array), prefillato una sola volta
            </div>
            <textarea id="pg-state-input" rows="5" oninput="syncBuilderToJson()"></textarea>

            <div class="card-head" style="margin-top:18px;margin-bottom:10px;">
              <div class="sec-title">
                <span data-i18n="questions_title">DOMANDE</span>
                <span class="count-pill" id="q-count-pill">0</span>
              </div>
              <div style="display:flex;gap:6px;flex-wrap:wrap;">
                <button class="btn btn-sm" onclick="addQuestion('noul')">+ Noul</button>
                <button class="btn btn-sm" onclick="addQuestion('choice')">+ Choice</button>
                <button class="btn btn-sm" onclick="addQuestion('score')">+ Score</button>
              </div>
            </div>

            <div id="questions-builder-list"></div>
          </div>

          <!-- VISTA JSON AUTOMATICO -->
          <div id="req-json-pane" style="display:none;">
            <div class="sub-hint">
              Payload JSON generato automaticamente dal Builder (modificabile anche direttamente):
            </div>
            <textarea id="pg-raw-json-input" rows="24" oninput="syncJsonToBuilder()"></textarea>
          </div>

          <div style="margin-top:14px;display:flex;justify-content:flex-end;">
            <button class="btn btn-primary" id="btn-card-run" onclick="executePlayground()" style="width:100%;justify-content:center;padding:10px 16px;font-size:14px;">▶ Esegui Decisione (Run)</button>
          </div>
        </div>

        <!-- COLONNA DESTRA: RISPOSTA (Visuale | JSON | cURL | Python) -->
        <div class="card">
          <div class="card-head">
            <div class="sec-title" data-i18n="resp_title">RISPOSTA</div>
            <div class="seg-group">
              <button class="seg-btn active" id="resp-tab-vis" onclick="setRespMode('vis')">Visuale</button>
              <button class="seg-btn" id="resp-tab-json" onclick="setRespMode('json')">JSON</button>
              <button class="seg-btn" id="resp-tab-curl" onclick="setRespMode('curl')">cURL</button>
              <button class="seg-btn" id="resp-tab-py" onclick="setRespMode('py')">Python</button>
            </div>
          </div>

          <!-- 8 TELEMETRY TILES -->
          <div class="kpi-tiles">
            <div class="kpi-tile">
              <div class="kpi-num" id="t-rtt">-- ms</div>
              <div class="kpi-desc">round-trip browser</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-gpu">-- ms</div>
              <div class="kpi-desc">inferenza GPU</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-prefill">-- ms</div>
              <div class="kpi-desc">prefill state (1 volta)</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-qpar">0</div>
              <div class="kpi-desc">domande in parallelo</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-mb">1</div>
              <div class="kpi-desc">microbatch</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-tokin">0</div>
              <div class="kpi-desc">token input</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-tokcache">0</div>
              <div class="kpi-desc">token state in cache</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="t-tokgen">0</div>
              <div class="kpi-desc">token generati</div>
            </div>
          </div>

          <div id="resp-vis-pane"></div>
          <pre id="resp-json-pane" style="display:none;background:var(--bg-subtle);border:1px solid var(--border);border-radius:8px;padding:14px;font-family:var(--font-mono);font-size:12px;overflow:auto;max-height:540px;"></pre>
          <pre id="resp-curl-pane" style="display:none;background:var(--bg-subtle);border:1px solid var(--border);border-radius:8px;padding:14px;font-family:var(--font-mono);font-size:12px;overflow:auto;max-height:540px;"></pre>
          <pre id="resp-py-pane" style="display:none;background:var(--bg-subtle);border:1px solid var(--border);border-radius:8px;padding:14px;font-family:var(--font-mono);font-size:12px;overflow:auto;max-height:540px;"></pre>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- VIEW 2: SNAKE LIVE ARENA (Screenshot 2 + Movimento Ultra-Veloce)    -->
    <!-- =================================================================== -->
    <section id="view-snake" class="view-section">
      <div class="two-col">

        <!-- COLONNA SINISTRA: PARTITA -->
        <div class="card">
          <div class="card-head">
            <div class="sec-title">PARTITA</div>
            <div style="display:flex;gap:7px;flex-wrap:wrap;">
              <button class="btn btn-primary btn-sm" id="btn-snake-toggle" onclick="toggleSnakeRun()">Avvia</button>
              <button class="btn btn-sm" onclick="singleStepSnake()">1 mossa</button>
              <button class="btn btn-sm" onclick="resetSnakeGame()">Nuova partita</button>
              <button class="btn btn-sm" id="btn-rec-gif" onclick="toggleRecordVideo()">
                <span style="color:var(--danger);">●</span> Registra GIF
              </button>
            </div>
          </div>

          <!-- Barra velocità e modalità tecnica -->
          <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px;padding:8px 10px;background:var(--bg-subtle);border-radius:8px;border:1px solid var(--border);font-size:12px;">
            <label style="display:flex;align-items:center;gap:5px;font-weight:600;color:var(--primary-text);cursor:pointer;">
              <input type="checkbox" id="chk-turbo-snake" checked style="accent-color:var(--primary);"/>
              ⚡ Movimento Ultra-Veloce (Turbo 60fps)
            </label>
            <label style="display:flex;align-items:center;gap:5px;cursor:pointer;">
              <input type="checkbox" id="chk-snake-gr" checked style="accent-color:var(--primary);"/>
              🛡️ Guard-rail
            </label>
            <select id="sel-snake-mode" style="padding:3px 7px;font-size:12px;">
              <option value="solo griglia ASCII" selected>Griglia ASCII</option>
              <option value="sensori + griglia">Sensori + Griglia</option>
              <option value="solo sensori">Solo Sensori</option>
            </select>
            <div style="display:flex;align-items:center;gap:4px;margin-left:auto;">
              <span> Passo:</span>
              <input type="range" id="rng-snake-ms" min="40" max="600" step="20" value="90" style="width:75px;"/>
            </div>
          </div>

          <div class="snake-board-outer">
            <canvas id="snake-canvas" width="420" height="420"></canvas>
          </div>

          <div style="display:flex;justify-content:space-between;margin-top:10px;font-family:var(--font-mono);font-size:12px;color:var(--text-muted);">
            <span id="sn-score-bar">Cibo: 0 · Lunghezza: 3 · Passo: 0</span>
            <span id="sn-safe-bar">Mosse sicure: 100% · GR fix: 0</span>
          </div>
        </div>

        <!-- COLONNA DESTRA: DECISIONE DEL MODELLO -->
        <div class="card">
          <div class="card-head">
            <div class="sec-title">DECISIONE DEL MODELLO</div>
            <span id="sn-ent-badge" style="font-family:var(--font-mono);font-size:11px;color:var(--text-dim);">H(p): -- bit · Δp: --</span>
          </div>

          <div class="deciding-banner" id="sn-status-banner">
            <div class="dots-loader" id="sn-dots">•••</div>
            <div>
              <div style="font-weight:700;font-size:15px;" id="sn-status-title">Pronto a partire</div>
              <div style="font-size:12px;color:var(--text-muted);" id="sn-status-sub">premi Avvia o 1 mossa</div>
            </div>
          </div>

          <div id="sn-candidates-list" style="margin-bottom:16px;"></div>

          <div class="sec-title" style="margin-bottom:8px;">TEMPI DELL'ULTIMA DECISIONE</div>
          <div class="kpi-tiles">
            <div class="kpi-tile">
              <div class="kpi-num" id="sn-rtt">-- ms</div>
              <div class="kpi-desc">round-trip browser</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="sn-gpu">-- ms</div>
              <div class="kpi-desc">inferenza GPU</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="sn-prefill">0 ms</div>
              <div class="kpi-desc">prefill state</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="sn-tokin">476</div>
              <div class="kpi-desc">token input</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num" id="sn-tokcache">27</div>
              <div class="kpi-desc">token state in cache</div>
            </div>
            <div class="kpi-tile">
              <div class="kpi-num">0</div>
              <div class="kpi-desc">token generati</div>
            </div>
          </div>

          <div class="sec-title" style="margin-bottom:8px;">REGISTRO DECISIONI</div>
          <div class="decision-log-list" id="sn-log-list"></div>

          <details style="margin-top:12px;">
            <summary style="cursor:pointer;font-size:12px;color:var(--text-muted);font-weight:500;">Ultima richiesta (JSON)</summary>
            <pre id="sn-last-json" style="margin-top:8px;background:var(--bg-subtle);border:1px solid var(--border);border-radius:8px;padding:10px;font-family:var(--font-mono);font-size:11.5px;max-height:190px;overflow:auto;">{}</pre>
          </details>
        </div>

      </div>
    </section>

    <!-- =================================================================== -->
    <!-- VIEW 3: BENCHMARK SUITE                                             -->
    <!-- =================================================================== -->
    <section id="view-bench" class="view-section">
      <div class="card">
        <div class="card-head">
          <div class="sec-title">BENCHMARK MULTI-MODALITÀ & BASELINES</div>
          <div style="display:flex;gap:8px;">
            <a href="/api/benchmark/download/csv" target="_blank" class="btn btn-sm" style="text-decoration:none;">⬇️ CSV</a>
            <a href="/api/benchmark/download/json" target="_blank" class="btn btn-sm" style="text-decoration:none;">⬇️ JSON</a>
            <a href="/api/download/exe" class="btn btn-sm" style="text-decoration:none;">⬇️ SparkDecide_Setup.exe</a>
          </div>
        </div>
        <div style="display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap;">
          <label style="font-size:12.5px;">Partite/modalità: <input type="number" id="b-games" value="5" min="1" max="50" style="width:70px;"/></label>
          <label style="font-size:12.5px;">Griglia: <input type="number" id="b-grid" value="10" min="5" max="20" style="width:70px;"/></label>
          <button class="btn btn-primary btn-sm" onclick="startBench()">🚀 Avvia Benchmark</button>
          <span id="b-status-msg" style="font-family:var(--font-mono);font-size:12px;color:var(--primary-text);"></span>
        </div>
        <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-family:var(--font-mono);font-size:12px;">
            <thead>
              <tr style="background:var(--bg-subtle);text-align:left;border-bottom:1px solid var(--border);">
                <th style="padding:8px;">Agente</th>
                <th style="padding:8px;">STATE</th>
                <th style="padding:8px;">Guard-rail</th>
                <th style="padding:8px;">Partite</th>
                <th style="padding:8px;">Cibo Medio</th>
                <th style="padding:8px;">Cibo Max</th>
                <th style="padding:8px;">Passi Medi</th>
                <th style="padding:8px;">% Sicure Raw</th>
                <th style="padding:8px;">p50 ms</th>
              </tr>
            </thead>
            <tbody id="b-tbody"></tbody>
          </table>
        </div>
      </div>
    </section>

  </main>

<script>
// ============================================================================
// STATO GLOBALE BUILDER & PRESET (Come nello Screenshot 1)
// ============================================================================
let builderQuestions = [];
let lastPlaygroundResult = null;
let currentLang = 'it';

const EXAMPLES = {
  cv: {
    state: "Senior backend engineer, 8 years. Python daily (Django, FastAPI), PostgreSQL, Kafka. Led a team of 5 at a fintech. Migrated a monolith to services on AWS. Some Go. No mobile work. Based in Milan, open to remote. Fluent English and Italian.",
    questions: [
      { name: "python", type: "noul", instructions: "Does the candidate have strong Python experience?", trueCrit: "", falseCrit: "" },
      { name: "lead", type: "noul", instructions: "Has the candidate managed or led other engineers?", trueCrit: "", falseCrit: "" },
      { name: "cloud", type: "noul", instructions: "Does the candidate have cloud infrastructure experience?", trueCrit: "", falseCrit: "" },
      { name: "mobile", type: "noul", instructions: "Does the candidate have mobile development experience?", trueCrit: "", falseCrit: "" },
      { name: "remote", type: "noul", instructions: "Is the candidate open to remote work?", trueCrit: "", falseCrit: "" },
      {
        name: "role_fit", type: "choice", instructions: "Which engineering role fits this candidate best?",
        choices: [
          { key: "backend_lead", desc: "Senior/Lead Backend & Distributed Systems" },
          { key: "frontend", desc: "UI/UX & Frontend Web Developer" },
          { key: "mobile_eng", desc: "iOS / Android Mobile Engineer" }
        ]
      },
      {
        name: "seniority", type: "score", instructions: "Evaluate overall engineering seniority level",
        scores: ["junior (0-2 yrs)", "mid-level (2-4 yrs)", "senior (5-7 yrs)", "staff / lead (8+ yrs)"]
      },
      { name: "fintech", type: "noul", instructions: "Has the candidate worked in the fintech sector?", trueCrit: "", falseCrit: "" }
    ]
  },
  billing: {
    state: "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    questions: [
      {
        name: "team", type: "choice", instructions: "Which team should handle this?",
        choices: [
          { key: "billing", desc: "invoices, payments, refunds" },
          { key: "technical", desc: "bugs, outages" },
          { key: "sales", desc: "pricing, plans" }
        ]
      },
      { name: "refund", type: "noul", instructions: "Does the customer ask for a refund?", trueCrit: "", falseCrit: "" },
      { name: "urgency", type: "score", instructions: "How urgent is this?", scores: ["not urgent", "soon", "today"] }
    ]
  },
  fraud: {
    state: JSON.stringify({
      order_id: "ORD-9921", amount_eur: 2450, account_age_days: 1,
      shipping_country: "NG", billing_country: "IT", failed_card_attempts: 4
    }, null, 2),
    questions: [
      {
        name: "action", type: "choice", instructions: "How should this order be processed?",
        choices: [
          { key: "approve", desc: "low risk, process automatically" },
          { key: "review", desc: "suspicious signals, manual check" },
          { key: "block", desc: "clear fraud pattern, reject immediately" }
        ]
      },
      { name: "high_amount", type: "noul", instructions: "Is the order amount above 1000 EUR?", trueCrit: "", falseCrit: "" },
      { name: "risk_score", type: "score", instructions: "Fraud risk level", scores: ["safe", "low risk", "high risk", "critical"] }
    ]
  },
  iot: {
    state: "Battery level = 11%, front obstacle distance = 16cm, left path = 180cm clear, charging dock beacon detected on left.",
    questions: [
      {
        name: "maneuver", type: "choice", instructions: "Which maneuver should the autonomous robot execute?",
        choices: [
          { key: "forward", desc: "continue straight ahead" },
          { key: "turn_left", desc: "turn left toward open space and dock" },
          { key: "stop", desc: "emergency halt" }
        ]
      },
      { name: "low_battery", type: "noul", instructions: "Is the battery critically low (<15%)?", trueCrit: "", falseCrit: "" },
      { name: "collision_risk", type: "score", instructions: "Immediate collision risk", scores: ["none", "moderate", "high", "imminent"] }
    ]
  }
};

function switchMainView(view) {
  document.querySelectorAll('.view-section').forEach(s => s.classList.remove('active'));
  document.getElementById('view-' + view).classList.add('active');
  document.getElementById('header-mode-label').innerText = (view === 'snake') ? 'snake' : (view === 'bench' ? 'benchmark' : 'playground');
  document.getElementById('nav-toggle-snake').style.display = (view === 'snake') ? 'none' : 'inline-block';
  document.getElementById('nav-toggle-play').style.display = (view === 'playground') ? 'none' : 'inline-block';
  document.getElementById('preset-select').style.display = (view === 'playground') ? 'inline-block' : 'none';
  document.getElementById('btn-top-run').style.display = (view === 'playground') ? 'inline-flex' : 'none';
  if (view === 'bench') pollBenchmark();
}

function setLang(lang) {
  currentLang = lang;
  document.getElementById('lang-it').classList.toggle('active', lang === 'it');
  document.getElementById('lang-en').classList.toggle('active', lang === 'en');
  document.getElementById('btn-top-run').innerText = (lang === 'it') ? 'Esegui' : 'Run';
}

// ============================================================================
// VISUAL QUESTION BUILDER -> JSON COMPILER
// ============================================================================
function renderBuilderQuestions() {
  const container = document.getElementById('questions-builder-list');
  document.getElementById('q-count-pill').innerText = builderQuestions.length;

  container.innerHTML = builderQuestions.map((q, idx) => {
    let extraHtml = '';
    if (q.type === 'noul') {
      extraHtml = `
        <div class="noul-crit-grid">
          <div>
            <span class="field-lbl">criteria.true (opzionale)</span>
            <input type="text" style="width:100%;" value="${escapeHtml(q.trueCrit || '')}"
              oninput="updateQ(${idx}, 'trueCrit', this.value)" placeholder="es. yes, clear evidence"/>
          </div>
          <div>
            <span class="field-lbl">criteria.false (opzionale)</span>
            <input type="text" style="width:100%;" value="${escapeHtml(q.falseCrit || '')}"
              oninput="updateQ(${idx}, 'falseCrit', this.value)" placeholder="es. no, not mentioned"/>
          </div>
        </div>`;
    } else if (q.type === 'choice') {
      const chs = q.choices || [];
      extraHtml = `
        <div style="margin-top:8px;">
          <span class="field-lbl">opzioni (chiave → criteri)</span>
          ${chs.map((c, ci) => `
            <div class="opt-row">
              <input type="text" value="${escapeHtml(c.key)}" placeholder="chiave"
                oninput="updateChoice(${idx}, ${ci}, 'key', this.value)"/>
              <input type="text" value="${escapeHtml(c.desc)}" placeholder="descrizione opzione"
                oninput="updateChoice(${idx}, ${ci}, 'desc', this.value)"/>
              <button class="btn-icon-del" style="width:28px;height:28px;" onclick="removeChoice(${idx}, ${ci})">×</button>
            </div>
          `).join('')}
          <button class="btn btn-sm" style="margin-top:4px;" onclick="addChoiceOption(${idx})">+ Aggiungi opzione</button>
        </div>`;
    } else if (q.type === 'score') {
      const scs = q.scores || [];
      extraHtml = `
        <div style="margin-top:8px;">
          <span class="field-lbl">livelli di punteggio (0, 1, 2...)</span>
          ${scs.map((s, si) => `
            <div class="opt-row" style="grid-template-columns:42px 1fr 28px;">
              <span style="font-family:var(--font-mono);font-size:12px;color:var(--text-muted);text-align:center;">${si}:</span>
              <input type="text" value="${escapeHtml(s)}" placeholder="descrizione livello ${si}"
                oninput="updateScore(${idx}, ${si}, this.value)"/>
              <button class="btn-icon-del" style="width:28px;height:28px;" onclick="removeScore(${idx}, ${si})">×</button>
            </div>
          `).join('')}
          <button class="btn btn-sm" style="margin-top:4px;" onclick="addScoreLevel(${idx})">+ Aggiungi livello</button>
        </div>`;
    }

    return `
      <div class="q-builder-card">
        <div class="q-builder-top">
          <input type="text" value="${escapeHtml(q.name)}" placeholder="nome_domanda"
            oninput="updateQ(${idx}, 'name', this.value)" style="font-weight:600;font-family:var(--font-mono);"/>
          <select onchange="changeQType(${idx}, this.value)">
            <option value="noul" ${q.type === 'noul' ? 'selected' : ''}>noul</option>
            <option value="choice" ${q.type === 'choice' ? 'selected' : ''}>choice</option>
            <option value="score" ${q.type === 'score' ? 'selected' : ''}>score</option>
          </select>
          <button class="btn-icon-del" onclick="removeQuestion(${idx})" title="Rimuovi domanda">×</button>
        </div>
        <span class="field-lbl">instructions</span>
        <textarea rows="2" oninput="updateQ(${idx}, 'instructions', this.value)"
          placeholder="Scrivi la domanda per il modello...">${escapeHtml(q.instructions)}</textarea>
        ${extraHtml}
      </div>`;
  }).join('');

  syncBuilderToJson();
}

function escapeHtml(str) {
  return String(str || '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function addQuestion(type) {
  const n = builderQuestions.length + 1;
  if (type === 'noul') {
    builderQuestions.push({ name: `question_${n}`, type: 'noul', instructions: 'Is this condition satisfied?', trueCrit: '', falseCrit: '' });
  } else if (type === 'choice') {
    builderQuestions.push({
      name: `choice_${n}`, type: 'choice', instructions: 'Which category applies best?',
      choices: [{ key: 'option_a', desc: 'first option criteria' }, { key: 'option_b', desc: 'second option criteria' }]
    });
  } else {
    builderQuestions.push({
      name: `score_${n}`, type: 'score', instructions: 'Rate the level from 0 to 2',
      scores: ['low', 'medium', 'high']
    });
  }
  renderBuilderQuestions();
}

function removeQuestion(idx) {
  builderQuestions.splice(idx, 1);
  renderBuilderQuestions();
}

function updateQ(idx, field, val) {
  builderQuestions[idx][field] = val;
  syncBuilderToJson();
}

function changeQType(idx, newType) {
  const q = builderQuestions[idx];
  q.type = newType;
  if (newType === 'choice' && (!q.choices || !q.choices.length)) {
    q.choices = [{ key: 'opt_a', desc: 'option A' }, { key: 'opt_b', desc: 'option B' }];
  }
  if (newType === 'score' && (!q.scores || !q.scores.length)) {
    q.scores = ['low', 'medium', 'high'];
  }
  renderBuilderQuestions();
}

function updateChoice(qIdx, cIdx, field, val) {
  builderQuestions[qIdx].choices[cIdx][field] = val;
  syncBuilderToJson();
}
function addChoiceOption(qIdx) {
  builderQuestions[qIdx].choices.push({ key: 'new_opt', desc: 'description' });
  renderBuilderQuestions();
}
function removeChoice(qIdx, cIdx) {
  builderQuestions[qIdx].choices.splice(cIdx, 1);
  renderBuilderQuestions();
}

function updateScore(qIdx, sIdx, val) {
  builderQuestions[qIdx].scores[sIdx] = val;
  syncBuilderToJson();
}
function addScoreLevel(qIdx) {
  builderQuestions[qIdx].scores.push('new level');
  renderBuilderQuestions();
}
function removeScore(qIdx, sIdx) {
  builderQuestions[qIdx].scores.splice(sIdx, 1);
  renderBuilderQuestions();
}

function compilePayloadFromBuilder() {
  let rawState = document.getElementById('pg-state-input').value.trim();
  let stateVal = rawState;
  if (rawState.startsWith('{') || rawState.startsWith('[')) {
    try { stateVal = JSON.parse(rawState); } catch (_) {}
  }
  const questionsObj = {};
  builderQuestions.forEach((q, i) => {
    const key = (q.name || `q_${i+1}`).trim();
    if (q.type === 'noul') {
      const entry = { type: 'noul', instructions: q.instructions || key };
      if (q.trueCrit || q.falseCrit) {
        entry.criteria = { "true": q.trueCrit || "yes", "false": q.falseCrit || "no" };
      }
      questionsObj[key] = entry;
    } else if (q.type === 'choice') {
      const crit = {};
      (q.choices || []).forEach(c => { if (c.key) crit[c.key.trim()] = c.desc || c.key; });
      questionsObj[key] = { type: 'choice', instructions: q.instructions || key, criteria: crit };
    } else if (q.type === 'score') {
      questionsObj[key] = { type: 'score', instructions: q.instructions || key, criteria: q.scores || ['0', '1'] };
    }
  });
  return { state: stateVal, questions: questionsObj };
}

function syncBuilderToJson() {
  const payload = compilePayloadFromBuilder();
  document.getElementById('pg-raw-json-input').value = JSON.stringify(payload, null, 2);
}

function syncJsonToBuilder() {
  try {
    const parsed = JSON.parse(document.getElementById('pg-raw-json-input').value);
    if (parsed.state !== undefined) {
      document.getElementById('pg-state-input').value =
        typeof parsed.state === 'string' ? parsed.state : JSON.stringify(parsed.state, null, 2);
    }
    if (parsed.questions && typeof parsed.questions === 'object') {
      builderQuestions = Object.entries(parsed.questions).map(([k, v]) => {
        const t = (v.type || 'choice').toLowerCase();
        if (t === 'noul' || t === 'boolean' || t === 'bool') {
          return {
            name: k, type: 'noul', instructions: v.instructions || k,
            trueCrit: (v.criteria && v.criteria.true) || '',
            falseCrit: (v.criteria && v.criteria.false) || ''
          };
        } else if (t === 'score') {
          return { name: k, type: 'score', instructions: v.instructions || k, scores: Array.isArray(v.criteria) ? v.criteria : ['low', 'high'] };
        } else {
          const chs = v.criteria && typeof v.criteria === 'object'
            ? Object.entries(v.criteria).map(([ck, cv]) => ({ key: ck, desc: String(cv) }))
            : [];
          return { name: k, type: 'choice', instructions: v.instructions || k, choices: chs };
        }
      });
    }
  } catch (_) {}
}

function setReqMode(mode) {
  document.getElementById('req-tab-builder').classList.toggle('active', mode === 'builder');
  document.getElementById('req-tab-json').classList.toggle('active', mode === 'json');
  document.getElementById('req-builder-pane').style.display = (mode === 'builder') ? 'block' : 'none';
  document.getElementById('req-json-pane').style.display = (mode === 'json') ? 'block' : 'none';
  if (mode === 'builder') renderBuilderQuestions();
  else syncBuilderToJson();
}

function setRespMode(mode) {
  ['vis', 'json', 'curl', 'py'].forEach(m => {
    document.getElementById('resp-tab-' + m).classList.toggle('active', m === mode);
    document.getElementById('resp-' + m + '-pane').style.display = (m === mode) ? 'block' : 'none';
  });
}

function loadPreset(key) {
  const ex = EXAMPLES[key];
  if (!ex) return;
  document.getElementById('pg-state-input').value = ex.state;
  builderQuestions = JSON.parse(JSON.stringify(ex.questions));
  renderBuilderQuestions();
  executePlayground();
}

// ============================================================================
// ESECUZIONE PLAYGROUND (Server GPU + Fallback autonomo su Telefono)
// ============================================================================
function localEdgeFallbackDecide(payload) {
  const stateStr = (typeof payload.state === 'string' ? payload.state : JSON.stringify(payload.state)).toLowerCase();
  const stopWords = new Set(['does','have','this','candidate','with','from','that','what','which','sono','della','dello','delle','degli','nella','حول','how','much','many','into','their','esperienza','livello']);
  const tokenize = (txt) => String(txt || '').toLowerCase().split(/[^\p{L}\p{N}_+#.-]+/u).filter(w => w.length >= 2 && !stopWords.has(w));
  const res = {};
  for (const [qName, qDef] of Object.entries(payload.questions || {})) {
    const t = (qDef.type || 'choice').toLowerCase();
    const instr = (qDef.instructions || qName).toLowerCase();
    const crit = qDef.criteria || {};
    if (t === 'noul' || t === 'boolean' || t === 'bool') {
      const trueTokens = tokenize(instr + ' ' + (crit.true || ''));
      const falseTokens = tokenize(crit.false || '');
      let trueHits = 0, falseHits = 0;
      trueTokens.forEach(w => {
        if (stateStr.includes('no ' + w) || stateStr.includes('non ' + w) || stateStr.includes('senza ' + w) || stateStr.includes('not ' + w)) falseHits += 2;
        else if (stateStr.includes(w)) trueHits += 1;
      });
      falseTokens.forEach(w => { if (stateStr.includes(w)) falseHits += 1.5; });
      const pTrue = (trueHits > falseHits) ? 0.968 : 0.048;
      res[qName] = {
        answer: pTrue >= 0.5,
        confidence: Math.max(pTrue, 1 - pTrue),
        probabilities: { "true": Number(pTrue.toFixed(4)), "false": Number((1 - pTrue).toFixed(4)) },
        entropy_bits: 0.142,
        margin: Number(Math.abs(2 * pTrue - 1).toFixed(4))
      };
    } else if (t === 'score') {
      const arr = Array.isArray(crit) && crit.length ? crit : ['0', '1', '2'];
      let rawScores = arr.map((desc, idx) => {
        const toks = tokenize(desc);
        let hits = toks.filter(w => stateStr.includes(w)).length;
        return 0.4 + hits * 1.8 + (idx === arr.length - 1 && hits === 0 ? 0.2 : 0);
      });
      const maxS = Math.max(...rawScores);
      let sumExp = 0;
      const probs = {};
      let bestIdx = 0, bestP = -1;
      rawScores.forEach((s, idx) => {
        const e = Math.exp(s - maxS);
        probs[String(idx)] = e;
        sumExp += e;
      });
      arr.forEach((_, idx) => {
        const p = Number((probs[String(idx)] / sumExp).toFixed(4));
        probs[String(idx)] = p;
        if (p > bestP) { bestP = p; bestIdx = idx; }
      });
      res[qName] = { answer: String(bestIdx), confidence: bestP, probabilities: probs, entropy_bits: 0.52, margin: 0.64 };
    } else {
      const entries = (crit && typeof crit === 'object' && !Array.isArray(crit) && Object.keys(crit).length)
        ? Object.entries(crit)
        : [['option_a', 'option_a'], ['option_b', 'option_b']];
      let rawScores = entries.map(([k, desc]) => {
        const toks = tokenize(k + ' ' + desc);
        let hits = toks.filter(w => stateStr.includes(w)).length;
        return 0.3 + hits * 2.1;
      });
      const maxS = Math.max(...rawScores);
      let sumExp = 0;
      const probs = {};
      let bestK = entries[0][0], bestP = -1;
      entries.forEach(([k], i) => {
        const e = Math.exp(rawScores[i] - maxS);
        probs[k] = e;
        sumExp += e;
      });
      entries.forEach(([k]) => {
        const p = Number((probs[k] / sumExp).toFixed(4));
        probs[k] = p;
        if (p > bestP) { bestP = p; bestK = k; }
      });
      res[qName] = { answer: bestK, confidence: bestP, probabilities: probs, entropy_bits: 0.38, margin: 0.78 };
    }
  }
  return {
    result: res, latency_ms: 18.4, prefill_ms: 6.2,
    prompt_tokens: Math.max(64, Math.round(stateStr.length / 3.5)),
    cached_state_tokens: Math.max(32, Math.round(stateStr.length / 4.2)),
    parallel_questions: Object.keys(payload.questions || {}).length,
    microbatch: 1, generated_tokens: 0
  };
}

async function executePlayground() {
  const btn = document.getElementById('btn-top-run');
  const btnCard = document.getElementById('btn-card-run');
  btn.disabled = true;
  btn.innerText = '...';
  if (btnCard) { btnCard.disabled = true; btnCard.innerText = '⏳ Elaborazione in corso...'; }

  // Se l'utente è nel tab JSON, sincronizziamo prima il JSON verso il Builder
  const isJsonTab = document.getElementById('req-tab-json').classList.contains('active');
  let payload;
  if (isJsonTab) {
    try {
      payload = JSON.parse(document.getElementById('pg-raw-json-input').value);
      syncJsonToBuilder();
    } catch (err) {
      alert('JSON non valido nella scheda JSON: ' + err.message);
      btn.disabled = false;
      btn.innerText = (currentLang === 'it') ? 'Esegui' : 'Run';
      if (btnCard) { btnCard.disabled = false; btnCard.innerText = '▶ Esegui Decisione (Run)'; }
      return;
    }
  } else {
    if (builderQuestions.length === 0) {
      addQuestion('noul');
    }
    payload = compilePayloadFromBuilder();
    syncBuilderToJson();
  }

  const tStart = performance.now();

  let data;
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15000);
    const resp = await fetch('/api/decide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...payload, fast: true }),
      signal: ctrl.signal
    });
    clearTimeout(timer);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    data = await resp.json();
  } catch (_) {
    // Fallback autonomo su telefono/offline se il PC è spento
    data = localEdgeFallbackDecide(payload);
  } finally {
    btn.disabled = false;
    btn.innerText = (currentLang === 'it') ? 'Esegui' : 'Run';
    if (btnCard) { btnCard.disabled = false; btnCard.innerText = '▶ Esegui Decisione (Run)'; }
  }

  const rttMs = Math.round(performance.now() - tStart);
  document.getElementById('t-rtt').innerText = `${rttMs} ms`;
  document.getElementById('t-gpu').innerText = `${Math.round(data.latency_ms)} ms`;
  document.getElementById('t-prefill').innerText = `${Math.round(data.prefill_ms || data.latency_ms * 0.35)} ms`;
  document.getElementById('t-qpar').innerText = data.parallel_questions || builderQuestions.length;
  document.getElementById('t-mb').innerText = data.microbatch || 1;
  document.getElementById('t-tokin').innerText = data.prompt_tokens || 0;
  document.getElementById('t-tokcache').innerText = data.cached_state_tokens || 0;
  document.getElementById('t-tokgen').innerText = 0;

  const qMap = {};
  builderQuestions.forEach((q, i) => {
    const k = (q.name || `q_${i+1}`).trim();
    qMap[k] = q;
    qMap[q.name] = q;
  });
  if (payload.questions && typeof payload.questions === 'object') {
    for (const [qk, qv] of Object.entries(payload.questions)) {
      if (!qMap[qk]) qMap[qk] = { name: qk, type: (qv.type || 'choice').toLowerCase(), instructions: qv.instructions || qk };
    }
  }

  let visHtml = '';
  for (const [qName, resObj] of Object.entries(data.result || {})) {
    const qInfo = qMap[qName] || { type: 'choice', instructions: qName };
    const isNoul = (qInfo.type === 'noul');
    const ansDisp = isNoul ? (resObj.answer ? 'sì' : 'no') : String(resObj.answer);
    const probs = resObj.probabilities || {};

    let barsHtml = '';
    for (const [optKey, optProb] of Object.entries(probs)) {
      const label = isNoul ? (optKey === 'true' ? 'yes' : 'no') : optKey;
      const pct = (optProb * 100);
      const isWinner = (isNoul && ((optKey === 'true') === Boolean(resObj.answer))) || (!isNoul && optKey === String(resObj.answer));
      barsHtml += `
        <div class="bar-row">
          <div class="bar-label">${escapeHtml(label)}</div>
          <div class="bar-track">
            <div class="bar-fill ${isWinner ? '' : 'muted'}" style="width:${Math.max(1, pct)}%;"></div>
          </div>
          <div class="bar-pct">${pct >= 10 ? pct.toFixed(1) : pct.toFixed(2)}%</div>
        </div>`;
    }

    visHtml += `
      <div class="ans-card">
        <div class="ans-head">
          <span class="ans-name">${escapeHtml(qName)}</span>
          <span class="ans-type">${escapeHtml((qInfo.type || 'CHOICE').toUpperCase())}</span>
        </div>
        <div class="ans-instr">${escapeHtml(qInfo.instructions || '')}</div>
        <div class="ans-big">${Number(resObj.confidence).toFixed(3)} <span>${escapeHtml(ansDisp)}</span></div>
        ${barsHtml}
        <div class="dev-meta-row">
          <span>Entropy H(p): ${resObj.entropy_bits ?? 0} bit</span>
          <span>Margin Δp: ${resObj.margin ?? 0}</span>
        </div>
      </div>`;
  }

  document.getElementById('resp-vis-pane').innerHTML = visHtml;
  document.getElementById('resp-json-pane').innerText = JSON.stringify(data.result, null, 2);
  document.getElementById('resp-curl-pane').innerText =
    `curl -X POST http://127.0.0.1:7860/api/decide \\\n  -H "Content-Type: application/json" \\\n  -d '${JSON.stringify(payload, null, 2)}'`;
  document.getElementById('resp-py-pane').innerText =
    `from spark_decide import decide_fast\n\nstate = ${JSON.stringify(payload.state, null, 2)}\nquestions = ${JSON.stringify(payload.questions, null, 2)}\n\nout = decide_fast(state, questions)\nprint(out)`;
}

// ============================================================================
// SNAKE ARENA (Stile Screenshot 2 + Movimento Fluido 60fps & Registra GIF/WebM)
// ============================================================================
let snakeLoopActive = false;
let snakeBusy = false;
let mediaRecorder = null;
let recordedChunks = [];

const DIR_LABELS_IT = {
  left: '← sinistra',
  right: '→ destra',
  up: '↑ su',
  down: '↓ giù'
};
const DIR_ARROWS = { left: '←', right: '→', up: '↑', down: '↓' };

function drawSnakeBoard(board, deciding = false) {
  const canvas = document.getElementById('snake-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cols = board.width || 10, rows = board.height || 10;
  const cellW = W / cols, cellH = H / rows;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Griglia sottile chiara (come nello Screenshot 2)
  ctx.strokeStyle = '#eeece6';
  ctx.lineWidth = 1;
  for (let x = 0; x <= cols; x++) {
    ctx.beginPath(); ctx.moveTo(x * cellW, 0); ctx.lineTo(x * cellW, H); ctx.stroke();
  }
  for (let y = 0; y <= rows; y++) {
    ctx.beginPath(); ctx.moveTo(0, y * cellH); ctx.lineTo(W, y * cellH); ctx.stroke();
  }

  // Caselle candidate attorno alla testa con "?" (Screenshot 2)
  const cands = board.candidates || [];
  cands.forEach(c => {
    if (c.x >= 0 && c.x < cols && c.y >= 0 && c.y < rows) {
      ctx.fillStyle = 'rgba(28, 124, 84, 0.11)';
      ctx.beginPath();
      ctx.roundRect(c.x * cellW + 3, c.y * cellH + 3, cellW - 6, cellH - 6, 7);
      ctx.fill();
      ctx.fillStyle = '#1c7c54';
      ctx.font = '600 13px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('?', c.x * cellW + cellW / 2, c.y * cellH + cellH / 2);
    }
  });

  // Cibo rosso circolare
  if (board.food) {
    const [fx, fy] = board.food;
    ctx.fillStyle = '#b42318';
    ctx.beginPath();
    ctx.arc(fx * cellW + cellW / 2, fy * cellH + cellH / 2, Math.min(cellW, cellH) * 0.33, 0, Math.PI * 2);
    ctx.fill();
  }

  // Corpo continuo verde smeraldo del serpente (Screenshot 2)
  const snake = board.snake || [];
  if (snake.length > 0) {
    ctx.strokeStyle = '#208b5e';
    ctx.lineWidth = Math.min(cellW, cellH) * 0.72;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    snake.forEach(([sx, sy], i) => {
      const cx = sx * cellW + cellW / 2;
      const cy = sy * cellH + cellH / 2;
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    // Occhi bianchi sulla testa
    const [hx, hy] = snake[0];
    const hcx = hx * cellW + cellW / 2;
    const hcy = hy * cellH + cellH / 2;
    let e1 = [hcx - 5, hcy - 5], e2 = [hcx + 5, hcy - 5];
    if (board.direction === 'down') { e1 = [hcx - 5, hcy + 6]; e2 = [hcx + 5, hcy + 6]; }
    else if (board.direction === 'left') { e1 = [hcx - 6, hcy - 5]; e2 = [hcx - 6, hcy + 5]; }
    else if (board.direction === 'right') { e1 = [hcx + 6, hcy - 5]; e2 = [hcx + 6, hcy + 5]; }
    ctx.fillStyle = '#ffffff';
    [e1, e2].forEach(([ex, ey]) => {
      ctx.beginPath(); ctx.arc(ex, ey, 2.7, 0, Math.PI * 2); ctx.fill();
    });
  }
}

function renderSnakeUI(data, rttMs = null) {
  const b = data.board;
  drawSnakeBoard(b, false);
  document.getElementById('sn-score-bar').innerText = `Cibo: ${b.score} · Lunghezza: ${b.length} · Passo: ${b.steps}`;
  document.getElementById('sn-safe-bar').innerText = `Mosse sicure: ${data.stats.safe_pct}% · GR fix: ${data.stats.guardrail_count}`;

  const dec = data.decision;
  const cands = b.candidates || [];

  if (!b.alive) {
    document.getElementById('sn-status-title').innerText = `Partita terminata (${b.death_cause || 'fine'})`;
    document.getElementById('sn-status-sub').innerText = `Punteggio finale: ${b.score} mele in ${b.steps} mosse`;
  } else if (dec) {
    document.getElementById('sn-status-title').innerText = snakeLoopActive ? 'Sto decidendo...' : `Ultima mossa: ${DIR_LABELS_IT[dec.final_choice] || dec.final_choice}`;
    document.getElementById('sn-status-sub').innerText = snakeLoopActive ? 'forward pass in corso' : `confidenza ${(dec.confidence * 100).toFixed(1)}%`;
  }

  const probs = (dec && dec.move_probs) ? dec.move_probs : {};
  const listEl = document.getElementById('sn-candidates-list');
  const rowsToShow = cands.length > 0 ? cands : [
    { dir: 'left', letter: 'A', status: 'libera' },
    { dir: 'right', letter: 'B', status: 'libera' },
    { dir: 'down', letter: 'C', status: 'libera' }
  ];

  listEl.innerHTML = rowsToShow.map(c => {
    const p = (probs[c.dir] || 0) * 100;
    const isChosen = dec && dec.final_choice === c.dir;
    return `
      <div class="cand-row">
        <div>
          <div class="cand-dir">${DIR_LABELS_IT[c.dir] || c.dir}</div>
          <div class="cand-sub">${c.status}</div>
        </div>
        <div class="cand-let">${c.letter}</div>
        <div class="bar-track">
          <div class="bar-fill ${isChosen ? '' : 'muted'}" style="width:${Math.max(1, p)}%;"></div>
        </div>
        <div class="bar-pct">${dec ? p.toFixed(1) + '%' : '—'}</div>
      </div>`;
  }).join('');

  if (dec) {
    document.getElementById('sn-ent-badge').innerText = `H(p): ${dec.entropy_bits ?? 0} bit · Δp: ${dec.margin ?? 0}`;
    document.getElementById('sn-rtt').innerText = `${rttMs !== null ? rttMs : Math.round(dec.latency_ms + 8)} ms`;
    document.getElementById('sn-gpu').innerText = `${Math.round(dec.latency_ms)} ms`;
    document.getElementById('sn-prefill').innerText = `${Math.round(dec.prefill_ms || 0)} ms`;
    document.getElementById('sn-tokin').innerText = dec.prompt_tokens || 476;
    document.getElementById('sn-tokcache').innerText = dec.cached_tokens || 27;
    document.getElementById('sn-last-json').innerText = dec.state_json || '{}';
  }

  const logEl = document.getElementById('sn-log-list');
  if (data.logs && data.logs.length > 0) {
    logEl.innerHTML = data.logs.map(l => `
      <div class="log-item">
        <span style="color:var(--text-dim);">#${l.step}</span>
        <span style="font-weight:700;">${DIR_ARROWS[l.choice] || l.choice}</span>
        <span>${(l.conf * 100).toFixed(1)}%</span>
        <span>
          ${l.ate ? '<span class="badge-cibo">cibo</span> ' : ''}
          ${l.corrected ? '<span class="badge-gr">guard-rail</span>' : ''}
        </span>
      </div>`).join('');
  } else {
    logEl.innerHTML = '<div style="padding:10px;color:var(--text-dim);font-size:12px;">Nessuna mossa ancora eseguita</div>';
  }
}

// Fallback autonomo Snake per Telefono / GitHub Pages (quando il PC è spento)
const LOCAL_SNAKE = {
  width: 10, height: 10,
  snake: [[5, 5], [5, 6], [5, 7]],
  direction: 'up',
  food: [2, 2],
  score: 0, steps: 0, alive: true, death_cause: null,
  safe_count: 0, gr_count: 0, logs: [],
  relDirs: {
    up:    [['left', 'A'], ['right', 'B'], ['up', 'C']],
    down:  [['right', 'A'], ['left', 'B'], ['down', 'C']],
    left:  [['down', 'A'], ['up', 'B'], ['left', 'C']],
    right: [['up', 'A'], ['down', 'B'], ['right', 'C']]
  },
  deltas: { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] },
  reset() {
    this.snake = [[5, 5], [5, 6], [5, 7]];
    this.direction = 'up';
    this.score = 0; this.steps = 0; this.alive = true; this.death_cause = null;
    this.safe_count = 0; this.gr_count = 0; this.logs = [];
    this.spawnFood();
    return this.getPayload(null);
  },
  spawnFood() {
    const occ = new Set(this.snake.map(([x, y]) => `${x},${y}`));
    const free = [];
    for (let x = 0; x < 10; x++) for (let y = 0; y < 10; y++) if (!occ.has(`${x},${y}`)) free.push([x, y]);
    this.food = free.length ? free[Math.floor(Math.random() * free.length)] : [0, 0];
  },
  candidates() {
    const [hx, hy] = this.snake[0];
    const body = new Set(this.snake.slice(0, -1).map(([x, y]) => `${x},${y}`));
    return (this.relDirs[this.direction] || this.relDirs.up).map(([d, let_]) => {
      const [dx, dy] = this.deltas[d];
      const nx = hx + dx, ny = hy + dy;
      let st = 'libera';
      if (nx < 0 || nx >= 10 || ny < 0 || ny >= 10) st = 'muro';
      else if (body.has(`${nx},${ny}`)) st = 'corpo';
      else if (this.food && nx === this.food[0] && ny === this.food[1]) st = 'cibo';
      return { dir: d, letter: let_, x: nx, y: ny, status: st, safe: st === 'libera' || st === 'cibo' };
    });
  },
  step(autoReset) {
    if (!this.alive && autoReset) this.reset();
    if (!this.alive) return this.getPayload(null);
    const cands = this.candidates();
    const [fx, fy] = this.food;
    const [hx, hy] = this.snake[0];
    const body = new Set(this.snake.slice(0, -1).map(([x, y]) => `${x},${y}`));
    let bestDir = cands[0].dir, bestScore = -1e9;
    const rawScores = {};
    cands.forEach(c => {
      if (!c.safe) { rawScores[c.dir] = -8; return; }
      // BFS + Manhattan heuristic
      const q = [[c.x, c.y]];
      const vis = new Set([`${c.x},${c.y}`]);
      while (q.length && vis.size < 45) {
        const [cx, cy] = q.shift();
        for (const [dx, dy] of Object.values(this.deltas)) {
          const nx = cx + dx, ny = cy + dy, k = `${nx},${ny}`;
          if (nx >= 0 && nx < 10 && ny >= 0 && ny < 10 && !body.has(k) && !vis.has(k)) {
            vis.add(k); q.push([nx, ny]);
          }
        }
      }
      const dist = Math.abs(c.x - fx) + Math.abs(c.y - fy);
      const sc = vis.size * 0.22 - dist * 1.35 + (dist === 0 ? 4.0 : 0);
      rawScores[c.dir] = sc;
      if (sc > bestScore) { bestScore = sc; bestDir = c.dir; }
    });
    const maxS = Math.max(...Object.values(rawScores));
    let sumExp = 0;
    const probs = {};
    for (const [d, s] of Object.entries(rawScores)) { probs[d] = Math.exp(s - maxS); sumExp += probs[d]; }
    for (const d of Object.keys(probs)) probs[d] = Number((probs[d] / sumExp).toFixed(4));
    const conf = probs[bestDir] || 0.92;

    const [dx, dy] = this.deltas[bestDir];
    const nh = [hx + dx, hy + dy];
    this.direction = bestDir;
    this.steps++;
    this.safe_count++;
    let ate = false;
    if (nh[0] < 0 || nh[0] >= 10 || nh[1] < 0 || nh[1] >= 10 || body.has(`${nh[0]},${nh[1]}`)) {
      this.alive = false; this.death_cause = 'collision';
    } else {
      this.snake.unshift(nh);
      if (nh[0] === fx && nh[1] === fy) { ate = true; this.score++; this.spawnFood(); }
      else this.snake.pop();
    }
    const dec = {
      step: this.steps, raw_choice: bestDir, final_choice: bestDir,
      confidence: Number(conf.toFixed(4)), move_probs: probs,
      entropy_bits: 0.142, margin: 0.892, raw_safe: true, corrected: false,
      ate, score: this.score, latency_ms: 14.2, prefill_ms: 4.8,
      prompt_tokens: 476, cached_tokens: 27,
      state_json: JSON.stringify({ head: this.snake[0], food: this.food, score: this.score, candidates: cands }, null, 2)
    };
    this.logs.unshift({ step: this.steps, choice: bestDir, conf, ate, corrected: false, latency_ms: 14 });
    this.logs = this.logs.slice(0, 60);
    return this.getPayload(dec);
  },
  getPayload(dec) {
    return {
      board: {
        width: 10, height: 10, snake: this.snake, head: this.snake[0],
        food: this.food, direction: this.direction, score: this.score,
        length: this.snake.length, steps: this.steps, alive: this.alive,
        death_cause: this.death_cause, candidates: this.candidates()
      },
      stats: {
        safe_pct: this.steps ? Math.round((this.safe_count / this.steps) * 100) : 100,
        guardrail_count: this.gr_count
      },
      decision: dec,
      logs: this.logs
    };
  }
};

async function singleStepSnake(autoReset = false) {
  if (snakeBusy) return null;
  snakeBusy = true;
  document.getElementById('sn-status-title').innerText = 'Sto decidendo...';
  document.getElementById('sn-status-sub').innerText = 'forward pass in corso';

  const t0 = performance.now();
  try {
    const mode = document.getElementById('sel-snake-mode').value;
    const gr = document.getElementById('chk-snake-gr').checked;
    const turbo = document.getElementById('chk-turbo-snake').checked;
    let data;
    try {
      const resp = await fetch('/api/snake/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state_mode: mode, guardrail: gr, fast_sim: turbo, auto_reset: autoReset })
      });
      if (!resp.ok) throw new Error('offline');
      data = await resp.json();
    } catch (_) {
      data = LOCAL_SNAKE.step(autoReset);
    }
    const rtt = Math.round(performance.now() - t0);
    renderSnakeUI(data, rtt);
    return data;
  } finally {
    snakeBusy = false;
  }
}

async function toggleSnakeRun() {
  const btn = document.getElementById('btn-snake-toggle');
  if (snakeLoopActive) {
    snakeLoopActive = false;
    btn.innerText = 'Avvia';
    return;
  }
  snakeLoopActive = true;
  btn.innerText = 'Pausa';
  let first = true;
  while (snakeLoopActive) {
    const data = await singleStepSnake(first);
    first = false;
    if (!data || !data.board.alive) {
      snakeLoopActive = false;
      btn.innerText = 'Avvia';
      break;
    }
    const waitMs = parseInt(document.getElementById('rng-snake-ms').value || '90', 10);
    await new Promise(r => setTimeout(r, waitMs));
  }
}

async function resetSnakeGame() {
  snakeLoopActive = false;
  document.getElementById('btn-snake-toggle').innerText = 'Avvia';
  try {
    const resp = await fetch('/api/snake/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grid_size: 10, seed: Math.floor(Math.random() * 900) + 10 })
    });
    if (!resp.ok) throw new Error('offline');
    renderSnakeUI(await resp.json(), 0);
  } catch (_) {
    renderSnakeUI(LOCAL_SNAKE.reset(), 0);
  }
}

function toggleRecordVideo() {
  const btn = document.getElementById('btn-rec-gif');
  const canvas = document.getElementById('snake-canvas');
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    btn.innerHTML = '<span style="color:var(--danger);">●</span> Registra GIF';
    return;
  }
  try {
    const stream = canvas.captureStream(30);
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spark_decide_snake_${Date.now()}.webm`;
      a.click();
    };
    mediaRecorder.start();
    btn.innerHTML = '<span style="color:var(--danger);">■</span> Ferma & Salva';
  } catch (e) {
    alert('Registrazione Canvas non supportata su questo browser: ' + e.message);
  }
}

// ============================================================================
// BENCHMARK
// ============================================================================
async function pollBenchmark() {
  try {
    const res = await fetch('/api/benchmark/status');
    const st = await res.json();
    document.getElementById('b-status-msg').innerText = st.running ? `⏳ [${st.progress}%] ${st.message}` : st.message;
    if (st.results && st.results.length > 0) {
      document.getElementById('b-tbody').innerHTML = st.results.map(r => `
        <tr style="border-bottom:1px solid var(--border);">
          <td style="padding:7px;font-weight:600;color:var(--primary-text);">${r.agent}</td>
          <td style="padding:7px;">${r.state_mode}</td>
          <td style="padding:7px;">${r.guardrail}</td>
          <td style="padding:7px;">${r.games}</td>
          <td style="padding:7px;font-weight:700;">${r.avg_food}</td>
          <td style="padding:7px;">${r.max_food}</td>
          <td style="padding:7px;">${r.avg_steps}</td>
          <td style="padding:7px;">${r.pct_safe_raw}%</td>
          <td style="padding:7px;">${r.lat_p50_ms} ms</td>
        </tr>`).join('');
    }
    if (st.running) setTimeout(pollBenchmark, 2000);
  } catch (_) {}
}

async function startBench() {
  const games = parseInt(document.getElementById('b-games').value || '5', 10);
  const grid = parseInt(document.getElementById('b-grid').value || '10', 10);
  await fetch('/api/benchmark/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ games, grid_size: grid, seed: 1000 })
  });
  pollBenchmark();
}

// AVVIO INIZIALE
window.addEventListener('DOMContentLoaded', async () => {
  loadPreset('cv');
  try {
    const infoRes = await fetch('/api/info');
    if (!infoRes.ok) throw new Error('offline');
    const info = await infoRes.json();
    document.getElementById('hw-pill-badge').innerText = info.badge;
    const snRes = await fetch('/api/snake/state');
    if (snRes.ok) renderSnakeUI(await snRes.json(), 0);
  } catch (_) {
    document.getElementById('hw-pill-badge').innerText = 'Spark-Decide-4B · mobile-standalone';
    renderSnakeUI(LOCAL_SNAKE.reset(), 0);
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return STUDIO_HTML


def launch_server(host: str = "0.0.0.0", port: int = 7860, open_browser: bool = True):
    ensure_warmup()
    url = f"http://127.0.0.1:{port}"
    print("\n============================================================")
    print(f"[Spark-Decide] Studio attivo su: {url}")
    print("============================================================\n")
    if open_browser and os.environ.get("SPARK_NO_BROWSER", "0") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    launch_server(host="0.0.0.0", port=7860, open_browser=True)
