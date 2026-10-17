import json
import os
import time
import threading
import webbrowser
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from snake_env import SnakeEnv, SNAKE_QUESTIONS
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
    print("[Spark-Studio] Esecuzione warmup iniziale su Spark-Decide (5 chiamate)...")
    warm_env = SnakeEnv(width=10, height=10, seed=42)
    warm_state = warm_env.get_state("grid_only")
    for _ in range(5):
        spark_decide.decide_fast(warm_state, SNAKE_QUESTIONS)
    spark_decide.sync()
    _WARMED_UP = True
    print("[Spark-Studio] Warmup completato!")


def count_prompt_tokens(state_obj, questions_obj):
    global _CACHED_Q_LEN
    if questions_obj is SNAKE_QUESTIONS:
        if _CACHED_Q_LEN is None:
            qs = spark_decide._parse(SNAKE_QUESTIONS)
            q_ids = []
            for j, q in enumerate(qs):
                q_ids += spark_decide.enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n")
                for i, d in enumerate(q["descs"]):
                    q_ids += spark_decide.enc(f"{spark_decide.LET[i]}) {d}\n")
                q_ids += spark_decide.enc("Answer:")
            _CACHED_Q_LEN = len(spark_decide._PREFIX_IDS) + len(spark_decide._T_IDS) + len(q_ids) + len(spark_decide._P_IDS)
        base_len = _CACHED_Q_LEN
    else:
        qs = spark_decide._parse(questions_obj)
        q_ids = []
        for j, q in enumerate(qs):
            q_ids += spark_decide.enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n")
            for i, d in enumerate(q["descs"]):
                q_ids += spark_decide.enc(f"{spark_decide.LET[i]}) {d}\n")
            q_ids += spark_decide.enc("Answer:")
        base_len = len(spark_decide._PREFIX_IDS) + len(spark_decide._T_IDS) + len(q_ids) + len(spark_decide._P_IDS)
    state_str = state_obj if isinstance(state_obj, str) else json.dumps(state_obj, ensure_ascii=False)
    s_len = min(len(spark_decide.enc(state_str)), max(32, spark_decide.MAXLEN - base_len))
    return base_len + s_len


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

    def step(self, state_mode_ui: str = "sensori + griglia", guardrail: bool = False):
        if not self.env.alive:
            return self.serialize()

        mode_key = STATE_MODE_MAP.get(state_mode_ui, "sensors_and_grid")
        state_obj = self.env.get_state(mode=mode_key)
        prompt_tokens = count_prompt_tokens(state_obj, SNAKE_QUESTIONS)

        with _MODEL_LOCK:
            spark_decide.sync()
            t0 = time.perf_counter()
            res = spark_decide.decide_fast(state_obj, SNAKE_QUESTIONS)
            spark_decide.sync()
            lat_ms = (time.perf_counter() - t0) * 1000.0

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
        state_summary = f"H({hx},{hy}) dir={self.env.direction[0].upper()} F({fx},{fy}) dx={fx-hx},dy={fy-hy}"

        step_out = self.env.step(final_choice)

        self.latencies.append(round(lat_ms, 1))
        self.confidences.append(round(conf, 4))
        self.cum_time_s += (lat_ms / 1000.0)
        if was_raw_safe:
            self.raw_safe_count += 1
        if corrected:
            self.guardrail_count += 1

        flags = []
        if corrected:
            flags.append(f"GR:{raw_choice.upper()}->{final_choice.upper()}")
        if step_out["ate"]:
            flags.append("MANGIATO")
        if not step_out["alive"]:
            if step_out["death_cause"] == "illegal_180":
                flags.append("ILLEGAL 180°")
            else:
                flags.append(f"MORTE:{step_out['death_cause'].upper()}")
        flag_str = " | ".join(flags) if flags else "OK"

        log_entry = {
            "step": self.env.steps,
            "summary": state_summary,
            "choice": final_choice.upper() + (f" (raw:{raw_choice.upper()})" if corrected else ""),
            "conf": round(conf, 3),
            "ms": round(lat_ms, 1),
            "flag": flag_str,
            "corrected": corrected,
            "ate": step_out["ate"],
            "alive": step_out["alive"],
        }
        self.logs.insert(0, log_entry)
        self.logs = self.logs[:50]

        self.last_decision = {
            "move_probs": move_probs,
            "raw_choice": raw_choice,
            "final_choice": final_choice,
            "confidence": conf,
            "guardrail_corrected": corrected,
            "danger_prob_true": danger_prob_true,
            "danger_ans": danger_ans,
            "risk_probs": risk_probs,
            "risk_ans": risk_ans,
            "latency_ms": round(lat_ms, 1),
            "prompt_tokens": prompt_tokens,
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
    gpu_name = "CPU"
    if spark_decide.DEV == "cuda":
        props = spark_decide.torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = round(props.total_memory / (1024 ** 3), 2)
    return {
        "device": spark_decide.DEV.upper(),
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "fast_4bit": getattr(spark_decide, "_use_4bit_fast", False),
        "model": "Spark-Decide (4B + DecisionHead)",
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
    state_mode = payload.get("state_mode", "sensori + griglia")
    guardrail = bool(payload.get("guardrail", False))
    if not SESSION.env.alive and payload.get("auto_reset", False):
        SESSION.reset(width=SESSION.env.width, height=SESSION.env.height, seed=SESSION.env.seed)
    return SESSION.step(state_mode_ui=state_mode, guardrail=guardrail)


@app.post("/api/decide")
def api_decide(payload: Dict[str, Any] = Body(...)):
    state_in = payload.get("state", "")
    questions_in = payload.get("questions", {})
    use_fast = bool(payload.get("fast", True))
    if not isinstance(questions_in, dict) or not questions_in:
        return JSONResponse({"error": "Il campo 'questions' deve essere un dizionario non vuoto."}, status_code=400)

    tokens = count_prompt_tokens(state_in, questions_in)
    fn = spark_decide.decide_fast if use_fast else spark_decide.decide
    with _MODEL_LOCK:
        spark_decide.sync()
        t0 = time.perf_counter()
        out = fn(state_in, questions_in)
        spark_decide.sync()
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "result": out,
        "latency_ms": lat_ms,
        "prompt_tokens": tokens,
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

            with _MODEL_LOCK:
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


SUPABASE_STUDIO_HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Spark-Decide Studio — Supabase Edition</title>
<link rel="icon" type="image/png" href="/logo.png"/>
<style>
  :root {
    --bg-app: #121212;
    --bg-sidebar: #161616;
    --bg-panel: #1c1c1c;
    --bg-card: #222222;
    --bg-input: #181818;
    --border: #2e2e2e;
    --border-light: #3a3a3a;
    --brand: #3ecf8e;
    --brand-hover: #24b47e;
    --brand-dim: rgba(62, 207, 142, 0.14);
    --text-main: #ededed;
    --text-muted: #9b9b9b;
    --text-dim: #686868;
    --danger: #f04438;
    --warn: #f79009;
    --info: #38bdf8;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-app);
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 13px;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }
  /* SIDEBAR */
  .sidebar {
    width: 245px;
    background: var(--bg-sidebar);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }
  .brand-header {
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .brand-logo {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background: var(--brand-dim);
    border: 1px solid var(--brand);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--brand);
    font-weight: 800;
    font-size: 15px;
    box-shadow: 0 0 14px rgba(62, 207, 142, 0.25);
  }
  .brand-title {
    font-weight: 700;
    font-size: 14px;
    letter-spacing: -0.2px;
    color: #fff;
  }
  .brand-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }
  .nav-section {
    padding: 14px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .nav-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-dim);
    padding: 4px 10px 6px;
    font-weight: 600;
  }
  .nav-btn {
    background: transparent;
    border: 1px solid transparent;
    color: var(--text-muted);
    text-align: left;
    padding: 9px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.15s ease;
  }
  .nav-btn:hover {
    background: var(--bg-panel);
    color: var(--text-main);
  }
  .nav-btn.active {
    background: var(--brand-dim);
    color: var(--brand);
    border-color: rgba(62, 207, 142, 0.3);
    font-weight: 600;
  }
  .sidebar-footer {
    margin-top: auto;
    padding: 14px;
    border-top: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-muted);
    background: #141414;
  }
  .hw-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--brand);
    margin-top: 6px;
    width: 100%;
  }
  .dot-live {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--brand);
    box-shadow: 0 0 8px var(--brand);
  }

  /* MAIN AREA */
  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .topbar {
    height: 52px;
    background: var(--bg-sidebar);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
    flex-shrink: 0;
  }
  .topbar-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .breadcrumb {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
  }
  .breadcrumb strong { color: var(--text-main); }
  .badge {
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-family: var(--font-mono);
    font-weight: 600;
    background: var(--brand-dim);
    color: var(--brand);
    border: 1px solid rgba(62, 207, 142, 0.35);
  }
  .content {
    flex: 1;
    overflow-y: auto;
    padding: 20px 22px;
  }
  .tab-view { display: none; }
  .tab-view.active { display: block; }

  /* SUPABASE CARDS & CONTROLS */
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
  }
  .panel-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .btn {
    background: var(--bg-card);
    color: var(--text-main);
    border: 1px solid var(--border-light);
    padding: 7px 13px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.12s;
  }
  .btn:hover { background: #2c2c2c; border-color: #4a4a4a; }
  .btn-primary {
    background: var(--brand);
    color: #0b2218;
    border-color: var(--brand);
    font-weight: 600;
  }
  .btn-primary:hover { background: var(--brand-hover); border-color: var(--brand-hover); }
  .btn-danger {
    background: rgba(240, 68, 56, 0.15);
    color: #fda29b;
    border-color: rgba(240, 68, 56, 0.4);
  }
  .btn-danger:hover { background: rgba(240, 68, 56, 0.25); }

  input, select, textarea {
    background: var(--bg-input);
    color: var(--text-main);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 12px;
    font-family: var(--font-sans);
    outline: none;
  }
  input:focus, select:focus, textarea:focus {
    border-color: var(--brand);
    box-shadow: 0 0 0 2px rgba(62, 207, 142, 0.15);
  }
  textarea { font-family: var(--font-mono); line-height: 1.45; }

  /* GRID LAYOUTS */
  .snake-grid {
    display: grid;
    grid-template-columns: 470px 1fr;
    gap: 16px;
    align-items: start;
  }
  @media (max-width: 1100px) {
    .snake-grid { grid-template-columns: 1fr; }
  }
  .controls-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
    margin-bottom: 12px;
  }
  .field-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .field-group label {
    font-size: 11px;
    color: var(--text-muted);
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 12px;
  }
  .kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }
  .kpi-label {
    font-size: 10px;
    text-transform: uppercase;
    color: var(--text-muted);
    letter-spacing: 0.4px;
  }
  .kpi-val {
    font-size: 18px;
    font-weight: 700;
    font-family: var(--font-mono);
    margin-top: 3px;
    color: var(--text-main);
  }
  .kpi-val.emerald { color: var(--brand); }
  .kpi-val.info { color: var(--info); }
  .kpi-val.warn { color: var(--warn); }

  /* PROBABILITY BARS */
  .prob-row {
    margin-bottom: 8px;
  }
  .prob-head {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 12px;
    margin-bottom: 3px;
  }
  .prob-track {
    height: 9px;
    background: #141414;
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
  }
  .prob-fill {
    height: 100%;
    background: var(--brand);
    transition: width 0.15s ease;
  }

  /* TABLES */
  .sb-table-wrap {
    max-height: 250px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  table.sb-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: var(--font-mono);
  }
  table.sb-table th {
    background: #181818;
    color: var(--text-muted);
    text-align: left;
    padding: 8px 10px;
    position: sticky;
    top: 0;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
  }
  table.sb-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #242424;
    color: var(--text-main);
  }
  table.sb-table tr:hover td { background: #222; }

  /* STUDIO PLAYGROUND */
  .studio-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 1050px) {
    .studio-grid { grid-template-columns: 1fr; }
  }
  .q-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 10px;
  }
</style>
</head>
<body>

  <!-- SIDEBAR SUPABASE -->
  <aside class="sidebar">
    <div class="brand-header">
      <img src="/logo.png" alt="Spark-Decide Mascot" style="width:36px;height:36px;border-radius:8px;border:1px solid var(--brand);box-shadow:0 0 12px rgba(62,207,142,0.3);object-fit:cover;"/>
      <div>
        <div class="brand-title">Spark-Decide</div>
        <div class="brand-sub">spark-decide • studio</div>
      </div>
    </div>

    <div class="nav-section">
      <div class="nav-label">Playground & Test</div>
      <button class="nav-btn active" onclick="switchTab('snake', this)">
        <span>🐍</span> Snake Live Arena
      </button>
      <button class="nav-btn" onclick="switchTab('studio', this)">
        <span>⚡</span> Usa il Modello (Studio)
      </button>
      <button class="nav-btn" onclick="switchTab('bench', this)">
        <span>📊</span> Benchmark Suite
      </button>
      <div class="nav-label" style="margin-top:12px;">Sviluppatori</div>
      <button class="nav-btn" onclick="switchTab('api', this)">
        <span>🔌</span> API & Integrazione
      </button>
    </div>

    <div class="sidebar-footer">
      <div>Motore d'inferenza locale</div>
      <div class="hw-pill">
        <span class="dot-live"></span>
        <span id="hw-badge-text">CUDA 4-bit NF4</span>
      </div>
      <div style="margin-top:6px;color:var(--text-dim);font-family:var(--font-mono);">
        KV-Prefix Cache: ATTIVA
      </div>
    </div>
  </aside>

  <!-- MAIN -->
  <div class="main">
    <header class="topbar">
      <div class="topbar-left">
        <span class="breadcrumb">local / <strong>spark-decide-4b</strong> / <span id="top-tab-name">snake-arena</span></span>
        <span class="badge" id="top-lat-badge">Pronto</span>
      </div>
      <div style="display:flex;gap:12px;align-items:center;">
        <span style="font-size:12px;color:var(--text-muted);font-family:var(--font-mono);" id="top-tokens-badge">Tokens: --</span>
        <a href="/api/download/exe" class="btn" style="text-decoration:none;padding:5px 10px;border-color:var(--brand);color:var(--brand);">⬇️ Scarica SparkDecide_Setup.exe</a>
      </div>
    </header>

    <div class="content">

      <!-- ==================== TAB 1: SNAKE LIVE ARENA ==================== -->
      <div id="tab-snake" class="tab-view active">
        <div class="snake-grid">
          <!-- Colonna Sinistra: Board + Controlli -->
          <div>
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">🎮 Controlli Partita & Agente</div>
                <span class="badge" id="game-status-pill">IN ATTESA</span>
              </div>

              <div class="controls-bar">
                <button class="btn btn-primary" id="btn-start" onclick="startSnake()">▶️ Start</button>
                <button class="btn" id="btn-pause" onclick="pauseSnake()">⏸️ Pausa</button>
                <button class="btn" id="btn-step" onclick="stepSnake()">⏭️ Step Singolo</button>
                <button class="btn btn-danger" onclick="resetSnake()">🔄 Reset</button>
              </div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;">
                <div class="field-group">
                  <label>Modalità STATE</label>
                  <select id="sel-state-mode">
                    <option value="solo griglia ASCII">⚡ solo griglia ASCII (~1.0s)</option>
                    <option value="sensori + griglia" selected>🎯 sensori + griglia (~1.8s)</option>
                    <option value="solo sensori">📡 solo sensori (~1.8s)</option>
                  </select>
                </div>
                <div class="field-group">
                  <label>Protezione Mosse Letali</label>
                  <label style="display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--bg-input);border:1px solid var(--border);border-radius:6px;cursor:pointer;color:var(--text-main);font-size:12px;">
                    <input type="checkbox" id="chk-guardrail" style="accent-color:var(--brand);"/>
                    🛡️ Guard-rail Attivo
                  </label>
                </div>
              </div>

              <div style="display:grid;grid-template-columns:1fr 90px 90px;gap:10px;margin-bottom:14px;">
                <div class="field-group">
                  <label>Ritardo extra tra mosse: <span id="lbl-delay">0</span> ms</label>
                  <input type="range" id="rng-delay" min="0" max="1000" step="50" value="0" oninput="document.getElementById('lbl-delay').innerText=this.value"/>
                </div>
                <div class="field-group">
                  <label>Griglia NxN</label>
                  <input type="number" id="inp-grid" value="10" min="5" max="20"/>
                </div>
                <div class="field-group">
                  <label>Seed</label>
                  <input type="number" id="inp-seed" value="42"/>
                </div>
              </div>

              <!-- CANVAS SNAKE -->
              <div style="display:flex;justify-content:center;background:#0d0d0d;padding:12px;border-radius:8px;border:1px solid var(--border);position:relative;">
                <canvas id="snake-canvas" width="416" height="416" style="border-radius:6px;"></canvas>
              </div>

              <!-- KPI CARDS -->
              <div class="kpi-grid">
                <div class="kpi-card">
                  <div class="kpi-label">Cibo / Score</div>
                  <div class="kpi-val emerald" id="kpi-score">0</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Lunghezza / Step</div>
                  <div class="kpi-val info" id="kpi-len">3 / 0</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">Ultima Latenza</div>
                  <div class="kpi-val" id="kpi-lat">-- ms</div>
                </div>
                <div class="kpi-card">
                  <div class="kpi-label">% Sicure (No GR)</div>
                  <div class="kpi-val warn" id="kpi-safe">100%</div>
                </div>
              </div>
              <div style="margin-top:8px;display:flex;justify-content:space-between;font-family:var(--font-mono);font-size:11px;color:var(--text-muted);">
                <span id="kpi-percentiles">p50: -- | p90: -- | max: -- ms</span>
                <span id="kpi-mps">0.00 mosse/s | GR fix: 0</span>
              </div>
            </div>
          </div>

          <!-- Colonna Destra: Probabilità, Danger/Risk, Grafici, Log -->
          <div>
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">🧠 DecisionHead Output in Tempo Reale</div>
                <span id="gr-alert-badge" style="display:none;background:rgba(247,144,9,0.18);color:var(--warn);border:1px solid var(--warn);padding:2px 8px;border-radius:4px;font-family:var(--font-mono);font-size:11px;">
                  🛡️ GUARD-RAIL INTERVENUTO
                </span>
              </div>

              <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:16px;">
                <div>
                  <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;font-weight:600;">
                    PROBABILITÀ DIREZIONE (`move`)
                  </div>
                  <div id="move-probs-container"></div>
                </div>
                <div>
                  <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;font-weight:600;">
                    VALUTAZIONE PERICOLO (`danger` & `risk`)
                  </div>
                  <div id="danger-risk-container"></div>
                </div>
              </div>
            </div>

            <!-- GRAFICI LIVE CANVAS -->
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">📈 Telemetria Live (Latenza ms & Confidence %)</div>
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                <div style="background:#141414;border:1px solid var(--border);border-radius:6px;padding:8px;">
                  <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;font-family:var(--font-mono);">Latenza per Mossa (ms)</div>
                  <canvas id="chart-lat" width="320" height="110" style="width:100%;height:110px;"></canvas>
                </div>
                <div style="background:#141414;border:1px solid var(--border);border-radius:6px;padding:8px;">
                  <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;font-family:var(--font-mono);">Confidence Modello (%)</div>
                  <canvas id="chart-conf" width="320" height="110" style="width:100%;height:110px;"></canvas>
                </div>
              </div>
            </div>

            <!-- LOG ULTIME 50 MOSSE + JSON STATE -->
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">📜 Log Scorrevole (Ultime 50 Mosse)</div>
                <button class="btn" style="padding:3px 8px;font-size:11px;" onclick="toggleStateJson()">🔍 Mostra/Nascondi STATE JSON</button>
              </div>
              <pre id="state-json-viewer" style="display:none;background:#111;border:1px solid var(--border);padding:10px;border-radius:6px;font-family:var(--font-mono);font-size:11px;color:var(--brand);max-height:180px;overflow:auto;margin-bottom:10px;">{}</pre>
              <div class="sb-table-wrap">
                <table class="sb-table">
                  <thead>
                    <tr>
                      <th>Step</th>
                      <th>Stato Sintetico</th>
                      <th>Scelta</th>
                      <th>Conf.</th>
                      <th>ms</th>
                      <th>Flag</th>
                    </tr>
                  </thead>
                  <tbody id="logs-tbody">
                    <tr><td colspan="6" style="color:var(--text-muted);text-align:center;">Premi Start o Step Singolo per iniziare</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== TAB 2: USA IL MODELLO (DECISION STUDIO) ==================== -->
      <div id="tab-studio" class="tab-view">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">⚡ Spark-Decide Studio — Usa il Modello su Qualsiasi Task</div>
            <div style="display:flex;gap:8px;align-items:center;">
              <span style="font-size:12px;color:var(--text-muted);">Preset rapido:</span>
              <select id="studio-preset" onchange="loadStudioPreset(this.value)">
                <option value="billing">💳 Supporto & Fatturazione</option>
                <option value="fraud">🛡️ Anti-Frode & Risk Triage Ordini</option>
                <option value="iot">🤖 Robotica / Smart IoT Decision</option>
                <option value="snake">🐍 Stato Snake (3 domande)</option>
              </select>
              <button class="btn btn-primary" id="btn-run-studio" onclick="runStudioDecide()">⚡ Esegui decide()</button>
            </div>
          </div>

          <div class="studio-grid">
            <div>
              <div class="field-group" style="margin-bottom:12px;">
                <label>STATE (Testo libero o oggetto JSON passato a <code>decide(state, questions)</code>)</label>
                <textarea id="studio-state" rows="8"></textarea>
              </div>
              <div class="field-group">
                <label>QUESTIONS (Dizionario JSON con domande <code>choice</code>, <code>noul</code>/<code>boolean</code>, <code>score</code>)</label>
                <textarea id="studio-questions" rows="14"></textarea>
              </div>
            </div>

            <div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-size:12px;font-weight:600;color:var(--text-main);">Risultati & Distribuzione Probabilità</span>
                <span class="badge" id="studio-lat-badge">In attesa</span>
              </div>
              <div id="studio-results-cards" style="margin-bottom:12px;"></div>
              <div class="field-group">
                <label>Output JSON grezzo restituito da <code>decide(state, questions)</code></label>
                <pre id="studio-raw-out" style="background:#111;border:1px solid var(--border);border-radius:6px;padding:12px;font-family:var(--font-mono);font-size:12px;color:var(--brand);max-height:240px;overflow:auto;">{}</pre>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ==================== TAB 3: BENCHMARK SUITE ==================== -->
      <div id="tab-bench" class="tab-view">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">📊 Benchmark Multi-Modalità + Baselines</div>
            <div style="display:flex;gap:8px;">
              <a href="/api/benchmark/download/csv" target="_blank" class="btn" style="text-decoration:none;">⬇️ Scarica CSV</a>
              <a href="/api/benchmark/download/json" target="_blank" class="btn" style="text-decoration:none;">⬇️ Scarica JSON</a>
            </div>
          </div>
          <div class="controls-bar" style="margin-bottom:14px;">
            <div class="field-group">
              <label>Partite per modalità</label>
              <input type="number" id="bench-games" value="5" min="1" max="50" style="width:110px;"/>
            </div>
            <div class="field-group">
              <label>Griglia NxN</label>
              <input type="number" id="bench-grid" value="10" min="5" max="20" style="width:90px;"/>
            </div>
            <div class="field-group">
              <label>Seed Base</label>
              <input type="number" id="bench-seed" value="1000" style="width:100px;"/>
            </div>
            <button class="btn btn-primary" onclick="startBenchmark()" style="margin-top:16px;">🚀 Avvia Benchmark</button>
            <span id="bench-msg" style="margin-top:16px;font-family:var(--font-mono);font-size:12px;color:var(--brand);"></span>
          </div>

          <div class="sb-table-wrap" style="max-height:460px;">
            <table class="sb-table">
              <thead>
                <tr>
                  <th>Agente</th>
                  <th>Modalità STATE</th>
                  <th>Guard-Rail</th>
                  <th>Partite</th>
                  <th>Cibo Medio</th>
                  <th>Cibo Max</th>
                  <th>Passi Medi</th>
                  <th>% Muro</th>
                  <th>% Corpo</th>
                  <th>% 180°</th>
                  <th>% Sicure Raw</th>
                  <th>Latenza p50 (ms)</th>
                  <th>Mosse/s</th>
                </tr>
              </thead>
              <tbody id="bench-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- ==================== TAB 4: API & INTEGRAZIONE ==================== -->
      <div id="tab-api" class="tab-view">
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">🔌 Usa Spark-Decide via Python o API HTTP in Localhost</div>
          </div>
          <p style="color:var(--text-muted);margin-bottom:12px;line-height:1.5;">
            Quando avvii <code>python spark_decide.py</code> (oppure <code>python snake_playground.py</code>), il server espone anche l'endpoint REST
            <code>POST http://127.0.0.1:7860/api/decide</code> oltre alla funzione Python <code>decide(state, questions)</code>.
          </p>
          <div class="studio-grid">
            <div class="field-group">
              <label>1. Import diretto in Python</label>
              <pre style="background:#111;border:1px solid var(--border);border-radius:6px;padding:14px;font-family:var(--font-mono);font-size:12px;color:#e2e8f0;overflow:auto;">from spark_decide import decide, decide_fast

state = "Hi, I was charged twice for invoice #4411. Please refund today."
questions = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages",
            "sales": "pricing, plans"
        }
    },
    "refund": {
        "type": "noul",
        "instructions": "Does the customer ask for a refund?"
    }
}

risultato = decide_fast(state, questions)
print(risultato)</pre>
            </div>
            <div class="field-group">
              <label>2. Chiamata HTTP REST API (cURL / JS / Python requests)</label>
              <pre style="background:#111;border:1px solid var(--border);border-radius:6px;padding:14px;font-family:var(--font-mono);font-size:12px;color:var(--brand);overflow:auto;">import requests

resp = requests.post("http://127.0.0.1:7860/api/decide", json={
    "state": "Temperature sensor = 94C, cooling fan = OFF",
    "questions": {
        "action": {
            "type": "choice",
            "instructions": "What action should the controller take?",
            "criteria": {
                "fan_on": "turn cooling fan on immediately",
                "shutdown": "emergency power off",
                "ignore": "keep normal operation"
            }
        },
        "overheating": {
            "type": "noul",
            "instructions": "Is the device overheating?"
        }
    }
})
print(resp.json())</pre>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>

<script>
let snakeRunning = false;
let stepInFlight = false;

const PRESETS = {
  billing: {
    state: "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
    questions: {
      team: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          billing: "invoices, payments, refunds",
          technical: "bugs, outages",
          sales: "pricing, plans"
        }
      },
      refund: {
        type: "noul",
        instructions: "Does the customer ask for a refund?"
      },
      urgency: {
        type: "score",
        instructions: "How urgent is this?",
        criteria: ["not urgent", "soon", "today"]
      }
    }
  },
  fraud: {
    state: {
      order_id: "ORD-9921",
      amount_eur: 2450,
      account_age_days: 1,
      shipping_country: "NG",
      billing_country: "IT",
      failed_card_attempts: 4
    },
    questions: {
      decision: {
        type: "choice",
        instructions: "How should this order be processed?",
        criteria: {
          approve: "low risk, process automatically",
          manual_review: "suspicious signals, hold for human check",
          block: "clear fraud pattern, reject immediately"
        }
      },
      high_value: {
        type: "noul",
        instructions: "Is the order amount above 1000 EUR?"
      },
      fraud_risk: {
        type: "score",
        instructions: "Rate the fraud risk level",
        criteria: ["low", "medium", "high", "critical"]
      }
    }
  },
  iot: {
    state: {
      battery_pct: 11,
      obstacle_front_cm: 18,
      obstacle_left_cm: 140,
      obstacle_right_cm: 95,
      dock_direction: "left"
    },
    questions: {
      maneuver: {
        type: "choice",
        instructions: "Which maneuver should the robot execute next?",
        criteria: {
          forward: "move straight ahead",
          turn_left: "turn left toward open space and dock",
          turn_right: "turn right",
          stop: "emergency halt"
        }
      },
      low_battery: {
        type: "noul",
        instructions: "Is the battery critically low (<15%)?"
      }
    }
  },
  snake: {
    state: {
      grid: [
        "############",
        "#..........#",
        "#..........#",
        "#......F...#",
        "#..........#",
        "#...ooH....#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "#..........#",
        "############"
      ],
      current_direction: "right",
      legend: "#=wall, H=head, o=body, F=food, .=empty"
    },
    questions: {
      move: {
        type: "choice",
        instructions: "Which direction should the snake move next to eat the food without dying?",
        criteria: {
          up: "move up (y-1)",
          down: "move down (y+1)",
          left: "move left (x-1)",
          right: "move right (x+1)"
        }
      },
      danger: {
        type: "noul",
        instructions: "Is the snake in immediate danger of dying next move?"
      },
      risk: {
        type: "score",
        instructions: "How risky is the current position?",
        criteria: ["safe", "some risk", "high risk", "critical"]
      }
    }
  }
};

function switchTab(name, btn) {
  document.querySelectorAll('.tab-view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  if (btn) btn.classList.add('active');
  document.getElementById('top-tab-name').innerText = name;
  if (name === 'bench') loadBenchmarkStatus();
}

function toggleStateJson() {
  const el = document.getElementById('state-json-viewer');
  el.style.display = (el.style.display === 'none') ? 'block' : 'none';
}

function drawBoard(board) {
  const canvas = document.getElementById('snake-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const cols = board.width, rows = board.height;
  const cellW = W / cols, cellH = H / rows;

  ctx.clearRect(0, 0, W, H);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      ctx.fillStyle = ((x + y) % 2 === 0) ? '#141414' : '#181818';
      ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
    }
  }

  if (board.food) {
    const [fx, fy] = board.food;
    const cx = fx * cellW + cellW / 2;
    const cy = fy * cellH + cellH / 2;
    const r = Math.min(cellW, cellH) * 0.36;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(240, 68, 56, 0.22)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = '#f04438';
    ctx.fill();
  }

  const snake = board.snake || [];
  snake.forEach(([sx, sy], idx) => {
    const pad = idx === 0 ? 2 : 4;
    const rx = sx * cellW + pad;
    const ry = sy * cellH + pad;
    const rw = cellW - pad * 2;
    const rh = cellH - pad * 2;
    if (idx === 0) {
      ctx.fillStyle = board.alive ? '#3ecf8e' : '#f04438';
    } else {
      const ratio = Math.max(0.35, 1 - (idx / (snake.length + 3)) * 0.6);
      ctx.fillStyle = `rgba(62, 207, 142, ${ratio})`;
    }
    ctx.beginPath();
    ctx.roundRect(rx, ry, rw, rh, idx === 0 ? 7 : 5);
    ctx.fill();
  });

  if (!board.alive) {
    ctx.fillStyle = 'rgba(12, 12, 12, 0.82)';
    ctx.fillRect(0, H / 2 - 42, W, 84);
    ctx.strokeStyle = '#f04438';
    ctx.strokeRect(10, H / 2 - 42, W - 20, 84);
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    const causes = {
      wall: '💥 COLLISIONE CON MURO',
      body: '🐍 COLLISIONE CON IL CORPO',
      illegal_180: '🚫 MOSSA ILLEGALE (INVERSIONE 180°)',
      starvation: '⏳ LIMITE MOSSE SENZA CIBO',
      won: '🏆 VITTORIA!'
    };
    ctx.fillText(causes[board.death_cause] || 'GAME OVER', W / 2, H / 2 - 6);
    ctx.fillStyle = '#3ecf8e';
    ctx.font = '13px JetBrains Mono, monospace';
    ctx.fillText(`Score: ${board.score} | Lunghezza: ${board.length} | Step: ${board.steps}`, W / 2, H / 2 + 22);
  }
}

function drawLineChart(canvasId, data, color, yMin, yMax) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#262626';
  ctx.beginPath();
  ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
  ctx.stroke();
  if (!data || data.length === 0) return;

  const minVal = (yMin !== null) ? yMin : Math.min(...data) * 0.9;
  const maxVal = (yMax !== null) ? yMax : Math.max(...data, minVal + 10) * 1.05;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (data.length === 1) ? W / 2 : (i / (data.length - 1)) * (W - 16) + 8;
    const norm = (v - minVal) / Math.max(1e-5, maxVal - minVal);
    const y = H - 10 - norm * (H - 20);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderSnakeState(data) {
  drawBoard(data.board);
  document.getElementById('kpi-score').innerText = data.board.score;
  document.getElementById('kpi-len').innerText = `${data.board.length} / ${data.board.steps}`;
  document.getElementById('kpi-lat').innerText = `${data.stats.last_lat_ms} ms`;
  document.getElementById('kpi-safe').innerText = `${data.stats.safe_pct}%`;
  document.getElementById('kpi-percentiles').innerText =
    `p50: ${data.stats.p50_ms} | p90: ${data.stats.p90_ms} | max: ${data.stats.max_ms} ms`;
  document.getElementById('kpi-mps').innerText =
    `${data.stats.mps} mosse/s | GR fix: ${data.stats.guardrail_count}`;

  const dec = data.decision;
  if (dec) {
    document.getElementById('top-lat-badge').innerText = `${dec.latency_ms} ms`;
    document.getElementById('top-tokens-badge').innerText = `Tokens: ${dec.prompt_tokens}`;
    document.getElementById('state-json-viewer').innerText = dec.state_json;
    document.getElementById('gr-alert-badge').style.display = dec.guardrail_corrected ? 'inline-block' : 'none';

    const dirs = ['up', 'down', 'left', 'right'];
    const icons = { up: '⬆️ UP', down: '⬇️ DOWN', left: '⬅️ LEFT', right: '➡️ RIGHT' };
    let moveHtml = '';
    dirs.forEach(d => {
      const p = (dec.move_probs[d] || 0) * 100;
      const isFinal = (dec.final_choice === d);
      const barCol = isFinal ? '#3ecf8e' : '#38bdf8';
      moveHtml += `
        <div class="prob-row">
          <div class="prob-head">
            <span style="color:${isFinal ? '#3ecf8e' : '#ededed'};font-weight:${isFinal ? '700' : '400'};">
              ${icons[d]} ${isFinal ? '✓' : ''}
            </span>
            <span>${p.toFixed(1)}%</span>
          </div>
          <div class="prob-track"><div class="prob-fill" style="width:${p}%;background:${barCol};"></div></div>
        </div>`;
    });
    document.getElementById('move-probs-container').innerHTML = moveHtml;

    const dPct = (dec.danger_prob_true * 100).toFixed(1);
    const dCol = dec.danger_ans ? '#f04438' : '#3ecf8e';
    let drHtml = `
      <div class="prob-row" style="margin-bottom:12px;">
        <div class="prob-head">
          <span>Danger: <strong style="color:${dCol}">${dec.danger_ans ? 'TRUE' : 'FALSE'}</strong></span>
          <span>P(true): ${dPct}%</span>
        </div>
        <div class="prob-track"><div class="prob-fill" style="width:${dPct}%;background:${dCol};"></div></div>
      </div>`;
    const rLabels = { "0": "0: safe", "1": "1: some risk", "2": "2: high risk", "3": "3: critical" };
    Object.keys(rLabels).forEach(k => {
      const rp = ((dec.risk_probs[k] || 0) * 100).toFixed(1);
      drHtml += `
        <div class="prob-row">
          <div class="prob-head"><span>${rLabels[k]}</span><span>${rp}%</span></div>
          <div class="prob-track"><div class="prob-fill" style="width:${rp}%;background:#f79009;"></div></div>
        </div>`;
    });
    document.getElementById('danger-risk-container').innerHTML = drHtml;
  }

  drawLineChart('chart-lat', data.series.latencies, '#38bdf8', 0, null);
  drawLineChart('chart-conf', data.series.confidences.map(c => c * 100), '#3ecf8e', 0, 100);

  const tbody = document.getElementById('logs-tbody');
  if (data.logs && data.logs.length > 0) {
    tbody.innerHTML = data.logs.map(l => `
      <tr>
        <td>${l.step}</td>
        <td style="color:var(--text-muted);">${l.summary}</td>
        <td style="color:var(--brand);font-weight:600;">${l.choice}</td>
        <td>${(l.conf * 100).toFixed(1)}%</td>
        <td>${l.ms}</td>
        <td>${l.flag}</td>
      </tr>`).join('');
  }
}

async function stepSnake(autoReset = false) {
  if (stepInFlight) return;
  stepInFlight = true;
  try {
    const mode = document.getElementById('sel-state-mode').value;
    const gr = document.getElementById('chk-guardrail').checked;
    const res = await fetch('/api/snake/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state_mode: mode, guardrail: gr, auto_reset: autoReset })
    });
    const data = await res.json();
    renderSnakeState(data);
    return data;
  } finally {
    stepInFlight = false;
  }
}

async function startSnake() {
  if (snakeRunning) return;
  snakeRunning = true;
  document.getElementById('game-status-pill').innerText = 'IN ESECUZIONE';
  let first = true;
  while (snakeRunning) {
    const data = await stepSnake(first);
    first = false;
    if (!data || !data.board.alive) {
      snakeRunning = false;
      document.getElementById('game-status-pill').innerText = 'GAME OVER';
      break;
    }
    const delay = parseInt(document.getElementById('rng-delay').value || '0', 10);
    if (delay > 0) await new Promise(r => setTimeout(r, delay));
  }
}

function pauseSnake() {
  snakeRunning = false;
  document.getElementById('game-status-pill').innerText = 'IN PAUSA';
}

async function resetSnake() {
  snakeRunning = false;
  document.getElementById('game-status-pill').innerText = 'PRONTO';
  const g = parseInt(document.getElementById('inp-grid').value || '10', 10);
  const s = parseInt(document.getElementById('inp-seed').value || '42', 10);
  const res = await fetch('/api/snake/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ grid_size: g, seed: s })
  });
  renderSnakeState(await res.json());
}

function loadStudioPreset(key) {
  const p = PRESETS[key];
  if (!p) return;
  document.getElementById('studio-state').value =
    typeof p.state === 'string' ? p.state : JSON.stringify(p.state, null, 2);
  document.getElementById('studio-questions').value = JSON.stringify(p.questions, null, 2);
}

async function runStudioDecide() {
  const btn = document.getElementById('btn-run-studio');
  btn.disabled = true;
  btn.innerText = '⏳ Calcolo in corso...';
  try {
    let rawState = document.getElementById('studio-state').value.trim();
    let parsedState = rawState;
    if (rawState.startsWith('{') || rawState.startsWith('[')) {
      try { parsedState = JSON.parse(rawState); } catch (_) {}
    }
    const parsedQuestions = JSON.parse(document.getElementById('studio-questions').value);
    const res = await fetch('/api/decide', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state: parsedState, questions: parsedQuestions, fast: true })
    });
    const data = await res.json();
    document.getElementById('studio-lat-badge').innerText = `${data.latency_ms} ms (${data.prompt_tokens} tok)`;
    document.getElementById('studio-raw-out').innerText = JSON.stringify(data.result, null, 2);

    let cardsHtml = '';
    for (const [qName, qVal] of Object.entries(data.result || {})) {
      let probsHtml = '';
      for (const [optKey, optProb] of Object.entries(qVal.probabilities || {})) {
        const pct = (optProb * 100).toFixed(1);
        probsHtml += `
          <div class="prob-row">
            <div class="prob-head"><span>${optKey}</span><span>${pct}%</span></div>
            <div class="prob-track"><div class="prob-fill" style="width:${pct}%;"></div></div>
          </div>`;
      }
      cardsHtml += `
        <div class="q-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <strong style="font-family:var(--font-mono);color:#fff;">${qName}</strong>
            <span class="badge">Risposta: ${String(qVal.answer)} (${(qVal.confidence * 100).toFixed(1)}%)</span>
          </div>
          ${probsHtml}
        </div>`;
    }
    document.getElementById('studio-results-cards').innerHTML = cardsHtml;
  } catch (e) {
    alert('Errore nel formato JSON di QUESTIONS: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerText = '⚡ Esegui decide()';
  }
}

async function loadBenchmarkStatus() {
  const res = await fetch('/api/benchmark/status');
  const st = await res.json();
  document.getElementById('bench-msg').innerText =
    st.running ? `⏳ [${st.progress}%] ${st.message}` : st.message;
  const tbody = document.getElementById('bench-tbody');
  if (st.results && st.results.length > 0) {
    tbody.innerHTML = st.results.map(r => `
      <tr>
        <td style="font-weight:600;color:var(--brand);">${r.agent}</td>
        <td>${r.state_mode}</td>
        <td>${r.guardrail}</td>
        <td>${r.games}</td>
        <td style="font-weight:700;">${r.avg_food}</td>
        <td>${r.max_food}</td>
        <td>${r.avg_steps}</td>
        <td>${r.pct_death_wall}%</td>
        <td>${r.pct_death_body}%</td>
        <td>${r.pct_death_illegal}%</td>
        <td>${r.pct_safe_raw}%</td>
        <td>${r.lat_p50_ms}</td>
        <td>${r.moves_per_sec}</td>
      </tr>`).join('');
  }
  if (st.running) setTimeout(loadBenchmarkStatus, 2000);
}

async function startBenchmark() {
  const games = parseInt(document.getElementById('bench-games').value || '5', 10);
  const grid = parseInt(document.getElementById('bench-grid').value || '10', 10);
  const seed = parseInt(document.getElementById('bench-seed').value || '1000', 10);
  await fetch('/api/benchmark/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ games, grid_size: grid, seed })
  });
  loadBenchmarkStatus();
}

window.addEventListener('DOMContentLoaded', async () => {
  loadStudioPreset('billing');
  const infoRes = await fetch('/api/info');
  const info = await infoRes.json();
  document.getElementById('hw-badge-text').innerText =
    `${info.gpu_name} (${info.fast_4bit ? '4-bit NF4' : info.device})`;
  const stRes = await fetch('/api/snake/state');
  renderSnakeState(await stRes.json());
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return SUPABASE_STUDIO_HTML


def launch_server(host: str = "0.0.0.0", port: int = 7860, open_browser: bool = True):
    ensure_warmup()
    url = f"http://127.0.0.1:{port}"
    print("\n============================================================")
    print(f"[Spark-Studio] Supabase Studio attivo su: {url}")
    print("============================================================\n")
    if open_browser and os.environ.get("SPARK_NO_BROWSER", "0") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    launch_server(host="0.0.0.0", port=7860, open_browser=True)
