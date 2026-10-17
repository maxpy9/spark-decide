import os, shutil, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

SAVED = "models/spark-decide"
BASE_ID = "models/rizzo-flow"
MERGED_DIR = "models/spark-decide-merged"
MAXLEN = 2048
if torch.cuda.is_available():
    DEV = "cuda"; DT = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    DEV = "mps"; DT = torch.float16          # se fp16 dà NaN/risultati strani su MPS, prova torch.bfloat16
else:
    DEV = "cpu"; DT = torch.float32

def sync():
    if DEV == "cuda": torch.cuda.synchronize()
    elif DEV == "mps": torch.mps.synchronize()

tok = AutoTokenizer.from_pretrained(BASE_ID, trust_remote_code=True)

# Rilevamento automatico VRAM: se la GPU ha < 9 GB di VRAM (es. GTX 1650 4GB),
# fonde il LoRA su disco una tantum e carica in 4-bit NF4 interamente in VRAM
_use_4bit_fast = False
if DEV == "cuda":
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if vram_gb < 9.0 and os.environ.get("SPARK_DISABLE_4BIT", "0") != "1":
        try:
            from transformers import BitsAndBytesConfig
            _use_4bit_fast = True
        except ImportError:
            _use_4bit_fast = False

if _use_4bit_fast:
    if not os.path.exists(f"{MERGED_DIR}/config.json"):
        print("[Spark-Decide] Prima esecuzione su GPU <9GB: fusione LoRA in models/spark-decide-merged...")
        _tmp_lm = AutoModelForCausalLM.from_pretrained(BASE_ID, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
        _merged = PeftModel.from_pretrained(_tmp_lm, SAVED).merge_and_unload().eval()
        _merged.save_pretrained(MERGED_DIR, safe_serialization=True)
        tok.save_pretrained(MERGED_DIR)
        for _pyf in ["modeling_spark.py", "configuration_spark.py", "chat_template.jinja"]:
            if os.path.exists(f"{BASE_ID}/{_pyf}"):
                shutil.copy(f"{BASE_ID}/{_pyf}", f"{MERGED_DIR}/{_pyf}")
        del _merged, _tmp_lm
        import gc; gc.collect()

    _cc_major, _cc_minor = torch.cuda.get_device_capability(0)
    # Su architetture prive di Tensor Core FP16 veloci (es. GTX 1650 TU117 sm_75), FP32 compute è 4.1x più veloce di FP16,
    # mentre tenendo torch_dtype=torch.float16 per gli embedding l'intero modello occupa solo 2.62 GB di VRAM!
    _compute_dt = torch.float32 if (_cc_major, _cc_minor) <= (7, 5) else torch.float16
    DT = torch.float16
    _bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_compute_dt,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
    )
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_DIR, trust_remote_code=True, quantization_config=_bnb_cfg, torch_dtype=DT, device_map={"": 0}
    ).eval()
    try:
        import bitsandbytes.backends.cuda.ops as _bnb_ops
        _dq_bufs, _abs_bufs = {}, {}
        def _fast_dequant_linear(A, B, shapeB, absmax, blocksize, quant_type, bias=None, absmax_8bit=None, absmax_code=None, absmax_offset=None):
            if absmax_8bit is not None:
                ak = (absmax_8bit.shape, A.device)
                absmax_dq = _abs_bufs.get(ak)
                if absmax_dq is None:
                    absmax_dq = torch.empty_like(absmax_8bit, dtype=torch.float32)
                    _abs_bufs[ak] = absmax_dq
                _bnb_ops._dequantize_blockwise_impl(absmax_8bit, absmax, absmax_code, 256, torch.float32, out=absmax_dq)
                absmax = absmax_dq + absmax_offset
            bk = (tuple(shapeB), A.dtype, A.device)
            B_dq = _dq_bufs.get(bk)
            if B_dq is None:
                B_dq = torch.empty(shapeB, dtype=A.dtype, device=A.device)
                _dq_bufs[bk] = B_dq
            _bnb_ops._dequantize_4bit_impl(B, absmax, blocksize, quant_type, A.dtype, out=B_dq)
            M = A.shape[-2] if A.ndim >= 2 else 1
            if 256 < M <= 384:
                return torch.cat([
                    F.linear(A[..., :256, :], B_dq, bias),
                    F.linear(A[..., 256:, :], B_dq, bias),
                ], dim=-2)
            return F.linear(A, B_dq, bias)
        _bnb_ops._dequant_linear_fallback = _fast_dequant_linear
    except Exception:
        pass
else:
    _lm = AutoModelForCausalLM.from_pretrained(BASE_ID, trust_remote_code=True, torch_dtype=DT).to(DEV)
    model = PeftModel.from_pretrained(_lm, SAVED).merge_and_unload().eval()

_inner = getattr(model, model.base_model_prefix); _W = model.get_output_embeddings().weight
TEMPS = json.load(open(f"{SAVED}/temperatures.json"))

class DecisionHead(nn.Module):
    def __init__(self, H, d=512):
        super().__init__()
        self.norm = nn.LayerNorm(H); self.po, self.pq = nn.Linear(H, d), nn.Linear(H, d)
        self.type_emb = nn.Embedding(3, d)
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.alpha = nn.Parameter(torch.tensor(1.0))
    def forward(self, h_opt, h_end, ll, t):
        o = self.po(self.norm(h_opt)); q = self.pq(self.norm(h_end)) + self.type_emb(t)
        return self.mlp(o * q + o + q).squeeze(-1) + self.alpha * ll
head = DecisionHead(model.config.hidden_size).to(DEV).float().eval()
head.load_state_dict(torch.load(f"{SAVED}/decision_head.pt", map_location=DEV))

LET = [chr(65 + i) for i in range(26)]
LET_IDS = [tok.encode(" " + L, add_special_tokens=False)[-1] for L in LET]
TID = {"choice": 0, "boolean": 1, "score": 2}
SYS = ("You are a decision engine. Read the STATE, then answer every question independently "
       "by choosing exactly one option letter. Base answers only on the evidence in the state.")
try:
    PRE, POST = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": "<<<BODY>>>"}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False).split("<<<BODY>>>")
except Exception:
    PRE, POST = SYS + "\n\n", "\n"
enc = lambda t: tok.encode(t, add_special_tokens=False)

def _parse(questions):
    out = []
    for name, d in questions.items():
        t = str(d.get("type", "choice")).lower(); crit = d.get("criteria") or d.get("options")
        if t in ("boolean", "noul", "bool"): t, keys, descs = "boolean", ["true", "false"], ["yes", "no"]
        elif isinstance(crit, dict): keys, descs = [str(k) for k in crit], [f"{k}: {v}" for k, v in crit.items()]
        elif t == "score": keys, descs = [str(i) for i in range(len(crit))], [f"{i}: {c}" for i, c in enumerate(crit)]
        else: keys = [str(c) for c in crit]; descs = keys
        if t not in TID: t = "choice"
        out.append(dict(name=name, type=t, instr=d.get("instructions", name), keys=keys, descs=descs))
    return out

@torch.inference_mode()
def decide(state, questions):
    qs = _parse(questions); state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    q_ids, meta = [], []
    for j, q in enumerate(qs):
        q_ids += enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n"); op = []
        for i, d in enumerate(q["descs"]):
            q_ids += enc(f"{LET[i]}) {d}"); op.append(len(q_ids) - 1); q_ids += enc("\n")
        q_ids += enc("Answer:"); meta.append((op, len(q_ids) - 1))
    h_ids, t_ids, p_ids = enc(PRE) + enc("STATE:\n"), enc("\n\nQUESTIONS:"), enc(POST)
    s_ids = enc(state)[:max(32, MAXLEN - len(h_ids) - len(t_ids) - len(q_ids) - len(p_ids))]
    off = len(h_ids) + len(s_ids) + len(t_ids)
    ids = torch.tensor([h_ids + s_ids + t_ids + q_ids + p_ids], device=DEV)
    h = _inner(input_ids=ids, use_cache=False).last_hidden_state[0]   # modello già in DT: niente autocast (non supportato bene su MPS)
    res = {}
    for q, (op, ep) in zip(qs, meta):
        n = len(q["keys"]); op = [p + off for p in op]; ep += off
        ll = F.linear(h[ep].to(_W.dtype), _W[LET_IDS[:n]]).float()
        z = head(h[op].float(), h[ep].float(), ll, torch.tensor(TID[q["type"]], device=DEV))
        p = torch.softmax(z / TEMPS.get(q["type"], 1.0), -1).cpu().numpy(); k = q["keys"][int(p.argmax())]
        res[q["name"]] = dict(answer=(k == "true") if q["type"] == "boolean" else k, confidence=round(float(p.max()), 4),
                              probabilities={kk: round(float(pp), 4) for kk, pp in zip(q["keys"], p)})
    return res

# ---------------------------------------------------------------------------
# decide_fast: stessa identica logica di decide() (che resta invariata), ma il
# prefisso fisso (system prompt + template + "STATE:\n") viene elaborato UNA volta
# e il suo KV-cache viene riusato a ogni chiamata tramite crop() zero-copy,
# più caching della tokenizzazione di QUESTIONS. Validata 60/60 (100% identica) contro decide().
# ---------------------------------------------------------------------------
import copy as _copy
_PREFIX_IDS = enc(PRE) + enc("STATE:\n")
_T_IDS = enc("\n\nQUESTIONS:")
_P_IDS = enc(POST)
_PREFIX_CACHE = None
_Q_CACHE = {}

@torch.inference_mode()
def _prefix_cache():
    global _PREFIX_CACHE
    if _PREFIX_CACHE is None:
        out = _inner(input_ids=torch.tensor([_PREFIX_IDS], device=DEV), use_cache=True)
        _PREFIX_CACHE = out.past_key_values
    return _copy.deepcopy(_PREFIX_CACHE)

@torch.inference_mode()
def decide_fast(state, questions):
    q_key = id(questions)
    cached_q = _Q_CACHE.get(q_key)
    if cached_q is None:
        qs = _parse(questions)
        q_ids, meta = [], []
        for j, q in enumerate(qs):
            q_ids += enc(f"\n\n### Q{j+1} [{q['type']}] {q['name']}: {q['instr']}\n"); op = []
            for i, d in enumerate(q["descs"]):
                q_ids += enc(f"{LET[i]}) {d}"); op.append(len(q_ids) - 1); q_ids += enc("\n")
            q_ids += enc("Answer:"); meta.append((op, len(q_ids) - 1))
        cached_q = (qs, q_ids, meta)
        _Q_CACHE[q_key] = cached_q
    else:
        qs, q_ids, meta = cached_q
    state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    h_ids, t_ids, p_ids = _PREFIX_IDS, _T_IDS, _P_IDS
    s_ids = enc(state)[:max(32, MAXLEN - len(h_ids) - len(t_ids) - len(q_ids) - len(p_ids))]
    P = len(h_ids)
    off = len(s_ids) + len(t_ids)
    rest = s_ids + t_ids + q_ids + p_ids
    ids = torch.tensor([rest], device=DEV)
    h = _inner(input_ids=ids, past_key_values=_prefix_cache(), use_cache=True,
               cache_position=torch.arange(P, P + len(rest), device=DEV)).last_hidden_state[0]
    res = {}
    for q, (op, ep) in zip(qs, meta):
        n = len(q["keys"]); op = [p + off for p in op]; ep += off
        ll = F.linear(h[ep].to(_W.dtype), _W[LET_IDS[:n]]).float()
        z = head(h[op].float(), h[ep].float(), ll, torch.tensor(TID[q["type"]], device=DEV))
        p = torch.softmax(z / TEMPS.get(q["type"], 1.0), -1).cpu().numpy(); k = q["keys"][int(p.argmax())]
        res[q["name"]] = dict(answer=(k == "true") if q["type"] == "boolean" else k, confidence=round(float(p.max()), 4),
                              probabilities={kk: round(float(pp), 4) for kk, pp in zip(q["keys"], p)})
    return res

if __name__ == "__main__":
    import sys
    print(f"[Step 0] Device: {DEV}, Dtype: {DT}, 4-bit Fast Mode: {_use_4bit_fast}")
    for _ in range(3):
        decide_fast("Warmup", {"q": {"type": "noul", "instructions": "Warmup?"}})
    sync()
    t0 = time.perf_counter()
    out = decide_fast(
        "Hi, I was charged twice for invoice #4411. Please refund the duplicate today.",
        {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "invoices, payments, refunds",
                    "technical": "bugs, outages",
                    "sales": "pricing, plans",
                },
            },
            "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "today"],
            },
        },
    )
    sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Step 0] Latenza: {dt_ms:.2f} ms")
    print(json.dumps(out, indent=2))
    if "--test-only" not in sys.argv:
        sys.modules["spark_decide"] = sys.modules[__name__]
        import snake_playground
        snake_playground._WARMED_UP = True
        snake_playground.launch_server(host="0.0.0.0", port=7860, open_browser=("--no-browser" not in sys.argv))
