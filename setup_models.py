import argparse
import concurrent.futures
import os
import ssl
import time
import urllib.request

parser = argparse.ArgumentParser(description="Spark-Decide Fast Model Downloader & Setup")
parser.add_argument("--dir", default=os.environ.get("SPARK_ROOT_DIR", "."), help="Cartella principale di destinazione")
args, _ = parser.parse_known_args()

ROOT_DIR = os.path.abspath(args.dir)
GITHUB_REPO = "maxpy9/spark-decide"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main"

HF_BASE_REPO = "rizzoaiacademy/rizzo-flow"
BASE_DIR = os.path.join(ROOT_DIR, "models", "rizzo-flow")
ADAPTER_DIR = os.path.join(ROOT_DIR, "models", "spark-decide")
MERGED_DIR = os.path.join(ROOT_DIR, "models", "spark-decide-merged")

SPARK_DECIDE_DIRECT_FILES = [
    "adapter_config.json",
    "temperatures.json",
    "decision_head.pt",
]

SPARK_DECIDE_PARTS = [
    "adapter_model.safetensors.part1",
    "adapter_model.safetensors.part2",
    "adapter_model.safetensors.part3",
]

HF_BASE_FILES = [
    "config.json",
    "generation_config.json",
    "configuration_spark.py",
    "modeling_spark.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "model-00001-of-00005.safetensors",
    "model-00002-of-00005.safetensors",
    "model-00003-of-00005.safetensors",
    "model-00004-of-00005.safetensors",
    "model-00005-of-00005.safetensors",
]


def download_file(url: str, dest: str):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp_dest = dest + ".part_dl"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={"User-Agent": "Spark-Decide-Installer/2.0"})
    t_start = time.time()
    with urllib.request.urlopen(req, context=ctx) as resp, open(tmp_dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        chunk_size = 16 * 1024 * 1024
        last_pct = -1
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = int(downloaded * 100 / total)
                if pct != last_pct and pct % 20 == 0:
                    elapsed = max(0.1, time.time() - t_start)
                    speed_mbs = (downloaded / (1024 * 1024)) / elapsed
                    mb_done = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024)
                    print(f"    -> {os.path.basename(dest)}: {pct}% ({mb_done:.1f}/{mb_tot:.1f} MB @ {speed_mbs:.1f} MB/s)", flush=True)
                    last_pct = pct
    os.replace(tmp_dest, dest)


def ensure_spark_decide_from_github():
    """Scarica il modello Spark-Decide dalla repo GitHub maxpy9/spark-decide."""
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    for fname in SPARK_DECIDE_DIRECT_FILES:
        dest = os.path.join(ADAPTER_DIR, fname)
        if not (os.path.exists(dest) and os.path.getsize(dest) > 0):
            url = f"{GITHUB_RAW_BASE}/models/spark-decide/{fname}"
            print(f"[GitHub] Download veloce {fname}...", flush=True)
            download_file(url, dest)

    safetensors_path = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if not (os.path.exists(safetensors_path) and os.path.getsize(safetensors_path) > 10_000_000):
        print(f"[GitHub] Download parallelo e ricostruzione di adapter_model.safetensors...", flush=True)
        part_paths = []
        dl_tasks = []
        for pname in SPARK_DECIDE_PARTS:
            pdest = os.path.join(ADAPTER_DIR, pname)
            part_paths.append(pdest)
            if not (os.path.exists(pdest) and os.path.getsize(pdest) > 0):
                url = f"{GITHUB_RAW_BASE}/models/spark-decide/{pname}"
                dl_tasks.append((url, pdest))

        if dl_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(download_file, u, d) for u, d in dl_tasks]
                concurrent.futures.wait(futures)

        tmp_st = safetensors_path + ".building"
        with open(tmp_st, "wb") as out_f:
            for ppath in part_paths:
                with open(ppath, "rb") as in_f:
                    while True:
                        buf = in_f.read(16 * 1024 * 1024)
                        if not buf:
                            break
                        out_f.write(buf)
        os.replace(tmp_st, safetensors_path)
        print("[GitHub] adapter_model.safetensors pronto!", flush=True)
    else:
        print(f"[GitHub] Modello Spark-Decide OK.", flush=True)


def ensure_rizzo_flow_from_hf():
    """Scarica il modello base Rizzo Flow da HuggingFace in parallelo multi-thread."""
    os.makedirs(BASE_DIR, exist_ok=True)
    missing = [f for f in HF_BASE_FILES if not (os.path.exists(os.path.join(BASE_DIR, f)) and os.path.getsize(os.path.join(BASE_DIR, f)) > 0)]
    if not missing:
        print(f"[HuggingFace] Modello base Rizzo Flow OK (tutti i file presenti).", flush=True)
        return

    print(f"[HuggingFace] Download parallelo ad alta velocita' (8 stream) di {len(missing)} file da {HF_BASE_REPO}...", flush=True)
    dl_tasks = []
    for fname in missing:
        url = f"https://huggingface.co/{HF_BASE_REPO}/resolve/main/{fname}"
        dest = os.path.join(BASE_DIR, fname)
        dl_tasks.append((url, dest, fname))

    def _task(item):
        u, d, fn = item
        try:
            download_file(u, d)
        except Exception as e:
            if fn in ("chat_template.jinja", "special_tokens_map.json"):
                print(f"[HuggingFace] File opzionale {fn} ignorato: {e}", flush=True)
            else:
                raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_task, it) for it in dl_tasks]
        concurrent.futures.wait(futures)


def patch_modeling_spark(py_path: str):
    if not os.path.exists(py_path):
        return
    txt = open(py_path, "r", encoding="utf-8").read()
    changed = False
    old_cast = "        hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)"
    new_cast = (
        "        if self.mlp.gate_proj.weight.dtype.is_floating_point:\n"
        "            hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)"
    )
    if old_cast in txt and "is_floating_point" not in txt:
        txt = txt.replace(old_cast, new_cast)
        changed = True
    if changed:
        open(py_path, "w", encoding="utf-8").write(txt)
        print(f"[Setup] Ottimizzazione 4-bit applicata a {py_path}.", flush=True)


if __name__ == "__main__":
    print(f"=== Setup e Download Modelli Spark-Decide (Cartella: {ROOT_DIR}) ===", flush=True)
    ensure_spark_decide_from_github()
    ensure_rizzo_flow_from_hf()
    patch_modeling_spark(os.path.join(BASE_DIR, "modeling_spark.py"))
    patch_modeling_spark(os.path.join(MERGED_DIR, "modeling_spark.py"))
    print("=== Setup Modelli Completato con Successo! ===", flush=True)
