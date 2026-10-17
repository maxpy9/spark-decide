<p align="center">
  <img src="spark_decide_logo.png" alt="Spark-Decide Mascot Logo" width="180" style="border-radius: 18px;"/>
</p>

# ⚡ Spark-Decide — Decision Engine & Studio Playground

**Spark-Decide** è un motore decisionale strutturato a singolo forward pass con **DecisionHead** calibrata, progettato per rispondere simultaneamente a domande multi-tipo (`choice`, `boolean`/`noul`, `score`) a partire da uno stato (`STATE` testuale o JSON) restituendo risposte, confidenza e distribuzioni di probabilità calibrate.

Include **Spark-Decide Studio** (`http://127.0.0.1:7860`), un'interfaccia web interattiva che si avvia automaticamente eseguendo il modello per:
1. **🐍 Testarlo in tempo reale su Snake** (*Snake Live Arena* con telemetria, barre di probabilità, `danger`, `risk` e Guard-rail).
2. **⚡ Usarlo su qualsiasi task decisionale** (*Decision Studio* interattivo + REST API `POST /api/decide`).
3. **📊 Eseguire la Benchmark Suite** multi-modalità contro agenti baseline (*Greedy* e *Casuale-Sicuro*).

---

## 🏗️ Architettura & Crediti (Basato su Rizzo Flow)

Il modello **Spark-Decide** è stato sviluppato basandosi sull'architettura foundational **Rizzo Flow** (`rizzoaiacademy/rizzo-flow`), estendendola con:
- **Adattamento specializzato (`models/spark-decide`)**: pesi LoRA (`r=32, alpha=64`) fusi sul backbone (`merge_and_unload()`) per l'estrazione deterministica di evidenze dallo `STATE`.
- **DecisionHead dedicata (`decision_head.pt` + `temperatures.json`)**: testa neurale leggera che combina gli hidden state delle singole opzioni (`h_opt`), il contesto finale della domanda (`h_end`), gli embedding di tipologia (`choice`, `boolean`, `score`) e i log-likelihood lessicali (`ll`) con scaling di temperatura calibrato per tipo.
- **Ottimizzazione estrema CUDA (`decide_fast`)**:
  - Quantizzazione **4-bit NF4 Double-Quant** (`models/spark-decide-merged`, **2,62 GB** di VRAM totali) con calcolo nativo **FP32** per GPU consumer con 4 GB di VRAM (es. NVIDIA GTX 1650).
  - **KV-Prefix Caching** del system prompt + template fisso, riuso dei buffer di dequantizzazione su GPU e split automatico dei blocchi cuBLAS (`M > 256`), riducendo la latenza di ~15× (**~1,07s – 1,80s/mossa**) con il **100% di equivalenza decisionale** rispetto a `decide()`.

---

## 📁 Struttura del Repository

- [`SparkDecide_Setup.exe`](SparkDecide_Setup.exe) — **Installer & Launcher nativo Windows (GUI)** con icona della mascotte: scarica e configura automaticamente Python 3.11 (se assente), l'ambiente virtuale `.venv`, PyTorch CUDA 12.4 + librerie AI, il modello **Spark-Decide** da questa repo GitHub e **Rizzo Flow** da HuggingFace ([`setup_models.py`](setup_models.py)), avviando **Spark-Decide Studio** su `http://127.0.0.1:7860`.
- [`spark_decide.py`](spark_decide.py) — Motore principale **Spark-Decide**: espone `decide(state, questions)` e `decide_fast(state, questions)`. Se eseguito direttamente (`python spark_decide.py`), esegue il test rapido e avvia automaticamente **Spark-Decide Studio** su `http://127.0.0.1:7860`.
- [`snake_playground.py`](snake_playground.py) — Server **FastAPI + Spark-Decide Studio Web App** su `http://127.0.0.1:7860` con *Snake Live Arena*, *Decision Studio*, *Benchmark Suite* ed endpoint REST `POST /api/decide`.
- [`setup_models.py`](setup_models.py) — Script automatico che scarica il modello **Spark-Decide** da GitHub, il modello base **Rizzo Flow** da HuggingFace e applica le patch 4-bit NF4.
- [`snake_env.py`](snake_env.py) — Motore completo di Snake (griglia NxN configurabile, seed riproducibile, rilevamento collisioni muro/corpo e inversione 180°, sensori direzionali con flood-fill `reachable_space`, griglia ASCII e policy baseline `policy_greedy` / `policy_random_safe`).
- [`benchmark.py`](benchmark.py) — Suite di benchmark automatico multi-modalità (`sensori + griglia`, `solo sensori`, `solo griglia ASCII`, Guard-rail OFF/ON + baselines) con esportazione in `benchmark_snake.csv` e `benchmark_snake.json`.
- [`requirements.txt`](requirements.txt) — Dipendenze Python del progetto.

---

## 🚀 Avvio Rapido (Con Eseguibile `.exe` o Terminale)

### Opzione A: Doppio clic sull'eseguibile Windows (Consigliato)
Fai doppio clic su **[`SparkDecide_Setup.exe`](SparkDecide_Setup.exe)** e premi il pulsante verde **"⚡ Scarica Tutto (GitHub + HF) & Avvia Spark-Decide"**: l'eseguibile verificherà/scaricherà automaticamente tutto il necessario e aprirà **Spark-Decide Studio** su `http://127.0.0.1:7860`.

### Opzione B: Da Terminale PowerShell
```powershell
cd $HOME\spark-snake
.\.venv\Scripts\Activate.ps1
python spark_decide.py
```

Il comando:
1. Carica **Spark-Decide** su GPU (`CUDA 4-bit NF4 + KV-Cache`).
2. Esegue il test di verifica iniziale in console.
3. Avvia e apre automaticamente **Spark-Decide Studio** su **[http://127.0.0.1:7860](http://127.0.0.1:7860)** *(usa `--no-browser` per non aprire il browser o `--test-only` per eseguire solo il test da terminale)*.

---

## 🔌 Utilizzo del Modello via Python o REST API

### 1. Import diretto in Python
```python
from spark_decide import decide, decide_fast

state = "Hi, I was charged twice for invoice #4411. Please refund the duplicate today."
questions = {
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
}

out = decide_fast(state, questions)
print(out)
```

### 2. Endpoint HTTP REST (`POST http://127.0.0.1:7860/api/decide`)
```python
import requests

resp = requests.post(
    "http://127.0.0.1:7860/api/decide",
    json={"state": state, "questions": questions, "fast": True},
)
print(resp.json())
```
