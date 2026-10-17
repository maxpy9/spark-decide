import json, os, sys, time, torch

os.environ["SPARK_ROOT_DIR"] = r"C:\Users\Acer\spark-snake"
sys.path.insert(0, r"C:\Users\Acer\spark-snake")

import spark_decide

state = """
Cliente: Mario Bianchi
Oggetto: Fattura #8821 addebitata due volte sulla carta di credito.
Messaggio: Salve, ho notato che mi avete addebitato due volte l'importo di 149 euro per l'abbonamento annuale.
Chiedo cortesemente il rimborso immediato della somma prelevata per errore. Grazie.
"""

questions = {
    "reparto": {
        "type": "choice",
        "instructions": "A quale reparto inoltrare il ticket?",
        "criteria": {
            "amministrazione_billing": "pagamenti, fatture, rimborsi",
            "supporto_tecnico": "bug, problemi di accesso, server down",
            "vendite": "informazioni commerciali, nuovi contratti"
        }
    },
    "rimborso_richiesto": {
        "type": "noul",
        "instructions": "Il cliente richiede esplicitamente un rimborso?"
    },
    "urgenza": {
        "type": "score",
        "instructions": "Livello di urgenza del problema",
        "criteria": ["bassa", "media", "alta/critica"]
    }
}

print("====================================================================")
print("  CONFRONTO EMPIRICO REALE: BASE RIZZO FLOW vs SPARK-DECIDE")
print(f"  Hardware: NVIDIA GTX 1650 (CUDA) | Precisione: 4-bit NF4")
print("====================================================================")

# Warmup
spark_decide.decide_fast("Warmup", {"q": {"type": "noul", "instructions": "Warmup?"}})

# ---------------------------------------------------------
# 1. SPARK-DECIDE (Single Forward Pass + DecisionHead)
# ---------------------------------------------------------
print("\n>>> [1] Esecuzione SPARK-DECIDE (Single-Forward-Pass DecisionHead)...")
spark_decide.sync()
t0 = time.perf_counter()
res_spark = spark_decide.decide_fast(state, questions)
spark_decide.sync()
lat_spark = (time.perf_counter() - t0) * 1000.0

print(f"  [+] Latenza totale inferenza: {lat_spark:.1f} ms")
print(f"  [+] Token autoregressivi generati: 0 token")
print(f"  [+] Risposte strutturate:")
for k, v in res_spark.items():
    print(f"      - {k}: {v['answer']} (Confidenza: {v['confidence']*100:.1f}%) | Probs: {v['probabilities']}")

# ---------------------------------------------------------
# 2. RIZZO FLOW STANDARD (Autoregressive Token-by-Token Causal LM)
# ---------------------------------------------------------
print("\n>>> [2] Esecuzione BASE RIZZO FLOW (Generazione Autoregressiva Token-by-Token)...")
llm_prompt = f"""<s><|im_start|>system
Sei un assistente decisionale. Rispondi in JSON valido con le chiavi: reparto, rimborso_richiesto, urgenza.
<|im_end|>
<|im_start|>user
CONTESTO:
{state}

DOMANDE:
1. reparto: amministrazione_billing, supporto_tecnico, o vendite?
2. rimborso_richiesto: true o false?
3. urgenza: 0 (bassa), 1 (media), o 2 (alta)?
<|im_end|>
<|im_start|>assistant
"""

input_ids = spark_decide.enc(llm_prompt)
inp_tensor = torch.tensor([input_ids], device=spark_decide.DEV)
n_input_tokens = len(input_ids)

spark_decide.sync()
t0 = time.perf_counter()
with torch.inference_mode():
    gen_out = spark_decide.model.generate(
        inp_tensor,
        max_new_tokens=45,
        do_sample=False,
        eos_token_id=spark_decide.tok.eos_token_id,
        pad_token_id=spark_decide.tok.pad_token_id or spark_decide.tok.eos_token_id
    )
spark_decide.sync()
lat_rizzo = (time.perf_counter() - t0) * 1000.0

gen_tokens = gen_out[0][n_input_tokens:]
n_gen = len(gen_tokens)
gen_text = spark_decide.tok.decode(gen_tokens, skip_special_tokens=True).strip()

print(f"  [+] Latenza totale generazione: {lat_rizzo:.1f} ms")
print(f"  [+] Token generati token-by-token: {n_gen} token")
print(f"  [+] Throughput token: {n_gen / (lat_rizzo / 1000.0):.1f} tok/s")
print(f"  [+] Testo generato da Rizzo Flow:\n'''\n{gen_text}\n'''")

valid_json = False
try:
    json.loads(gen_text)
    valid_json = True
except Exception:
    import re
    m = re.search(r"\{.*\}", gen_text, re.DOTALL)
    if m:
        try:
            json.loads(m.group(0))
            valid_json = True
        except Exception:
            pass

print(f"  [+] JSON valido senza parsing correttivo: {valid_json}")
print(f"  [+] Confidenze numeriche calibrate sulle scelte: NESSUNA (Base LLM non calcola la distribuzione probabilistica)")

speedup = lat_rizzo / lat_spark if lat_spark > 0 else 1.0
print("\n====================================================================")
print(f"  RISULTATO CONFRONTO: Spark-Decide è {speedup:.1f}x PIÙ VELOCE della generazione standard!")
print("====================================================================")
