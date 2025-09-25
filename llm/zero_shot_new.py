import os, argparse, time, re
from typing import List, Tuple

# --- BACKENDS ---
# OpenAI (hosted): optional
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# Hugging Face (local): optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

RESPONSE_INTERVAL = 1

_LANG_TAG = re.compile(r'@[a-z]{2,3}\b', re.IGNORECASE)
# non-greedy capture of value until next "COL ... VAL" or end
_COL_VAL_RE = re.compile(r'COL\s+([^\s]+)\s+VAL\s+(.*?)(?=\s+COL\s+\S+\s+VAL\s+|$)', re.IGNORECASE | re.DOTALL)
_WS = re.compile(r'\s+')

def _fmt_safe(s: str) -> str:
    # escape braces so str.format won't treat them as placeholders
    return s.replace("{", "{{").replace("}", "}}")

def _parse_ditto_kv(s: str) -> list[tuple[str, str]]:
    """Parse 'COL <attr> VAL <value>' sequences from a DITTO record string."""
    kv = []
    for m in _COL_VAL_RE.finditer(s):
        attr = m.group(1).strip()
        val = m.group(2).strip()
        # drop language tags like @en or @de
        val = _LANG_TAG.sub("", val).strip()
        # strip surrounding quotes if present
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        # handle quoted-with-lang: "text"@en -> text
        q = re.match(r'^"([^"]*)"', val)
        if q:
            val = q.group(1)
        # normalize whitespace
        val = _WS.sub(" ", val).strip()
        # None -> N/A (Jellyfish prompt mentions N/A explicitly)
        if val.lower() in {"none", "nan"}:
            val = "N/A"
        # prettify attr
        attr = attr.replace("_", " ").lower()
        kv.append((attr, val))
    return kv

def _serialize_for_jellyfish(s: str, multiline: bool = False) -> str:
    """Convert a DITTO record string into 'attr: val; attr2: val2; ...' or multi-line."""
    kv = _parse_ditto_kv(s)
    if not kv:
        # fall back to raw string if parsing fails
        return s.strip()
    if multiline:
        return "\n".join(f"- {k}: {v}" for k, v in kv)
    return ", ".join(f"{k}: {v}" for k, v in kv)

# -------------------- Data utils --------------------
def read_ditto_file(path: str) -> List[Tuple[str, str, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            left, right, label = parts[0], parts[1], int(parts[-1])  # your files use 0/1 labels
            pairs.append((left, right, label))
    return pairs

PROMPT_TEMPLATE = (
    "You are an entity matching system.\n"
    "Question: Do the two entity descriptions refer to the same real-world entity?\n"
    "Answer with 'Yes' if they do and 'No' if they do not.\n\n"
    "Entity A:\n{left}\n\nEntity B:\n{right}"
)

# Jellyfish-specific EM prompt (auto-used when hf_model contains "jellyfish")
JELLYFISH_PROMPT_TEMPLATE = (
    "You are tasked with determining whether two records listed below describe the same entity.\n"
    "Return exactly one token: 'Yes' or 'No'.\n\n"
    "Record A:\n{left}\n\nRecord B:\n{right}\n\nAnswer:"
)

REAL_JELLYFISH_PROMPT_TEMPLATE = (
    "You are tasked with determining whether two records listed below are the same based on the information provided.\n"
    "Carefully compare the {{attribute 1}}, {{attribute 2}}... for each record before making your decision.\n"  
    "Note: Missing values (N/A or \"nan\") should not be used as a basis for your decision.\n"
    "Record A: [{left}]\n"
    "Record B: [{right}]\n"
    "Are record A and record B the same entity? Choose your answer from: [Yes, No]."
)

def to_label(answer: str) -> int:
    ans = (answer or "").strip().lower()
    return 1 if re.search(r'^\s*yes\b', ans) else 0

def f1_pos(preds: List[int], golds: List[int]):
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0, 0.0, 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec

def get_test_files(input_dir):
    test_files = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith("_test.txt"):
            test_files.append(os.path.join(input_dir, filename))
    return test_files

# -------------------- Backends --------------------
def build_openai_ask(model: str, api_key_env: str = "OPENAI_API_KEY"):
    if _OpenAI is None:
        raise RuntimeError("openai package not installed; use --backend hf or `pip install openai`.")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Please export {api_key_env}")
    client = _OpenAI(api_key=api_key)

    def ask(left: str, right: str) -> str:
        prompt = PROMPT_TEMPLATE.format(left=left, right=right)
        r = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4
        )
        return (r.choices[0].message.content or "").strip()
    return ask

def build_hf_ask(
    hf_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    four_bit: bool = True,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    hf_token=None
):
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
    ) if four_bit else None

    auth = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True, **auth)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant,
        **auth
    )
    try:
        model.config.use_cache = True
    except Exception:
        pass

    # Use Jellyfish EM prompt when requested; otherwise the generic prompt
    prompt_tmpl = REAL_JELLYFISH_PROMPT_TEMPLATE if "jellyfish" in hf_model.lower() else PROMPT_TEMPLATE # !!! TEMP CHANGED TEMPLATE TO SEE IF IMPROVEMENT 

    # Use chat template if available (Instruct models).
    def ask(left: str, right: str) -> str:
        # If we're using a Jellyfish checkpoint, serialize DITTO strings into attr: val lists
        if "jellyfish" in hf_model.lower():
            left_s  = _serialize_for_jellyfish(left,  multiline=False)
            right_s = _serialize_for_jellyfish(right, multiline=False)
        else:
            left_s, right_s = left, right
        # Always escape braces from data to avoid KeyError during .format()
        user = prompt_tmpl.format(left=_fmt_safe(left_s), right=_fmt_safe(right_s))
        #print(f"USER PROMPT: {user}")
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are an entity matching system."},
                {"role": "user", "content": user},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = user  # fallback

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                #temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return txt

    return ask

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="../data/wdc_multi_dimensional/spec_og_splits/",
                    help="Folder with DITTO-formatted test files")
    # backend + model args
    ap.add_argument("--backend", default="openai", choices=["openai", "hf"])
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model name if --backend openai")
    ap.add_argument("--hf_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="HF model if --backend hf")
    ap.add_argument("--hf_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--hf_4bit", action="store_true",
                    help="Use 4-bit quantization (QLoRA-style) for local model")
    ap.add_argument("--limit", type=int, default=0, help="optional cap on pairs")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between calls")
    ap.add_argument("--output", default="results.txt", help="File to store results")
    ap.add_argument("--hf_token_env", default="HUGGINGFACE_HUB_TOKEN")
    args = ap.parse_args()

    # Build ask() according to backend
    if args.backend == "openai":
        ask = build_openai_ask(args.model)
    else:
        hf_token = os.getenv(args.hf_token_env)
        ask = build_hf_ask(
            hf_model=args.hf_model,
            four_bit=args.hf_4bit,
            dtype=args.hf_dtype,
            device_map="auto",
            hf_token=hf_token
        )

    # Input folder
    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input folder not found: {args.input_dir}")
    test_files = get_test_files(args.input_dir)
    if not test_files:
        raise SystemExit(f"No *_test.txt files found in {args.input_dir}")

    # Evaluate per file
    with open(args.output, "a", encoding="utf-8") as outf:
        outf.write(f"~~~ {args.backend}:{args.model if args.backend=='openai' else args.hf_model} 0-shot ~~~\n")
        for testf in test_files:
            print(f"Running zero-shot for: {testf}")
            data = read_ditto_file(testf)
            if args.limit and args.limit > 0:
                data = data[:args.limit]

            preds, golds = [], []
            for i, (left, right, gold) in enumerate(data, 1):
                try:
                    ans = ask(left, right)
                except Exception:
                    time.sleep(2.0)
                    ans = ask(left, right)
                preds.append(to_label(ans))
                golds.append(gold)
                if args.sleep > 0:
                    time.sleep(args.sleep)
                if i % RESPONSE_INTERVAL == 0:
                    print(f"[{os.path.basename(testf)} {i}/{len(data)}] last answer: {ans!r}", flush=True)

            f1, prec, rec = f1_pos(preds, golds)
            name = os.path.basename(testf)
            domain = name[:-9] if name.endswith("_test.txt") else name
            print(f"\nDomain: {domain} | Backend: {args.backend} | Model: {args.model if args.backend=='openai' else args.hf_model} | Pairs: {len(data)} | F1+: {f1:.4f}", flush=True)
            outf.write(f"{name} F1: {f1:.6f}\n")
            outf.write(f"{name} Precision: {prec:.6f}\n")
            outf.write(f"{name} Recall: {rec:.6f}\n")

if __name__ == "__main__":
    main()
