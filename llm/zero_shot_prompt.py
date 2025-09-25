import os, argparse, time, re
from typing import List, Tuple
from openai import OpenAI
import glob

RESPONSE_INTERVAL = 1

def read_ditto_file(path: str) -> List[Tuple[str, str, int]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: 
                continue
            left, right, label = parts[0], parts[1], int(parts[-1])
            pairs.append((left, right, label))
    return pairs

"""
_LANG_TAG = re.compile(r'@[a-z]{2,3}\b', re.IGNORECASE)
_COL_VAL = re.compile(r'\bCOL\b\s+\S+\s+\bVAL\b', re.IGNORECASE)
_WS = re.compile(r'\s+')

def clean_record(s: str) -> str:
    # make it terser but retain content (works fine if you leave it raw as well)
    s = _LANG_TAG.sub("", s)
    s = _COL_VAL.sub("", s)
    s = s.replace('"', '').strip()
    s = _WS.sub(" ", s)
    return s
"""

PROMPT_TEMPLATE = (
    "You are an entity matching system.\n"
    "Question: Do the two entity descriptions refer to the same real-world entity?\n"
    "Answer with 'Yes' if they do and 'No' if they do not.\n\n"
    "Entity A:\n{left}\n\nEntity B:\n{right}"
)

def ask_llm(client: OpenAI, model: str, left: str, right: str) -> str:
    prompt = PROMPT_TEMPLATE.format(left=left, right=right)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4
    )
    return resp.choices[0].message.content.strip()

def to_label(answer: str) -> int:
    ans = (answer or "").strip().lower()
    return 1 if re.search(r'^\s*yes\b', ans) else 0  # stricter than substring

def f1_pos(preds: List[int], golds: List[int]) -> float:
    tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
    if tp == 0 and (fp > 0 or fn > 0): 
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec

# enables testing across categories
def get_test_files(input_dir):
    test_files = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith("_test.txt"):
            test_files.append(os.path.join(input_dir, filename))
    return test_files
            

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="../data/wdc_multi_dimensional/spec_og_splits/", help="Folder with DITTO-formated test files")
    ap.add_argument("--model", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"])
    ap.add_argument("--limit", type=int, default=0, help="optional cap on pairs")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between calls (set >0 if you hit rate limits)")
    ap.add_argument("--output", default="results.txt", help="File to store f1-score results")
    args = ap.parse_args()
    

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please export OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # get test files from input folder
    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input folder not found: {args.input_dir}")
    test_files = get_test_files(args.input_dir)
    if not test_files:
        raise SystemExit(f"No *_test.txt files found in {args.input_dir}")

    results = {}
    for testf in test_files:
        print(f"Running zero-shot for: {testf}")
        data = read_ditto_file(testf)
        if args.limit and args.limit > 0:
            data = data[:args.limit]

        preds, golds = [], []
        for i, (left, right, gold) in enumerate(data, 1):
            #left_c, right_c = clean_record(left), clean_record(right)
            try:
                ans = ask_llm(client, args.model, left, right)
            except Exception as e:
                # simple retry once
                time.sleep(2.0)
                ans = ask_llm(client, args.model, left, right)
            preds.append(to_label(ans))
            golds.append(gold)
            if args.sleep > 0:
                time.sleep(args.sleep)

            if i % RESPONSE_INTERVAL == 0:
                print(f"[{os.path.basename(testf)} {i}/{len(data)}] last answer: {ans!r}", flush=True)
         
        
        f1, prec, rec = f1_pos(preds, golds)
        results[testf] = [f1, prec, rec]
        name = os.path.basename(testf)
        domain = name[:-9] if name.endswith("_test.txt") else name
        print(f"\nDomain: {domain} | Model: {args.model} | Pairs: {len(data)} | F1 (positive class): {f1:.4f}", flush=True)
     
    
    # write results to file
    with open(args.output, "a") as f:
        f.write(f"~~~ {args.model} 0-shot ~~~\n")
        for testf in test_files:
            name = os.path.basename(testf)
            domain = name[:-9] if name.endswith("_test.txt") else name
            f.write(f"{name} F1: {results[testf][0]:.6f}\n")
            f.write(f"{name} Prec: {results[testf][1]:.6f}\n")
            f.write(f"{name} Rec: {results[testf][2]:.6f}\n")

if __name__ == "__main__":
    main()
