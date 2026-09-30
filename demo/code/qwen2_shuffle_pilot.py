"""First reproduction step: Qwen2 tokenizer baseline vs. fixed merge shuffle.

Run on your Mac after installing `huggingface_hub` and `tokenizers`:
    python qwen2_shuffle_pilot.py

This downloads tokenizer.json only (not the 7B model weights). It changes the
order of BPE merges, while retaining the vocabulary, IDs, pre-tokenizer,
decoder, and special-token settings in the original tokenizer.json.

The paper specifies a fixed random seed, but not its value or full code.
We explicitly choose one global permutation with seed 42 for this pilot.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


MODEL_ID = "Qwen/Qwen2-7B-Instruct"
EXAMPLES = [
    "Standard byte pair encoding uses a vocabulary and a learned merge order.",
    "If we shuffle the merge rules, will the same model answer differently?",
    "The researchers compared question answering and machine translation.",
    "Quantum chromodynamics and immunohistochemistry are technical terms.",
    "这篇论文研究改变输入的切分方式会怎样影响语言模型。",
    "def tokenize(text: str) -> list[int]: return [len(text)]",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text", action="append", help="Add a sample; can repeat")
    parser.add_argument("--out", type=Path, default=Path("tokenizer_pilot_results.json"))
    args = parser.parse_args()

    tokenizer_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.json")
    original = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    assert original["model"]["type"] == "BPE", "Expected a BPE tokenizer"
    merges = original["model"]["merges"]
    assert merges and isinstance(merges, list), "Expected a merge list"

    changed = json.loads(json.dumps(original))  # independent deep copy
    random.Random(args.seed).shuffle(changed["model"]["merges"])
    assert changed["model"]["vocab"] == original["model"]["vocab"]
    assert changed["model"]["merges"] != merges

    baseline = Tokenizer.from_str(json.dumps(original, ensure_ascii=False))
    shuffled = Tokenizer.from_str(json.dumps(changed, ensure_ascii=False))
    samples = EXAMPLES + (args.text or [])
    rows = []
    for sample in samples:
        a = baseline.encode(sample, add_special_tokens=False)
        b = shuffled.encode(sample, add_special_tokens=False)
        # BPE must preserve the original bytes/text in both conditions.
        if baseline.decode(a.ids, skip_special_tokens=False) != sample:
            raise AssertionError(f"Baseline did not round-trip: {sample!r}")
        if shuffled.decode(b.ids, skip_special_tokens=False) != sample:
            raise AssertionError(f"Shuffle did not round-trip: {sample!r}")
        row = {
            "text": sample,
            "standard_ids": a.ids,
            "shuffle_ids": b.ids,
            "standard_tokens": a.tokens,
            "shuffle_tokens": b.tokens,
            "standard_count": len(a.ids),
            "shuffle_count": len(b.ids),
            "changed": a.ids != b.ids,
        }
        rows.append(row)
        print(f"\nTEXT: {sample}")
        print(f"standard ({len(a.ids)}): {a.tokens}")
        print(f"shuffle  ({len(b.ids)}): {b.tokens}")

    result = {
        "model_id": MODEL_ID,
        "tokenizer_path": tokenizer_path,
        "seed": args.seed,
        "shuffle_scope": "one global permutation of tokenizer.json model.merges",
        "merge_count": len(merges),
        "changed_examples": sum(row["changed"] for row in rows),
        "total_examples": len(rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nChanged {result['changed_examples']}/{len(rows)} examples. Saved {args.out}")
    if not result["changed_examples"]:
        raise RuntimeError("No input changed; inspect tokenizer construction before evaluating a model.")


if __name__ == "__main__":
    main()
