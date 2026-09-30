"""Two-prompt local smoke test of Qwen2-7B-Instruct with original vs. shuffled BPE.

The shuffle seed (42) is the earlier local demo choice, not a published paper seed.
This tests inference plumbing, not the unverified merge-completion hypothesis.
"""

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
SHUFFLE_DIR = MODEL_DIR / "shuffle_seed42"
SEED = 42
PROMPTS = [
    "What does BPE stand for? Answer briefly.",
    "请用一句话解释什么是词元。",
]


def make_tokenizers():
    original_json = json.loads((MODEL_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    changed_json = copy.deepcopy(original_json)
    merges = changed_json["model"]["merges"]
    assert original_json["model"]["type"] == "BPE"
    assert len(merges) == 151387
    random.Random(SEED).shuffle(merges)
    assert merges != original_json["model"]["merges"]
    assert changed_json["model"]["vocab"] == original_json["model"]["vocab"]
    assert {k: v for k, v in changed_json.items() if k != "model"} == {
        k: v for k, v in original_json.items() if k != "model"
    }
    assert {k: v for k, v in changed_json["model"].items() if k != "merges"} == {
        k: v for k, v in original_json["model"].items() if k != "merges"
    }

    SHUFFLE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        shutil.copy2(MODEL_DIR / name, SHUFFLE_DIR / name)
    (SHUFFLE_DIR / "tokenizer.json").write_text(
        json.dumps(changed_json, ensure_ascii=False), encoding="utf-8"
    )
    standard = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, local_files_only=True)
    shuffled = AutoTokenizer.from_pretrained(SHUFFLE_DIR, use_fast=True, local_files_only=True)
    assert standard.is_fast and shuffled.is_fast
    assert type(standard) is type(shuffled)
    assert standard.get_vocab() == shuffled.get_vocab()
    assert standard.chat_template == shuffled.chat_template
    assert standard.special_tokens_map == shuffled.special_tokens_map
    assert standard.all_special_ids == shuffled.all_special_ids
    assert len(standard) == len(shuffled)
    for token in standard.all_special_tokens:
        assert standard.convert_tokens_to_ids(token) == shuffled.convert_tokens_to_ids(token)

    pilot = json.loads((ROOT / "demo/results/tokenizer_pilot_results.json").read_text())
    assert pilot["seed"] == SEED
    sample = pilot["rows"][0]
    assert standard.encode(sample["text"], add_special_tokens=False) == sample["standard_ids"]
    assert shuffled.encode(sample["text"], add_special_tokens=False) == sample["shuffle_ids"]
    return standard, shuffled, len(merges)


def prepare_inputs(standard, shuffled):
    prepared = []
    special_ids = set(standard.all_special_ids)
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        rendered = standard.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert rendered == shuffled.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        conditions = {}
        for name, tokenizer in (("original", standard), ("shuffle_seed42", shuffled)):
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            assert tokenizer.decode(raw_ids, skip_special_tokens=False) == prompt
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            template_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            assert ids == template_ids
            assert tokenizer.decode(ids, skip_special_tokens=False) == rendered
            conditions[name] = {
                "input_token_ids": ids,
                "input_token_count": len(ids),
                "special_token_ids": [i for i in ids if i in special_ids],
            }
        assert conditions["original"]["special_token_ids"] == conditions["shuffle_seed42"]["special_token_ids"]
        prepared.append({"prompt": prompt, "rendered_chat": rendered, "conditions": conditions})
    assert any(
        row["conditions"]["original"]["input_token_ids"]
        != row["conditions"]["shuffle_seed42"]["input_token_ids"]
        for row in prepared
    )
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = json.loads((ROOT / "model_manifest.json").read_text())
    assert manifest["model_id"] == "Qwen/Qwen2-7B-Instruct"
    standard, shuffled, merge_count = make_tokenizers()
    rows = prepare_inputs(standard, shuffled)
    print(f"Preflight OK: {merge_count} merges, same vocabulary/IDs/special tokens/chat template; all inputs round-trip.", flush=True)
    for row in rows:
        a = row["conditions"]["original"]["input_token_count"]
        b = row["conditions"]["shuffle_seed42"]["input_token_count"]
        print(f"  {row['prompt']!r}: original {a} tokens, shuffle {b} tokens", flush=True)
    if args.preflight_only:
        return

    assert torch.backends.mps.is_available(), "MPS is unavailable"
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype="auto", device_map="mps"
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started
    assert model.device.type == "mps"
    assert next(model.parameters()).dtype == torch.bfloat16
    print(f"Model loaded once on MPS, BF16, in {load_seconds:.1f} s", flush=True)

    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.use_cache = True
    for row in rows:
        for name in ("original", "shuffle_seed42"):
            condition = row["conditions"][name]
            ids = condition["input_token_ids"]
            input_ids = torch.tensor([ids], dtype=torch.long, device="mps")
            attention_mask = torch.ones_like(input_ids)
            torch.mps.synchronize()
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
            torch.mps.synchronize()
            condition["generation_seconds"] = round(time.perf_counter() - generation_started, 3)
            answer_ids = generated[0, len(ids):].tolist()
            condition["output_token_ids"] = answer_ids
            condition["output_token_count"] = len(answer_ids)
            condition["answer"] = standard.decode(answer_ids, skip_special_tokens=True)
            assert condition["answer"] == shuffled.decode(answer_ids, skip_special_tokens=True)
            print(
                f"  {name}: {len(ids)} input / {len(answer_ids)} output tokens, "
                f"{condition['generation_seconds']:.1f} s, answer={condition['answer']!r}",
                flush=True,
            )

    result = {
        "model_id": manifest["model_id"],
        "revision": manifest["revision"],
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(model.device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "shuffle_seed": SEED,
        "shuffle_scope": "one global permutation of tokenizer.json model.merges",
        "merge_count": merge_count,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "eos_token_id": generation_config.eos_token_id,
            "pad_token_id": generation_config.pad_token_id,
            "use_cache": True,
        },
        "load_seconds": round(load_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "rows": rows,
    }
    output = Path(__file__).with_name("results.json")
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {output}; total {result['total_seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
