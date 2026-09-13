"""2.3 Empirical: Perplexity and Sampling Strategies (DistilGPT2)."""
import math
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "distilgpt2"
SEED = 42

PARAGRAPH = (
    "The library closed early on Friday because of the storm. "
    "Students packed their laptops and hurried toward the bus stop. "
    "Rain hammered the windows while the wind bent the young trees. "
    "By midnight the streets were quiet again."
)


@torch.no_grad()
def perplexity(text, model, tok):
    """Perplexity = exp(mean token-level NLL) under the causal LM."""
    ids = tok(text, return_tensors="pt").input_ids
    loss = model(ids, labels=ids).loss  # mean NLL over the n-1 predicted tokens
    return math.exp(loss.item())


def shuffled(text, rng):
    words = text.split()
    rng.shuffle(words)
    return " ".join(words)


def part_a(model, tok):
    rng = random.Random(SEED)
    shuf = shuffled(PARAGRAPH, rng)
    ppl_o, ppl_s = perplexity(PARAGRAPH, model, tok), perplexity(shuf, model, tok)

    print("=" * 70, "\n(a) Perplexity Analysis\n" + "=" * 70)
    print(f"\nOriginal  ({len(tok(PARAGRAPH).input_ids)} tokens): {PARAGRAPH}")
    print(f"  perplexity = {ppl_o:.2f}")
    print(f"\nShuffled  ({len(tok(shuf).input_ids)} tokens): {shuf}")
    print(f"  perplexity = {ppl_s:.2f}")
    print(f"\nRatio (shuffled / original) = {ppl_s / ppl_o:.2f}x")


@torch.no_grad()
def part_b(model, tok):
    prompt = "Once upon a time"
    enc = tok(prompt, return_tensors="pt")
    common = dict(max_new_tokens=150, pad_token_id=tok.eos_token_id)

    print("\n" + "=" * 70, "\n(b) Sampling Comparison (prompt: 'Once upon a time')\n" + "=" * 70)

    configs = [("greedy decoding", {})]
    # T=0 is the zero-temperature limit of sampling (argmax), i.e. identical to greedy.
    configs += [(f"sampling T={t}", dict(do_sample=True, temperature=t, top_k=0) if t else {})
                for t in (0, 0.3, 0.6, 0.9, 1.2, 1.5)]

    for name, kw in configs:
        torch.manual_seed(SEED)
        out = model.generate(**enc, **common, **kw)
        print(f"\n--- {name} ---\n{tok.decode(out[0], skip_special_tokens=True)}")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    part_a(model, tok)
    part_b(model, tok)


if __name__ == "__main__":
    main()
