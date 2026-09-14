"""4.2 Empirical: LoRA-tuned ModernBERT classifier on SST-2."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "answerdotai/ModernBERT-base"
SEED = 42
EPOCHS, BS, LR, EVAL_EVERY = 3, 32, 5e-4, 500
TRAIN_EVAL = 5000  # fixed train subset, so tracking train accuracy stays cheap
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_splits():
    """GLUE SST-2 test labels are hidden, so: train -> train/dev (95/5), GLUE validation -> test."""
    ds = load_dataset("nyu-mll/glue", "sst2")
    split = ds["train"].train_test_split(test_size=0.05, seed=SEED)
    train_sub = split["train"].shuffle(seed=SEED).select(range(TRAIN_EVAL))
    return {"train": split["train"], "train_sub": train_sub, "dev": split["test"], "test": ds["validation"]}


def loader(data, tok, shuffle=False):
    def collate(batch):
        enc = tok([b["sentence"] for b in batch], padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["label"] for b in batch])
        return enc
    return DataLoader(data, batch_size=BS if shuffle else 256, shuffle=shuffle, collate_fn=collate)


def autocast():
    return torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda")


@torch.no_grad()
def accuracy(model, dl):
    model.eval()
    correct = 0
    for b in dl:
        b = b.to(DEV)
        with autocast():
            correct += (model(**b).logits.argmax(-1) == b["labels"]).sum().item()
    model.train()
    return correct / len(dl.dataset)


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    # LoRA on attention qkv + output projections (r=6) -> ~0.6M params, same ballpark as head tuning (4.1).
    # task_type SEQ_CLS also keeps the new classifier layer trainable.
    cfg = LoraConfig(task_type="SEQ_CLS", r=6, lora_alpha=12, lora_dropout=0.1, target_modules=["Wqkv", "attn.Wo"])
    model = get_peft_model(base, cfg).to(DEV)
    model.print_trainable_parameters()

    splits = load_splits()
    dls = {k: loader(v, tok, shuffle=k == "train") for k, v in splits.items()}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    total = EPOCHS * len(dls["train"])
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    hist, best_acc, best_state, step = {"step": [], "train": [], "dev": []}, 0, None, 0
    model.train()
    for ep in range(EPOCHS):
        for b in dls["train"]:
            with autocast():
                loss = model(**b.to(DEV)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % EVAL_EVERY == 0 or step == total:
                tr, dev = accuracy(model, dls["train_sub"]), accuracy(model, dls["dev"])
                hist["step"].append(step)
                hist["train"].append(tr)
                hist["dev"].append(dev)
                print(f"epoch {ep + 1} step {step:5d}  loss={loss.item():.4f}  train={tr:.4f}  dev={dev:.4f}")
                if dev > best_acc:
                    best_acc = dev
                    best_state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

    set_peft_model_state_dict(model, best_state)
    print(f"\nbest dev = {best_acc:.4f}  ->  test (GLUE validation) = {accuracy(model, dls['test']):.4f}")

    plt.plot(hist["step"], hist["train"], label=f"train (subset of {TRAIN_EVAL})")
    plt.plot(hist["step"], hist["dev"], label="dev")
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title("ModernBERT LoRA on SST-2")
    plt.legend()
    plt.savefig("4.2_accuracy.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
