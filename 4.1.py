"""4.1 Empirical: Head-tuned ModernBERT classifier on SST-2 (backbone frozen)."""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "answerdotai/ModernBERT-base"
SEED = 42
EPOCHS, BS, LR = 30, 256, 1e-3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_splits():
    """GLUE SST-2 test labels are hidden, so: train -> train/dev (95/5), GLUE validation -> test."""
    ds = load_dataset("nyu-mll/glue", "sst2")
    split = ds["train"].train_test_split(test_size=0.05, seed=SEED)
    return {"train": split["train"], "dev": split["test"], "test": ds["validation"]}


@torch.no_grad()
def extract(model, tok, data, bs=512):
    """The backbone is frozen, so run it once and cache the pooled sentence features."""
    feats = []
    for i in range(0, len(data), bs):
        enc = tok(data[i:i + bs]["sentence"], padding=True, return_tensors="pt").to(DEV)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            h = model.model(**enc).last_hidden_state.float()
        if model.config.classifier_pooling == "cls":
            feats.append(h[:, 0])
        else:
            m = enc.attention_mask.unsqueeze(-1)
            feats.append((h * m).sum(1) / m.sum(1))
    return torch.cat(feats), torch.tensor(data["label"], device=DEV)


@torch.no_grad()
def accuracy(head, x, y):
    head.eval()
    return (head(x).argmax(-1) == y).float().mean().item()


def main():
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(DEV).eval()
    data = {k: extract(model, tok, v) for k, v in load_splits().items()}

    # Classification head = ModernBertPredictionHead (dense+act+norm) -> dropout -> linear.
    head = torch.nn.Sequential(model.head, model.drop, model.classifier)
    print(f"trainable params: {sum(p.numel() for p in head.parameters()):,}")
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    xtr, ytr = data["train"]
    hist, best_acc, best_state = {"train": [], "dev": []}, 0, None
    for ep in range(1, EPOCHS + 1):
        head.train()
        for idx in torch.randperm(len(ytr), device=DEV).split(BS):
            loss = F.cross_entropy(head(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        for s in hist:
            hist[s].append(accuracy(head, *data[s]))
        print(f"epoch {ep:2d}  train={hist['train'][-1]:.4f}  dev={hist['dev'][-1]:.4f}")
        if hist["dev"][-1] > best_acc:
            best_acc, best_state = hist["dev"][-1], copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    print(f"\nbest dev = {best_acc:.4f}  ->  test (GLUE validation) = {accuracy(head, *data['test']):.4f}")

    epochs = range(1, EPOCHS + 1)
    plt.plot(epochs, hist["train"], label="train")
    plt.plot(epochs, hist["dev"], label="dev")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("ModernBERT head tuning on SST-2")
    plt.legend()
    plt.savefig("4.1_accuracy.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
