import json
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model import CharacterVocab, CanineLoRACharPredictor, ModelConfig, text_to_codepoints

INPUT_PAD_CODEPOINT = 0
MIN_CANINE_INPUT_LEN = 4


@dataclass
class TrainConfig:
    input_path: str = "data/open-dev/input.txt"
    answer_path: str = "data/open-dev/answer.txt"
    work_dir: str = "work"
    vocab_size: int = 8000
    min_freq: int = 1
    max_seq_len: int = 512
    batch_size: int = 32
    num_epochs: int = 5
    val_split: float = 0.1
    lr_encoder_lora: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 0
    device: str = "cuda"
    log_every: int = 100


class PairedCharacterDataset(Dataset):
    def __init__(
        self,
        contexts: Sequence[str],
        targets: Sequence[str],
        vocab: CharacterVocab,
        max_seq_len: int = 512,
    ) -> None:
        self.contexts = list(contexts)
        self.target_ids = [vocab.get_id(t) for t in targets]
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        ids = text_to_codepoints(self.contexts[idx], max_len=self.max_seq_len)
        if not ids:
            ids = [INPUT_PAD_CODEPOINT]
        return ids, self.target_ids[idx]


def collate_batch(batch: Sequence[Tuple[List[int], int]]) -> Dict[str, torch.Tensor]:
    max_len = max(max(len(context_ids) for context_ids, _ in batch), MIN_CANINE_INPUT_LEN)
    input_ids = []
    attention_masks = []
    labels = []
    for context_ids, label in batch:
        pad_count = max_len - len(context_ids)
        input_ids.append(context_ids + [INPUT_PAD_CODEPOINT] * pad_count)
        attention_masks.append([1] * len(context_ids) + [0] * pad_count)
        labels.append(label)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def read_paired_data(input_path: str, answer_path: str) -> Tuple[List[str], List[str]]:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.isfile(answer_path):
        raise FileNotFoundError(f"Answer file not found: {answer_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        contexts = [line.rstrip("\n") for line in f]
    with open(answer_path, "r", encoding="utf-8") as f:
        targets = [line.rstrip("\n") for line in f]

    if len(contexts) != len(targets):
        raise ValueError(
            "Input/answer line count mismatch: "
            f"{input_path} has {len(contexts)} lines, "
            f"{answer_path} has {len(targets)} lines."
        )

    for line_num, target in enumerate(targets, start=1):
        if len(target) != 1:
            raise ValueError(
                "Each answer line must contain exactly one Unicode character: "
                f"{answer_path}:{line_num} has {target!r} (length={len(target)})."
            )

    return contexts, targets


def split_train_val(
    contexts: Sequence[str],
    targets: Sequence[str],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    if len(contexts) != len(targets):
        raise ValueError(
            f"Contexts and targets must have the same length, got {len(contexts)} and {len(targets)}."
        )

    idxs = list(range(len(contexts)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_size = int(len(idxs) * val_split)

    val_idxs = set(idxs[:val_size])
    train_contexts: List[str] = []
    train_targets: List[str] = []
    val_contexts: List[str] = []
    val_targets: List[str] = []

    for i in range(len(contexts)):
        context = contexts[i]
        target = targets[i]
        if i in val_idxs:
            val_contexts.append(context)
            val_targets.append(target)
        else:
            train_contexts.append(context)
            train_targets.append(target)

    return train_contexts, train_targets, val_contexts, val_targets


@torch.no_grad()
def evaluate_top3_accuracy(
    model: CanineLoRACharPredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        top3 = model.predict_topk(input_ids=input_ids, attention_mask=attention_mask, k=3)
        matches = (top3 == labels.unsqueeze(-1)).any(dim=-1)
        correct += int(matches.sum().item())
        total += int(labels.size(0))

    if total == 0:
        return 0.0
    return correct / total


def build_optimizer(
    model: CanineLoRACharPredictor,
    lr_encoder_lora: float,
    lr_head: float,
    weight_decay: float,
) -> AdamW:
    lora_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        elif name.startswith("head."):
            head_params.append(param)

    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr_encoder_lora})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")
    return AdamW(param_groups, weight_decay=weight_decay)


def load_train_config(config_path: str, overrides: Dict[str, Any]) -> TrainConfig:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Training config must be a JSON object.")

    allowed_keys = {field.name for field in fields(TrainConfig)}
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {', '.join(unknown_keys)}")

    missing_keys = sorted(allowed_keys - set(payload))
    if missing_keys:
        raise ValueError(f"Missing config keys: {', '.join(missing_keys)}")

    resolved = dict(payload)
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value

    return TrainConfig(**resolved)


def train(config: TrainConfig) -> None:
    os.makedirs(config.work_dir, exist_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = config.device
    if device == "cuda":
        if torch.cuda.is_available():
            runtime_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            runtime_device = torch.device("mps")
        else:
            runtime_device = torch.device("cpu")
    elif device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            runtime_device = torch.device("mps")
        else:
            runtime_device = torch.device("cpu")
    else:
        runtime_device = torch.device("cpu")

    contexts, targets = read_paired_data(config.input_path, config.answer_path)
    train_contexts, train_targets, val_contexts, val_targets = split_train_val(
        contexts=contexts,
        targets=targets,
        val_split=config.val_split,
        seed=config.seed,
    )

    vocab = CharacterVocab.build(
        texts=train_targets,
        min_freq=config.min_freq,
        max_size=config.vocab_size,
    )
    vocab.save(os.path.join(config.work_dir, "vocab.json"))

    train_dataset = PairedCharacterDataset(
        contexts=train_contexts,
        targets=train_targets,
        vocab=vocab,
        max_seq_len=config.max_seq_len,
    )
    if len(train_dataset) == 0:
        raise ValueError("No training examples found.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = None
    if val_contexts:
        val_dataset = PairedCharacterDataset(
            contexts=val_contexts,
            targets=val_targets,
            vocab=vocab,
            max_seq_len=config.max_seq_len,
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_batch,
            )

    model_cfg = ModelConfig(
        vocab_size=len(vocab),
    )
    model = CanineLoRACharPredictor(config=model_cfg).to(runtime_device)

    optimizer = build_optimizer(
        model=model,
        lr_encoder_lora=config.lr_encoder_lora,
        lr_head=config.lr_head,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    metrics: List[Dict[str, float]] = []
    total_batches = len(train_loader)
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        print(f"Epoch {epoch + 1}/{config.num_epochs} started (batches={total_batches})")

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(runtime_device)
            attention_mask = batch["attention_mask"].to(runtime_device)
            labels = batch["labels"].to(runtime_device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if step % max(1, config.log_every) == 0 or step == total_batches:
                avg_so_far = total_loss / n_batches
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} "
                    f"batch {step}/{total_batches} "
                    f"loss={loss.item():.4f} avg_loss={avg_so_far:.4f}"
                )

        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": total_loss / max(1, n_batches),
            "epoch_seconds": float(time.time() - epoch_start),
        }
        if val_loader is not None:
            epoch_result["val_top3_acc"] = evaluate_top3_accuracy(model, val_loader, runtime_device)

        metrics.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))

    model.save(config.work_dir)
    with open(os.path.join(config.work_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with open(os.path.join(config.work_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    counts = model.count_parameters()
    print(
        f"Saved checkpoint to {config.work_dir} "
        f"(trainable={counts['trainable']}, total={counts['total']})"
    )


def parse_args() -> TrainConfig:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()

    return load_train_config(
        config_path=args.config,
        overrides={
            "work_dir": args.work_dir,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "device": args.device,
        },
    )


if __name__ == "__main__":
    train(parse_args())
