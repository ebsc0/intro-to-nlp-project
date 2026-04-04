import json
import os
import random
import shutil
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from model import CharacterVocab, CanineLoRACharPredictor, ModelConfig, text_to_codepoints

INPUT_PAD_CODEPOINT = 0
MIN_CANINE_INPUT_LEN = 4
CHECKPOINT_BEST = "best"
CHECKPOINT_LAST = "last"
EARLY_STOPPING_PATIENCE = 2


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
        langs: Optional[Sequence[str]] = None,
        max_seq_len: int = 512,
    ) -> None:
        self.contexts = list(contexts)
        self.target_ids = [vocab.get_id(t) for t in targets]
        self.langs = list(langs) if langs is not None else None
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, str]:
        ids = text_to_codepoints(self.contexts[idx], max_len=self.max_seq_len)
        if not ids:
            ids = [INPUT_PAD_CODEPOINT]
        lang = self.langs[idx] if self.langs is not None else ""
        return ids, self.target_ids[idx], lang


def collate_batch(batch: Sequence[Tuple[List[int], int, str]]) -> Dict[str, Any]:
    max_len = max(max(len(context_ids) for context_ids, _, _ in batch), MIN_CANINE_INPUT_LEN)
    input_ids = []
    attention_masks = []
    labels = []
    langs: List[str] = []
    for context_ids, label, lang in batch:
        pad_count = max_len - len(context_ids)
        input_ids.append(context_ids + [INPUT_PAD_CODEPOINT] * pad_count)
        attention_masks.append([1] * len(context_ids) + [0] * pad_count)
        labels.append(label)
        langs.append(lang)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "langs": langs,
    }


def read_paired_data(input_path: str, answer_path: str) -> Tuple[List[str], List[str], Optional[List[str]]]:
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

    lang_path = os.path.join(os.path.dirname(input_path), "lang.txt")
    langs: Optional[List[str]] = None
    if os.path.isfile(lang_path):
        with open(lang_path, "r", encoding="utf-8") as f:
            langs = [line.rstrip("\n") for line in f]
        if len(langs) != len(contexts):
            raise ValueError(
                "Input/lang line count mismatch: "
                f"{input_path} has {len(contexts)} lines, "
                f"{lang_path} has {len(langs)} lines."
            )

    return contexts, targets, langs


def split_train_val(
    contexts: Sequence[str],
    targets: Sequence[str],
    langs: Optional[Sequence[str]],
    val_split: float,
    seed: int,
) -> Tuple[List[str], List[str], Optional[List[str]], List[str], List[str], Optional[List[str]]]:
    if len(contexts) != len(targets):
        raise ValueError(
            f"Contexts and targets must have the same length, got {len(contexts)} and {len(targets)}."
        )
    if langs is not None and len(contexts) != len(langs):
        raise ValueError(
            f"Contexts and langs must have the same length, got {len(contexts)} and {len(langs)}."
        )

    idxs = list(range(len(contexts)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_size = int(len(idxs) * val_split)

    val_idxs = set(idxs[:val_size])
    train_contexts: List[str] = []
    train_targets: List[str] = []
    train_langs: Optional[List[str]] = [] if langs is not None else None
    val_contexts: List[str] = []
    val_targets: List[str] = []
    val_langs: Optional[List[str]] = [] if langs is not None else None

    for i in range(len(contexts)):
        context = contexts[i]
        target = targets[i]
        lang = langs[i] if langs is not None else ""
        if i in val_idxs:
            val_contexts.append(context)
            val_targets.append(target)
            if val_langs is not None:
                val_langs.append(lang)
        else:
            train_contexts.append(context)
            train_targets.append(target)
            if train_langs is not None:
                train_langs.append(lang)

    return train_contexts, train_targets, train_langs, val_contexts, val_targets, val_langs


@torch.no_grad()
def evaluate_top3_accuracy(
    model: CanineLoRACharPredictor,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    model.eval()
    correct = 0
    total = 0
    correct_by_lang: Dict[str, int] = defaultdict(int)
    total_by_lang: Dict[str, int] = defaultdict(int)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        langs = batch["langs"]

        top3 = model.predict_topk(input_ids=input_ids, attention_mask=attention_mask, k=3)
        matches = (top3 == labels.unsqueeze(-1)).any(dim=-1)
        correct += int(matches.sum().item())
        total += int(labels.size(0))
        for lang, matched in zip(langs, matches.tolist()):
            if not lang:
                continue
            total_by_lang[lang] += 1
            if matched:
                correct_by_lang[lang] += 1

    if total == 0:
        return 0.0, {}, {}

    per_lang_acc = {
        lang: correct_by_lang[lang] / total_by_lang[lang]
        for lang in sorted(total_by_lang.keys())
        if total_by_lang[lang] > 0
    }
    per_lang_counts = {lang: total_by_lang[lang] for lang in sorted(total_by_lang.keys())}
    return correct / total, per_lang_acc, per_lang_counts


def maybe_init_wandb(enabled: bool, train_config: TrainConfig, model_config: ModelConfig):
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise ImportError("`--wandb` was set but `wandb` is not installed.") from exc

    tags_env = os.getenv("WANDB_TAGS", "")
    tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]
    return wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        name=os.getenv("WANDB_NAME"),
        mode=os.getenv("WANDB_MODE"),
        group=os.getenv("WANDB_GROUP"),
        tags=tags or None,
        config={
            "train": asdict(train_config),
            "model": asdict(model_config),
        },
    )


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


def checkpoint_dir(work_dir: str, checkpoint_name: str) -> str:
    return os.path.join(work_dir, checkpoint_name)


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def save_checkpoint(
    model: CanineLoRACharPredictor,
    vocab: CharacterVocab,
    config: TrainConfig,
    checkpoint_path: str,
) -> None:
    remove_path(checkpoint_path)
    model.save(checkpoint_path)
    vocab.save(os.path.join(checkpoint_path, "vocab.json"))
    with open(os.path.join(checkpoint_path, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def copy_checkpoint(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        raise FileNotFoundError(f"Cannot copy missing checkpoint directory: {src}")
    remove_path(dst)
    shutil.copytree(src, dst)


def train(config: TrainConfig, use_wandb: bool = False) -> None:
    os.makedirs(config.work_dir, exist_ok=True)
    best_dir = checkpoint_dir(config.work_dir, CHECKPOINT_BEST)
    last_dir = checkpoint_dir(config.work_dir, CHECKPOINT_LAST)
    remove_path(best_dir)
    remove_path(last_dir)

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

    contexts, targets, langs = read_paired_data(config.input_path, config.answer_path)
    (
        train_contexts,
        train_targets,
        train_langs,
        val_contexts,
        val_targets,
        val_langs,
    ) = split_train_val(
        contexts=contexts,
        targets=targets,
        langs=langs,
        val_split=config.val_split,
        seed=config.seed,
    )

    vocab = CharacterVocab.build(
        texts=train_targets,
        min_freq=config.min_freq,
        max_size=config.vocab_size,
    )

    train_dataset = PairedCharacterDataset(
        contexts=train_contexts,
        targets=train_targets,
        vocab=vocab,
        langs=train_langs,
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
            langs=val_langs,
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
    wandb_run = maybe_init_wandb(use_wandb, config, model_cfg)

    optimizer = build_optimizer(
        model=model,
        lr_encoder_lora=config.lr_encoder_lora,
        lr_head=config.lr_head,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    metrics: List[Dict[str, Any]] = []
    total_batches = len(train_loader)
    global_step = 0
    best_val: Optional[float] = None
    best_epoch: Optional[int] = None
    best_is_last_only = False
    epochs_without_improvement = 0
    stopped_early = False
    last_epoch = 0
    for epoch in range(config.num_epochs):
        if val_loader is not None and best_is_last_only and epoch > 0:
            copy_checkpoint(last_dir, best_dir)
            best_is_last_only = False

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
            global_step += 1
            if step % max(1, config.log_every) == 0 or step == total_batches:
                avg_so_far = total_loss / n_batches
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs} "
                    f"batch {step}/{total_batches} "
                    f"loss={loss.item():.4f} avg_loss={avg_so_far:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/batch_loss": float(loss.item()),
                            "train/avg_loss": float(avg_so_far),
                            "epoch": float(epoch + 1),
                            "step": int(global_step),
                        }
                    )

        save_checkpoint(model, vocab, config, last_dir)
        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": total_loss / max(1, n_batches),
            "epoch_seconds": float(time.time() - epoch_start),
        }
        improved = False
        if val_loader is not None:
            val_acc, val_by_lang, val_counts = evaluate_top3_accuracy(model, val_loader, runtime_device)
            epoch_result["val_top3_acc"] = val_acc
            if val_by_lang:
                epoch_result["val_top3_acc_by_lang"] = val_by_lang
                epoch_result["val_count_by_lang"] = val_counts
                for lang in sorted(val_by_lang.keys()):
                    print(
                        f"Validation {lang}: "
                        f"count={val_counts[lang]} "
                        f"top3_acc={val_by_lang[lang]:.5f}"
                    )
            improved = best_val is None or val_acc > best_val
            if improved:
                best_val = val_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                best_is_last_only = True
                remove_path(best_dir)
            else:
                epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        epoch_result["is_best"] = improved
        epoch_result["epochs_without_improvement"] = epochs_without_improvement

        metrics.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        if wandb_run is not None:
            log_payload: Dict[str, Any] = {
                "train/epoch_loss": float(epoch_result["train_loss"]),
                "train/epoch_seconds": float(epoch_result["epoch_seconds"]),
                "epoch": float(epoch + 1),
                "checkpoint/is_best": bool(improved),
                "checkpoint/epochs_without_improvement": int(epochs_without_improvement),
            }
            if "val_top3_acc" in epoch_result:
                log_payload["val/top3_acc"] = float(epoch_result["val_top3_acc"])
            for lang, acc in epoch_result.get("val_top3_acc_by_lang", {}).items():
                log_payload[f"val/by_lang/{lang}"] = float(acc)
            wandb_run.log(log_payload)

        last_epoch = epoch + 1
        if val_loader is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            print(
                f"Early stopping triggered at epoch {last_epoch} "
                f"after {epochs_without_improvement} non-improving epochs."
            )
            break

    with open(os.path.join(config.work_dir, "train_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    summary_best_epoch = best_epoch if best_epoch is not None else (last_epoch if last_epoch > 0 else None)
    default_checkpoint = CHECKPOINT_LAST if best_is_last_only or val_loader is None else CHECKPOINT_BEST
    with open(os.path.join(config.work_dir, "checkpoint_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_checkpoint": default_checkpoint,
                "best_epoch": summary_best_epoch,
                "best_val_top3_acc": best_val,
                "last_epoch": last_epoch,
                "stopped_early": stopped_early,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            f,
            indent=2,
        )

    counts = model.count_parameters()
    print(
        f"Saved checkpoints under {config.work_dir} "
        f"(default_checkpoint={default_checkpoint}, "
        f"trainable={counts['trainable']}, total={counts['total']})"
    )
    if wandb_run is not None:
        wandb_run.finish()


def parse_args() -> Tuple[TrainConfig, bool]:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    return (
        load_train_config(
            config_path=args.config,
            overrides={
                "work_dir": args.work_dir,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "device": args.device,
            },
        ),
        args.wandb,
    )


if __name__ == "__main__":
    train(*parse_args())
