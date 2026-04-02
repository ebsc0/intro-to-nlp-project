#!/usr/bin/env python
import csv
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import List, Optional, Sequence, Tuple

import torch

from model import CharacterVocab, CanineLoRACharPredictor, text_to_codepoints

INPUT_PAD_CODEPOINT = 0
MIN_CANINE_INPUT_LEN = 4
DEFAULT_MAX_SEQ_LEN = 512

def build_batch(texts: Sequence[str], max_seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    contexts: List[List[int]] = []
    for text in texts:
        ids = text_to_codepoints(text, max_len=max_seq_len)
        if not ids:
            ids = [INPUT_PAD_CODEPOINT]
        contexts.append(ids)

    max_len = max(max(len(ids) for ids in contexts), MIN_CANINE_INPUT_LEN)
    input_ids = []
    attention_mask = []
    for ids in contexts:
        pad_count = max_len - len(ids)
        input_ids.append(ids + [INPUT_PAD_CODEPOINT] * pad_count)
        attention_mask.append([1] * len(ids) + [0] * pad_count)

    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
    )


def load_saved_max_seq_len(work_dir: str) -> int:
    config_path = os.path.join(work_dir, "train_config.json")
    if not os.path.isfile(config_path):
        return DEFAULT_MAX_SEQ_LEN

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    value = config.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)
    if isinstance(value, int) and value > 0:
        return value
    return DEFAULT_MAX_SEQ_LEN


class MyModel:
    def __init__(self, predictor: CanineLoRACharPredictor, vocab: CharacterVocab, max_seq_len: int, batch_size: int, device: torch.device) -> None:
        self.predictor = predictor
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device

    @classmethod
    def load_test_data(cls, fname: str) -> List[str]:
        with open(fname, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    @classmethod
    def load_test_csv(
        cls,
        fname: str,
        id_col: str = "id",
        context_col: str = "context",
    ) -> Tuple[List[str], List[str]]:
        ids: List[str] = []
        contexts: List[str] = []
        with open(fname, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(str(row[id_col]))
                contexts.append(str(row[context_col]))
        return ids, contexts

    @classmethod
    def write_pred_csv(cls, ids: Sequence[str], preds: Sequence[str], fname: str) -> None:
        out_dir = os.path.dirname(fname)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(fname, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction"])
            for sample_id, pred in zip(ids, preds):
                writer.writerow([sample_id, pred])

    @classmethod
    def load(
        cls,
        work_dir: str,
        batch_size: int = 32,
        max_seq_len: Optional[int] = None,
        device: str = "cuda",
    ) -> "MyModel":
        if device == "cuda":
            if torch.cuda.is_available():
                runtime_device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                runtime_device = torch.device("mps")
            else:
                runtime_device = torch.device("cpu")
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            runtime_device = torch.device("mps")
        else:
            runtime_device = torch.device("cpu")
        predictor = CanineLoRACharPredictor.load(work_dir, map_location="cpu")
        predictor = predictor.to(runtime_device)
        predictor.eval()

        vocab_path = os.path.join(work_dir, "vocab.json")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(f"Missing vocab file: {vocab_path}")
        vocab = CharacterVocab.load(vocab_path)

        resolved_max_seq_len = max_seq_len if max_seq_len is not None else load_saved_max_seq_len(work_dir)

        return cls(
            predictor=predictor,
            vocab=vocab,
            max_seq_len=resolved_max_seq_len,
            batch_size=batch_size,
            device=runtime_device,
        )

    @torch.no_grad()
    def predict_batches(self, texts: Sequence[str], k: int = 3):
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            input_ids, attention_mask = build_batch(batch_texts, self.max_seq_len, self.device)
            topk = self.predictor.predict_topk(input_ids=input_ids, attention_mask=attention_mask, k=k)

            batch_preds: List[str] = []
            for row in topk.tolist():
                chars = [self.vocab.get_char(idx) for idx in row]
                while len(chars) < k:
                    chars.append(" ")
                batch_preds.append("".join(chars[:k]))

            done = start + len(batch_texts)
            yield batch_preds, done, total

    def run_pred_to_file(self, data: Sequence[str], fname: str, progress_every: int = 1000) -> int:
        out_dir = os.path.dirname(fname)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        written = 0
        with open(fname, "wt", encoding="utf-8") as f:
            for batch_preds, done, total in self.predict_batches(data, k=3):
                for p in batch_preds:
                    f.write(f"{p}\n")
                written += len(batch_preds)
                f.flush()
                if done % max(1, progress_every) == 0 or done == total:
                    print(f"Progress: {done}/{total}")
        return written


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_dir", default="work")

    parser.add_argument("--test_data", default="")
    parser.add_argument("--test_output", default="output/pred.txt")
    parser.add_argument("--test_csv", default="")
    parser.add_argument("--test_csv_output", default="output/pred.csv")
    parser.add_argument("--test_csv_id_col", default="id")
    parser.add_argument("--test_csv_context_col", default="context")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    parser.add_argument("--progress_every", type=int, default=1000)
    args = parser.parse_args()

    if bool(args.test_data) == bool(args.test_csv):
        parser.error("Provide exactly one of --test_data or --test_csv.")

    model = MyModel.load(
        work_dir=args.work_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )
    if args.test_csv:
        print(f"Loading test CSV from {args.test_csv}")
        sample_ids, test_data = MyModel.load_test_csv(
            args.test_csv,
            id_col=args.test_csv_id_col,
            context_col=args.test_csv_context_col,
        )
        print("Making predictions")
        preds: List[str] = []
        for batch_preds, done, total in model.predict_batches(test_data, k=3):
            preds.extend(batch_preds)
            if done % max(1, args.progress_every) == 0 or done == total:
                print(f"Progress: {done}/{total}")
        print(f"Writing CSV predictions to {args.test_csv_output}")
        MyModel.write_pred_csv(sample_ids, preds, args.test_csv_output)
        return

    print(f"Loading test data from {args.test_data}")
    test_data = MyModel.load_test_data(args.test_data)
    print("Making predictions")
    print(f"Writing predictions to {args.test_output}")
    model.run_pred_to_file(test_data, args.test_output, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
