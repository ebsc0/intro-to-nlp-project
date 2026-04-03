#!/usr/bin/env python
import csv
import gc
import json
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from model import CharacterVocab, CanineLoRACharPredictor, text_to_codepoints
from router import ALL_BUCKETS, route_text

INPUT_PAD_CODEPOINT = 0
MIN_CANINE_INPUT_LEN = 4
DEFAULT_MAX_SEQ_LEN = 512


def build_batch(
    texts: Sequence[str],
    max_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def write_pred_txt(preds: Sequence[str], fname: str) -> None:
    out_dir = os.path.dirname(fname)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(fname, "w", encoding="utf-8") as f:
        for pred in preds:
            f.write(f"{pred}\n")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_split_script_work_dirs(args, parser: ArgumentParser) -> Dict[str, str]:
    work_dirs = {
        "global": args.global_work_dir,
        "latin": args.latin_work_dir,
        "cyrillic": args.cyrillic_work_dir,
        "arabic": args.arabic_work_dir,
        "devanagari": args.devanagari_work_dir,
        "hangul": args.hangul_work_dir,
        "zh": args.zh_work_dir,
        "ja": args.ja_work_dir,
    }
    missing = [bucket for bucket in ALL_BUCKETS if not work_dirs[bucket]]
    if missing:
        parser.error(
            "--split_script requires work dirs for all buckets: "
            + ", ".join(f"--{bucket}_work_dir" for bucket in missing)
        )
    return work_dirs


class MyModel:
    def __init__(
        self,
        predictor: CanineLoRACharPredictor,
        vocab: CharacterVocab,
        max_seq_len: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
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

        resolved_max_seq_len = (
            max_seq_len if max_seq_len is not None else load_saved_max_seq_len(work_dir)
        )

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
            topk = self.predictor.predict_topk(
                input_ids=input_ids,
                attention_mask=attention_mask,
                k=k,
            )

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
                for pred in batch_preds:
                    f.write(f"{pred}\n")
                written += len(batch_preds)
                f.flush()
                if done % max(1, progress_every) == 0 or done == total:
                    print(f"Progress: {done}/{total}")
        return written


def predict_with_split_script(
    texts: Sequence[str],
    work_dirs: Dict[str, str],
    batch_size: int,
    max_seq_len: Optional[int],
    device: str,
    progress_every: int,
) -> List[str]:
    routed_indices: Dict[str, List[int]] = {bucket: [] for bucket in ALL_BUCKETS}
    for idx, text in enumerate(texts):
        routed_indices[route_text(text)].append(idx)

    preds = [""] * len(texts)
    total_done = 0
    for bucket in ALL_BUCKETS:
        indices = routed_indices[bucket]
        if not indices:
            continue

        print(f"Routing bucket {bucket}: {len(indices)}")
        model = MyModel.load(
            work_dir=work_dirs[bucket],
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=device,
        )
        subset = [texts[idx] for idx in indices]
        offset = 0
        for batch_preds, done, bucket_total in model.predict_batches(subset, k=3):
            batch_indices = indices[offset : offset + len(batch_preds)]
            for original_idx, pred in zip(batch_indices, batch_preds):
                preds[original_idx] = pred
            offset += len(batch_preds)
            if done % max(1, progress_every) == 0 or done == bucket_total:
                print(f"{bucket} progress: {done}/{bucket_total}")

        total_done += len(indices)
        print(f"Overall progress: {total_done}/{len(texts)}")
        model_device = model.device
        del model
        gc.collect()
        clear_device_cache(model_device)

    return preds


def main() -> None:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work_dir", default="work")
    parser.add_argument("--split_script", action="store_true")
    parser.add_argument("--global_work_dir", default="")
    parser.add_argument("--latin_work_dir", default="")
    parser.add_argument("--cyrillic_work_dir", default="")
    parser.add_argument("--arabic_work_dir", default="")
    parser.add_argument("--devanagari_work_dir", default="")
    parser.add_argument("--hangul_work_dir", default="")
    parser.add_argument("--zh_work_dir", default="")
    parser.add_argument("--ja_work_dir", default="")

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

    split_work_dirs = None
    if args.split_script:
        split_work_dirs = build_split_script_work_dirs(args, parser)

    if args.test_csv:
        print(f"Loading test CSV from {args.test_csv}")
        sample_ids, test_data = MyModel.load_test_csv(
            args.test_csv,
            id_col=args.test_csv_id_col,
            context_col=args.test_csv_context_col,
        )
        print("Making predictions")
        if args.split_script:
            preds = predict_with_split_script(
                texts=test_data,
                work_dirs=split_work_dirs,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                device=args.device,
                progress_every=args.progress_every,
            )
        else:
            model = MyModel.load(
                work_dir=args.work_dir,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                device=args.device,
            )
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
    if args.split_script:
        preds = predict_with_split_script(
            texts=test_data,
            work_dirs=split_work_dirs,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            device=args.device,
            progress_every=args.progress_every,
        )
        write_pred_txt(preds, args.test_output)
    else:
        model = MyModel.load(
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            device=args.device,
        )
        model.run_pred_to_file(test_data, args.test_output, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
