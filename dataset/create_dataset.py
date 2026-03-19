#!/usr/bin/env python
import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def save_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def create_dataset(
    input_path: str,
    answer_path: str,
    lang_path: str,
    out_dir: str,
    per_lang: int,
    seed: int,
) -> Tuple[int, Dict[str, int]]:
    inputs = load_lines(input_path)
    answers = load_lines(answer_path)
    langs = load_lines(lang_path)

    if not (len(inputs) == len(answers) == len(langs)):
        raise ValueError("input.txt, answer.txt, and lang.txt must have the same number of lines")

    by_lang: Dict[str, List[int]] = defaultdict(list)
    for i, lang in enumerate(langs):
        by_lang[lang].append(i)

    rng = random.Random(seed)
    selected: List[int] = []
    counts: Dict[str, int] = {}
    for lang in sorted(by_lang.keys()):
        indices = by_lang[lang]
        rng.shuffle(indices)
        take = min(per_lang, len(indices))
        selected.extend(indices[:take])
        counts[lang] = take

    rng.shuffle(selected)

    out_inputs = [inputs[i] for i in selected]
    out_answers = [answers[i] for i in selected]
    out_langs = [langs[i] for i in selected]

    os.makedirs(out_dir, exist_ok=True)
    save_lines(os.path.join(out_dir, "input.txt"), out_inputs)
    save_lines(os.path.join(out_dir, "answer.txt"), out_answers)
    save_lines(os.path.join(out_dir, "lang.txt"), out_langs)

    return len(selected), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/open-dev/input.txt")
    parser.add_argument("--answer", default="data/open-dev/answer.txt")
    parser.add_argument("--lang", default="data/open-dev/lang.txt")
    parser.add_argument("--out_dir", default="data/open-dev/baseline")
    parser.add_argument("--per_lang", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total, counts = create_dataset(
        input_path=args.input,
        answer_path=args.answer,
        lang_path=args.lang,
        out_dir=args.out_dir,
        per_lang=args.per_lang,
        seed=args.seed,
    )
    print(f"Wrote {total} examples to {args.out_dir}")
    for lang in sorted(counts.keys()):
        print(f"{lang}: {counts[lang]}")


if __name__ == "__main__":
    main()
