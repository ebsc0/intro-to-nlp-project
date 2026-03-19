#!/usr/bin/env python3
import argparse
import os
import random
from typing import Dict, Iterator, List, Sequence, Tuple

Example = Tuple[str, str, str]


def save_lines(path: str, lines: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def list_language_dirs(translations_dir: str) -> List[str]:
    if not os.path.isdir(translations_dir):
        raise FileNotFoundError(f"Translations directory not found: {translations_dir}")

    languages = sorted(
        name
        for name in os.listdir(translations_dir)
        if os.path.isdir(os.path.join(translations_dir, name))
    )
    if not languages:
        raise ValueError(f"No language directories found in {translations_dir}")
    return languages


def list_text_files(lang_dir: str) -> List[str]:
    files = sorted(
        name
        for name in os.listdir(lang_dir)
        if name.endswith(".txt") and os.path.isfile(os.path.join(lang_dir, name))
    )
    if not files:
        raise ValueError(f"No .txt files found in language directory: {lang_dir}")
    return files


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def validate_translations_dir(translations_dir: str) -> Tuple[List[str], List[str]]:
    languages = list_language_dirs(translations_dir)

    reference_lang = languages[0]
    reference_dir = os.path.join(translations_dir, reference_lang)
    reference_files = list_text_files(reference_dir)
    reference_counts = {
        filename: count_lines(os.path.join(reference_dir, filename)) for filename in reference_files
    }

    for lang in languages[1:]:
        lang_dir = os.path.join(translations_dir, lang)
        files = list_text_files(lang_dir)
        if files != reference_files:
            missing = sorted(set(reference_files) - set(files))
            extra = sorted(set(files) - set(reference_files))
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            detail_text = f" ({', '.join(details)})" if details else ""
            raise ValueError(
                f"Language directory {lang_dir} does not match {reference_dir}{detail_text}"
            )

        for filename in reference_files:
            path = os.path.join(lang_dir, filename)
            line_count = count_lines(path)
            expected_count = reference_counts[filename]
            if line_count != expected_count:
                raise ValueError(
                    "Mismatched line count for "
                    f"{filename}: {reference_lang} has {expected_count}, {lang} has {line_count}"
                )

    return languages, reference_files


def iter_prefix_examples(
    translations_dir: str,
    lang: str,
    filenames: Sequence[str],
) -> Iterator[Example]:
    lang_dir = os.path.join(translations_dir, lang)
    for filename in filenames:
        path = os.path.join(lang_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                text = raw_line.rstrip("\n")
                if len(text) < 2:
                    continue
                for i in range(1, len(text)):
                    yield text[:i], text[i], lang


def reservoir_sample(examples: Iterator[Example], k: int, rng: random.Random) -> List[Example]:
    if k <= 0:
        return []

    sample: List[Example] = []
    seen = 0
    for example in examples:
        seen += 1
        if len(sample) < k:
            sample.append(example)
            continue

        replace_idx = rng.randrange(seen)
        if replace_idx < k:
            sample[replace_idx] = example

    return sample


def create_dataset(
    translations_dir: str,
    out_dir: str,
    per_lang: int,
    seed: int,
) -> Tuple[int, Dict[str, int]]:
    if per_lang < 0:
        raise ValueError("--per_lang must be non-negative")

    languages, filenames = validate_translations_dir(translations_dir)

    rng = random.Random(seed)
    selected: List[Example] = []
    counts: Dict[str, int] = {}
    for lang in languages:
        sample = reservoir_sample(
            iter_prefix_examples(translations_dir=translations_dir, lang=lang, filenames=filenames),
            per_lang,
            rng,
        )
        selected.extend(sample)
        counts[lang] = len(sample)

    rng.shuffle(selected)

    inputs = [context for context, _, _ in selected]
    answers = [target for _, target, _ in selected]
    langs = [lang for _, _, lang in selected]

    os.makedirs(out_dir, exist_ok=True)
    save_lines(os.path.join(out_dir, "input.txt"), inputs)
    save_lines(os.path.join(out_dir, "answer.txt"), answers)
    save_lines(os.path.join(out_dir, "lang.txt"), langs)

    return len(selected), counts


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--translations_dir", default="data/spacelog/translations")
    parser.add_argument("--out_dir", default="data/training")
    parser.add_argument("--per_lang", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    total, counts = create_dataset(
        translations_dir=args.translations_dir,
        out_dir=args.out_dir,
        per_lang=args.per_lang,
        seed=args.seed,
    )
    print(f"Wrote {total} examples to {args.out_dir}")
    for lang in sorted(counts.keys()):
        print(f"{lang}: {counts[lang]}")


if __name__ == "__main__":
    main()
