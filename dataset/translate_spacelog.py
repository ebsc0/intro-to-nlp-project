#!/usr/bin/env python3

import os
import re
import time

from dotenv import load_dotenv
from google.auth import exceptions as google_auth_exceptions
from google.cloud import translate
from tqdm import tqdm

IN_DIR = "data/spacelog/clean"
OUT_DIR = "data/spacelog/translations"

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "global"

SOURCE_LANG = "en-US"
SOURCE_KEY = "en"
TARGET_LANGS = ["en", "fr", "de", "it", "ru", "zh", "ja", "ko", "hi", "ar"]
TARGET_CODE = {
    "en": "en",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "ru": "ru",
    "zh": "zh-CN",
    "ja": "ja",
    "ko": "ko",
    "hi": "hi",
    "ar": "ar",
}

OVERWRITE = False
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.0
BATCH_SIZE = 64
BATCH_MAX_CHARS = 12000

NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
MULTI_SPACE_RE = re.compile(r" {2,}")
ELLIPSIS_RE = re.compile(r"\.{4,}")
TRAILING_PUNCT_RE = re.compile(r"([,.;:!?]+)$")


def list_text_files():
    files = []
    for root, _, names in os.walk(IN_DIR):
        for name in sorted(names):
            if name.endswith(".txt"):
                src_path = os.path.join(root, name)
                rel_path = os.path.relpath(src_path, IN_DIR)
                files.append((src_path, rel_path))
    return sorted(files, key=lambda x: x[1])


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def normalize_line(source_text, translated_text):
    text = " ".join((translated_text or "").splitlines()).strip()
    text = NBSP_RE.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = ELLIPSIS_RE.sub("...", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    src_match = TRAILING_PUNCT_RE.search(source_text.strip())
    dst_match = TRAILING_PUNCT_RE.search(text)
    if src_match and dst_match and src_match.group(1) != dst_match.group(1):
        text = text[: -len(dst_match.group(1))] + src_match.group(1)
    return text


def make_batches(lines):
    items = [(i, line) for i, line in enumerate(lines) if line.strip()]
    batches = []
    batch = []
    chars = 0

    for i, text in items:
        too_many_lines = len(batch) >= BATCH_SIZE
        too_many_chars = batch and chars + len(text) > BATCH_MAX_CHARS
        if too_many_lines or too_many_chars:
            batches.append(batch)
            batch = []
            chars = 0
        batch.append((i, text))
        chars += len(text)

    if batch:
        batches.append(batch)
    return batches


def translate_batch(client, parent, batch, target_lang):
    lines = [text for _, text in batch]
    request = {
        "parent": parent,
        "contents": lines,
        "mime_type": "text/plain",
        "source_language_code": SOURCE_LANG,
        "target_language_code": TARGET_CODE[target_lang],
    }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.translate_text(request=request)
            translated = [x.translated_text for x in response.translations]
            if len(translated) != len(lines):
                raise RuntimeError(
                    f"Batch size mismatch: got={len(translated)} expected={len(lines)}"
                )
            return [normalize_line(src, dst) for src, dst in zip(lines, translated)]
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Translate failed ({target_lang}): {last_error}") from last_error
            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))


def translate_lines(client, parent, lines, target_lang, desc):
    out = list(lines)
    batches = make_batches(lines)
    for batch in tqdm(batches, desc=desc, leave=False, unit="batch"):
        translated = translate_batch(client, parent, batch, target_lang)
        for (line_idx, _), text in zip(batch, translated):
            out[line_idx] = text
    return out


def run():
    if not PROJECT_ID:
        raise RuntimeError("Set GCP_PROJECT_ID (env or .env).")

    files = list_text_files()
    if not files:
        raise FileNotFoundError(f"No .txt files found in {IN_DIR}")

    try:
        client = translate.TranslationServiceClient()
    except google_auth_exceptions.DefaultCredentialsError as exc:
        raise RuntimeError(
            "Google credentials not found. Run `gcloud auth application-default login` "
            "or set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON file."
        ) from exc
    parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
    tasks = [(lang, src_path, rel_path) for lang in TARGET_LANGS for src_path, rel_path in files]

    for lang, src_path, rel_path in tqdm(tasks, desc="Translate files", unit="file"):
        out_path = os.path.join(OUT_DIR, lang, rel_path)
        if os.path.exists(out_path) and not OVERWRITE:
            continue

        src_lines = load_lines(src_path)
        if lang == SOURCE_KEY:
            out_lines = src_lines
        else:
            short_name = os.path.basename(rel_path)
            out_lines = translate_lines(
                client=client,
                parent=parent,
                lines=src_lines,
                target_lang=lang,
                desc=f"{lang}:{short_name}",
            )

        write_lines(out_path, out_lines)


def main():
    start = time.time()
    run()
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
