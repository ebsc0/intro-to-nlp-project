#!/usr/bin/env python3

import html
import os
import re
import unicodedata

IN_DIR = "data/spacelog"
OUT_DIR = "data/spacelog/clean"

TIMESTAMP_RE = re.compile(r"^\[\s*[+-]?\d{1,2}:\d{2}:\d{2}(?::\d{2})?\s*\]$")
METADATA_RE = re.compile(r"^_[a-zA-Z0-9]+\s*:")
SPEAKER_RE = re.compile(r"^[\w .'\-()/]{1,64}:\s*", flags=re.UNICODE)
BRACKET_TAG_RE = re.compile(r"\[[^\]\n]*:[^\]\n]*\]")
HTML_TAG_RE = re.compile(r"</?\s*[A-Za-z][A-Za-z0-9:_-]*(?:\s+[^<>]*?)?>", flags=re.IGNORECASE)
LEADING_STAGE_RE = re.compile(r"^(?:\([^)\n]{1,40}\)\s*)+")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
MULTI_SPACE_RE = re.compile(r" {2,}")


def remove_control_chars(text):
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def is_number_punct_only(line):
    s = line.strip()
    if not s:
        return True
    for ch in s:
        if ch.isspace():
            continue
        if unicodedata.category(ch)[0] not in {"N", "P", "S"}:
            return False
    return True


def clean_line(raw_line):
    line = raw_line.replace("\t", " ")
    line = remove_control_chars(line)
    stripped = line.strip()
    if not stripped:
        return ""
    if TIMESTAMP_RE.match(stripped):
        return ""
    if METADATA_RE.match(stripped):
        return ""

    text = SPEAKER_RE.sub("", stripped)
    text = BRACKET_TAG_RE.sub("", text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub("", text)  # keep inner text, remove tags only
    text = LEADING_STAGE_RE.sub("", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = text.replace(".,", ",")
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""
    if is_number_punct_only(text):
        return ""
    return text


def preprocess_file(src_path, dst_path):
    with open(src_path, "r", encoding="utf-8") as f:
        raw = f.read()

    lines = []
    for raw_line in raw.split("\n"):
        was_indented = raw_line.startswith(" ")
        text = clean_line(raw_line)
        if not text:
            continue
        if was_indented and lines:
            lines[-1] = MULTI_SPACE_RE.sub(" ", f"{lines[-1]} {text}").strip()
        else:
            lines.append(text)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")
    return len(lines)


def preprocess_all():
    if not os.path.isdir(IN_DIR):
        raise FileNotFoundError(f"Directory not found: {IN_DIR}")

    abs_out = os.path.abspath(OUT_DIR)
    files_written = 0
    lines_written = 0

    for root, _, files in os.walk(IN_DIR):
        if os.path.abspath(root).startswith(abs_out):
            continue
        for filename in files:
            src_path = os.path.join(root, filename)
            rel = os.path.relpath(src_path, IN_DIR)
            rel_base, _ = os.path.splitext(rel)
            dst_path = os.path.join(OUT_DIR, f"{rel_base}.txt")
            lines_written += preprocess_file(src_path, dst_path)
            files_written += 1
    return files_written, lines_written


def main():
    files_written, lines_written = preprocess_all()
    print(f"Preprocessed {files_written} files to {OUT_DIR}/ ({lines_written} lines)")


if __name__ == "__main__":
    main()
