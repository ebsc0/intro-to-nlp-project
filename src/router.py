import unicodedata
from collections import Counter
from typing import Dict, Sequence, Tuple

SCRIPT_BUCKETS: Tuple[str, ...] = (
    "latin",
    "cyrillic",
    "arabic",
    "devanagari",
    "hangul",
    "zh",
    "ja",
)
ALL_BUCKETS: Tuple[str, ...] = SCRIPT_BUCKETS + ("global",)

SCRIPT_RANGES: Dict[str, Sequence[Tuple[int, int]]] = {
    "latin": (
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00C0, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
    ),
    "cyrillic": (
        (0x0400, 0x04FF),
        (0x0500, 0x052F),
    ),
    "arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "devanagari": (
        (0x0900, 0x097F),
        (0xA8E0, 0xA8FF),
    ),
    "hangul": (
        (0x1100, 0x11FF),
        (0x3130, 0x318F),
        (0xAC00, 0xD7AF),
    ),
    "hiragana": (
        (0x3040, 0x309F),
    ),
    "katakana": (
        (0x30A0, 0x30FF),
        (0x31F0, 0x31FF),
    ),
    "cjk": (
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
    ),
}


def codepoint_group(char: str) -> str:
    category = unicodedata.category(char)
    if category.startswith(("Z", "P", "N", "S")):
        return ""

    codepoint = ord(char)
    for group, ranges in SCRIPT_RANGES.items():
        for start, end in ranges:
            if start <= codepoint <= end:
                return group
    return ""


def route_text(text: str) -> str:
    counts = Counter()
    for char in text:
        group = codepoint_group(char)
        if group:
            counts[group] += 1

    if counts["hiragana"] + counts["katakana"] > 0:
        return "ja"

    ranked = [
        ("hangul", counts["hangul"]),
        ("devanagari", counts["devanagari"]),
        ("arabic", counts["arabic"]),
        ("cyrillic", counts["cyrillic"]),
        ("cjk", counts["cjk"]),
        ("latin", counts["latin"]),
    ]
    ranked = [(bucket, score) for bucket, score in ranked if score > 0]
    if not ranked:
        return "global"

    ranked.sort(key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "global"

    bucket = ranked[0][0]
    if bucket == "cjk":
        return "zh"
    return bucket
