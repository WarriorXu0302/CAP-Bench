"""Parse the first number in a grader's extracted, English-language field."""

import re

# A comma is part of a number only when it groups three digits. Removing
# commas from the whole string would also join separate values such as "2, 5".
_INTEGER_PART = r"(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
_INTEGER = re.compile(r"[+-]?" + _INTEGER_PART)
_DECIMAL = re.compile(r"[+-]?(?:" + _INTEGER_PART + r"(?:\.\d+)?|\.\d+)")


def extract_float(text: str | float | int | None) -> float | None:
    match = _DECIMAL.search(str(text)) if text is not None else None
    return float(match[0].replace(",", "")) if match else None


def extract_int(text: str | int | None) -> int | None:
    match = _INTEGER.search(str(text)) if text is not None else None
    if not match:
        return None
    try:
        return int(match[0].replace(",", ""))
    except ValueError:
        # Python limits integer conversions of excessively long digit strings.
        return None
