"""Calendar predicates for extracted publication and itinerary dates."""

import calendar
import re
from datetime import date, timedelta


_MONTHS = dict(zip(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".lower().split(), range(1, 13)
))
_MONTH = (r"(?P<name>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
          r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?|"
          r"Nov(?:ember)?|Dec(?:ember)?)\b\.?")
_DAY = r"(?P<day>\d{1,2})(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>\d{4})(?!\d)"
_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    rf"(?<!\d){_YEAR}[-/.年](?P<month>\d{{1,2}})(?:[-/.月]{_DAY})?(?!\d)",
    rf"(?<!\d)(?P<month>\d{{1,2}})/{_DAY}(?:/{_YEAR})?(?![\d/])",
    rf"\b{_DAY}\s+{_MONTH}(?:[ ,]+{_YEAR})?(?!\w)",
    rf"\b{_MONTH}\s+{_YEAR}",
    rf"\b{_MONTH}\s+{_DAY}(?:[ ,]+{_YEAR})?(?!\w)",
    rf"^{_YEAR}$",
)]


def _calendar_parts(text):
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            values = match.groupdict()
            month = _MONTHS[values["name"][:3].lower()] if values.get("name") else values.get("month")
            return tuple(int(v) if v else None for v in (values.get("year"), month, values.get("day")))
    return None


def months_before(day: date, months: int) -> date:
    """Subtract calendar months, clipping the day at the end of shorter months."""
    year, zero_based_month = divmod(day.year * 12 + day.month - 1 - months, 12)
    month = zero_based_month + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def publication_in_window(text: str | None, start: date, as_of: date, *, require_month=False) -> bool:
    """Match a past release at its stated precision, including month-only dates."""
    if not text:
        return False
    value = text.strip().lower()
    if value == "today":
        return start <= as_of
    if value == "yesterday":
        return start <= as_of - timedelta(days=1)
    relative = re.fullmatch(r"(\d+)\s+(day|week|month|year)s?\s+ago", value)
    if relative:
        n, unit = int(relative[1]), relative[2]
        try:
            if unit in ("month", "year"):
                released = months_before(as_of, n if unit == "month" else n * 12)
            else:
                released = as_of - timedelta(days=n if unit == "day" else n * 7)
        except (ValueError, OverflowError):
            return False
        return start <= released <= as_of

    parts = _calendar_parts(value)
    if not parts:
        return False
    year, month, day = parts
    if not year or (require_month and not month):
        return False
    try:
        first_month = 1 if month is None else month
        last_month = 12 if month is None else month
        earliest = date(year, first_month, 1 if day is None else day)
        last_day = calendar.monthrange(year, last_month)[1] if day is None else day
        latest = date(year, last_month, last_day)
    except ValueError:
        return False
    return earliest <= as_of and latest >= start


_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def coming_friday(as_of: date) -> date:
    """The first Friday on or after the reference date."""
    return as_of + timedelta(days=(4 - as_of.weekday()) % 7)


def on_itinerary_day(text: str | None, expected: date) -> bool:
    """Allow an omitted year or a weekday within the specified trip."""
    if not text:
        return False
    parts = _calendar_parts(text.strip())
    if parts:
        year, month, day = parts
        return year in (None, expected.year) and (month, day) == (expected.month, expected.day)
    # A weekday is unambiguous on this itinerary. Do not use it to override
    # a contradictory explicit date or a different relative week.
    weekday = _WEEKDAYS[expected.weekday()]
    return bool(re.fullmatch(
        rf"(?:this (?:coming )?)?(?:{weekday}|{weekday[:3]}\.?)"
        r"(?:\s+(?:morning|afternoon|evening|night))?"
        r"(?:[ ,]+(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?(?:\s+[A-Z]{2,4})?)?",
        text.strip(), re.IGNORECASE,
    ))
