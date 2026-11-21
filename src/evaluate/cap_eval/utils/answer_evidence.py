"""Check that evidence quoted by an extractor actually appears in the answer."""


def contains_quote(answer: str, quote: str | None) -> bool:
    if not quote or not quote.strip():
        return False
    return " ".join(quote.casefold().split()) in " ".join(answer.casefold().split())
