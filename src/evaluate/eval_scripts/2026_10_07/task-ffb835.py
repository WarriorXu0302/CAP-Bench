import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-ffb835"
TASK_DESCRIPTION = 'I\'m a broke student looking to pass the time during the holidays. I\'m debating whether to buy and play "The Witcher 3: Wild Hunt" or read the first book of "The Witcher" original series, "The Last Wish." Please help me calculate the "entertainment cost per hour" for each to aid my decision.\n\nFirst, go to HowLongToBeat and find the average completion time for "The Witcher 3" in "Main + Extras" mode. Then, check its current price on Steam. While there, please also identify the minimum RAM required to run it, to help me gauge if my system can handle it.\n\nNext, go to Goodreads and find the page count for the paperback edition of "The Last Wish" by Andrzej Sapkowski. Afterward, search Google Books for the price of its e-book version. Assume I read one page every 2 minutes.\n\nFinally, calculate and compare the "price per hour" for both options and tell me which one is more cost-effective.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GameInfo(BaseModel):
    """Game information extracted from the answer for The Witcher 3"""
    completion_time_text: Optional[str] = None
    steam_price_text: Optional[str] = None
    minimum_ram_text: Optional[str] = None


class BookInfo(BaseModel):
    """Book information extracted from the answer for The Last Wish"""
    page_count_text: Optional[str] = None
    ebook_price_text: Optional[str] = None


class CostComparison(BaseModel):
    """Cost per hour comparison extracted from the answer"""
    game_cost_per_hour_text: Optional[str] = None
    book_cost_per_hour_text: Optional[str] = None
    recommendation_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_game_info_from_answer() -> str:
    return """
Extract the game information for "The Witcher 3: Wild Hunt" from the answer:

Return:
- completion_time_text: the average completion time for "Main + Extras" mode from HowLongToBeat exactly as stated (include units if present).
- steam_price_text: the current price on Steam exactly as stated (include currency symbols if present).
- minimum_ram_text: the minimum RAM required to run the game exactly as stated (include units if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_book_info_from_answer() -> str:
    return """
Extract the book information for "The Last Wish" by Andrzej Sapkowski from the answer:

Return:
- page_count_text: the page count for the paperback edition from Goodreads exactly as stated.
- ebook_price_text: the e-book price from Google Books exactly as stated (include currency symbols if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_cost_comparison_from_answer() -> str:
    return """
Extract the cost per hour comparison and recommendation from the answer:

Return:
- game_cost_per_hour_text: the calculated price per hour for The Witcher 3 exactly as stated.
- book_cost_per_hour_text: the calculated price per hour for The Last Wish exactly as stated.
- recommendation_text: which option is recommended as more cost-effective.

If any field is missing in the answer, set it to null.
"""


# --------------------------------------------------------------------------- #
# Helper functions for lenient checks                                         #
# --------------------------------------------------------------------------- #
def ci_contains(text: Optional[str], substr: str) -> bool:
    if not text:
        return False
    return substr.lower() in text.lower()


def has_any_ci(text: Optional[str], substrs: List[str]) -> bool:
    return any(ci_contains(text, s) for s in substrs)


def contains_digits(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\d', text))


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_time_hours(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['hour', 'hr', 'h'])


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['$', '€', '£', '¥', 'usd', 'eur', 'gbp']) or re.search(r'\d+\.\d{2}', text)


def looks_like_ram(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['gb', 'mb', 'ram', 'memory'])


def looks_like_page_count(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return has_any_ci(text, ['page', 'pg']) or (num > 50 and num < 2000)


def looks_like_cost_per_hour(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    has_currency = has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp', 'per hour', '/hour', '/hr'])
    has_number = extract_float(text) is not None
    return has_currency and has_number


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: Any,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    Restrict evaluator.verify to at most one usage (we don't use it here).
    Favor lenient, fault-tolerant checks and allow partial credit.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()

    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract information from the answer --------------------- #
    game_info = await evaluator.extract(
        prompt=prompt_extract_game_info_from_answer(),
        template_class=GameInfo,
        extraction_name="game_info"
    )

    book_info = await evaluator.extract(
        prompt=prompt_extract_book_info_from_answer(),
        template_class=BookInfo,
        extraction_name="book_info"
    )

    cost_comparison = await evaluator.extract(
        prompt=prompt_extract_cost_comparison_from_answer(),
        template_class=CostComparison,
        extraction_name="cost_comparison"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 HowLongToBeat section
    hltb_node = evaluator.add_sequential(
        id="howlongtobeat_section",
        desc="HowLongToBeat information for The Witcher 3",
        parent=root,
        critical=False
    )

    # Check if answer mentions HowLongToBeat
    hltb_mention = has_any_ci(answer, ['howlongtobeat', 'how long to beat', 'hltb'])
    evaluator.add_custom_node(
        result=bool(hltb_mention),
        id="hltb_action_visit",
        desc="Mentions visiting HowLongToBeat website",
        parent=hltb_node,
        critical=False
    )

    # [Perception Node] howlongtobeat.com:F1:P1 - Extract Main + Extras completion time
    completion_time_ok = looks_like_time_hours(game_info.completion_time_text)
    main_extras_mention = has_any_ci(answer, ['main + extras', 'main+extras', 'main and extras'])

    evaluator.add_custom_node(
        result=bool(completion_time_ok and main_extras_mention),
        id="hltb_perception_main_extras",
        desc="[Perception Node] howlongtobeat.com:F1:P1 - Extract average completion time for 'Main + Extras' mode",
        parent=hltb_node,
        critical=False
    )

    # 3.2 Steam section
    steam_node = evaluator.add_sequential(
        id="steam_section",
        desc="Steam information for The Witcher 3",
        parent=root,
        critical=False
    )

    # Check if answer mentions Steam
    steam_mention = has_any_ci(answer, ['steam'])
    evaluator.add_custom_node(
        result=bool(steam_mention),
        id="steam_action_visit",
        desc="Mentions visiting Steam platform",
        parent=steam_node,
        critical=False
    )

    # Check Steam price extraction
    steam_price_ok = looks_like_price(game_info.steam_price_text)
    evaluator.add_custom_node(
        result=bool(steam_price_ok),
        id="steam_perception_price",
        desc="Extract current price of The Witcher 3 on Steam",
        parent=steam_node,
        critical=False
    )

    # [Perception Node] store.steampowered.com:F2:P7 - Extract minimum RAM from system requirements
    min_ram_ok = looks_like_ram(game_info.minimum_ram_text)
    system_req_mention = has_any_ci(answer, ['system requirements', 'minimum', 'requirements', 'specs'])

    evaluator.add_custom_node(
        result=bool(min_ram_ok and system_req_mention),
        id="steam_perception_min_ram",
        desc="[Perception Node] store.steampowered.com:F2:P7 - Extract minimum RAM required from system requirements table",
        parent=steam_node,
        critical=False
    )

    # 3.3 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads information for The Last Wish",
        parent=root,
        critical=False
    )

    # Check if answer mentions Goodreads
    goodreads_mention = has_any_ci(answer, ['goodreads'])
    evaluator.add_custom_node(
        result=bool(goodreads_mention),
        id="goodreads_action_visit",
        desc="Mentions visiting Goodreads website",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P5 - Extract page count for paperback edition
    page_count_ok = looks_like_page_count(book_info.page_count_text)
    paperback_mention = has_any_ci(answer, ['paperback', 'paper back'])
    last_wish_mention = has_any_ci(answer, ['last wish', 'andrzej sapkowski'])

    evaluator.add_custom_node(
        result=bool(page_count_ok and paperback_mention and last_wish_mention),
        id="goodreads_perception_page_count",
        desc="[Perception Node] goodreads.com:F2:P5 - Extract page count for paperback edition of The Last Wish",
        parent=goodreads_node,
        critical=False
    )

    # 3.4 Google Books section
    google_books_node = evaluator.add_sequential(
        id="google_books_section",
        desc="Google Books information for The Last Wish",
        parent=root,
        critical=False
    )

    # Check if answer mentions Google Books
    google_books_mention = has_any_ci(answer, ['google books', 'google book'])
    evaluator.add_custom_node(
        result=bool(google_books_mention),
        id="google_books_action_visit",
        desc="Mentions visiting Google Books",
        parent=google_books_node,
        critical=False
    )

    # [Perception Node] books.google.com:F5:P4 - Extract e-book price
    ebook_price_ok = looks_like_price(book_info.ebook_price_text)
    ebook_mention = has_any_ci(answer, ['e-book', 'ebook', 'electronic', 'digital'])

    evaluator.add_custom_node(
        result=bool(ebook_price_ok and ebook_mention),
        id="google_books_perception_ebook_price",
        desc="[Perception Node] books.google.com:F5:P4 - Extract e-book price from purchase/acquisition links area",
        parent=google_books_node,
        critical=False
    )

    # 3.5 Cost comparison and calculation section
    comparison_node = evaluator.add_sequential(
        id="cost_comparison_section",
        desc="Calculate and compare cost per hour for both options",
        parent=root,
        critical=False
    )

    # Check if reading speed assumption is mentioned
    reading_speed_mention = has_any_ci(answer, ['2 minutes', 'two minutes', 'page every 2'])
    evaluator.add_custom_node(
        result=bool(reading_speed_mention),
        id="comparison_reading_speed_assumption",
        desc="Mentions the reading speed assumption (1 page per 2 minutes)",
        parent=comparison_node,
        critical=False
    )

    # Check game cost per hour calculation
    game_cost_per_hour_ok = looks_like_cost_per_hour(cost_comparison.game_cost_per_hour_text)
    evaluator.add_custom_node(
        result=bool(game_cost_per_hour_ok),
        id="comparison_game_cost_calculation",
        desc="Calculate price per hour for The Witcher 3 game",
        parent=comparison_node,
        critical=False
    )

    # Check book cost per hour calculation
    book_cost_per_hour_ok = looks_like_cost_per_hour(cost_comparison.book_cost_per_hour_text)
    evaluator.add_custom_node(
        result=bool(book_cost_per_hour_ok),
        id="comparison_book_cost_calculation",
        desc="Calculate price per hour for The Last Wish book",
        parent=comparison_node,
        critical=False
    )

    # Check final recommendation
    has_recommendation = bool(cost_comparison.recommendation_text and cost_comparison.recommendation_text.strip())
    cost_effective_mention = has_any_ci(answer, ['cost-effective', 'cost effective', 'cheaper', 'better value', 'more economical'])

    evaluator.add_custom_node(
        result=bool(has_recommendation and cost_effective_mention),
        id="comparison_final_recommendation",
        desc="Provide final recommendation on which option is more cost-effective",
        parent=comparison_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
