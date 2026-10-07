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
TASK_ID = "task-2fd0d4"
TASK_DESCRIPTION = 'As a film noir enthusiast, I want to collect three classic films released between 1940 and 1959 that are highly acclaimed.\n\nFirst, on Metacritic, filter for film noir titles released between 1940 and 1959 with a Metascore of 85 or higher. Sort them in descending order by score and select the top three.\n\nNext, visit the official Criterion website to confirm if Blu-ray versions (non-4K, non-DVD) of these three films are available for purchase. Check the technical specifications for a "4K digital restoration" note and record the official Criterion website price.\n\nFinally, search on eBay for used Blu-ray discs of these three films. Requirements: condition "Very Good" or better, "Buy It Now" format only, and the total price (including shipping) must be lower than the official Criterion website price.\n\nOutput the following for each film: Title, Metascore, Criterion official Blu-ray price, whether it has a "4K digital restoration" note, and the lowest eligible eBay item price and link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmInfo(BaseModel):
    """Information for a single film extracted from the answer"""
    title: Optional[str] = None
    metascore: Optional[int] = None
    criterion_bluray_price: Optional[str] = None
    has_4k_restoration: Optional[bool] = None
    ebay_price: Optional[str] = None
    ebay_link: Optional[str] = None


class FilmsCollection(BaseModel):
    """Collection of three films extracted from the answer"""
    films: List[FilmInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_films_from_answer() -> str:
    return """
Extract information about the three film noir movies from the answer. For each film, extract:

- title: the film's title
- metascore: the Metascore value as an integer
- criterion_bluray_price: the official Criterion Blu-ray price (include currency symbol if present)
- has_4k_restoration: boolean indicating if the film has "4K digital restoration" mentioned
- ebay_price: the lowest eligible eBay price (include currency and shipping if mentioned)
- ebay_link: the eBay item URL/link

Return a list of up to 3 films. If any field is missing for a film, set it to null.
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


def extract_int(text: Optional[Any]) -> Optional[int]:
    if text is None:
        return None
    if isinstance(text, int):
        return text
    if isinstance(text, str):
        m = re.search(r'\d+', text)
        if m:
            try:
                return int(m.group())
            except Exception:
                return None
    return None


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


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain a number and optionally currency symbol
    has_number = bool(re.search(r'\d', text))
    has_currency = bool(re.search(r'[$£€¥]', text))
    return has_number


def scores_descending_or_equal(scores: List[Optional[int]]) -> bool:
    """Check if scores are in descending or equal order"""
    valid_scores = [s for s in scores if s is not None]
    if len(valid_scores) < 2:
        return True
    for i in range(len(valid_scores) - 1):
        if valid_scores[i] < valid_scores[i + 1]:
            return False
    return True


def all_scores_85_or_higher(scores: List[Optional[int]]) -> bool:
    """Check if all valid scores are 85 or higher"""
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return False
    return all(s >= 85 for s in valid_scores)


def ebay_price_lower_than_criterion(ebay_price: Optional[str], criterion_price: Optional[str]) -> bool:
    """Check if eBay price is lower than Criterion price"""
    if not ebay_price or not criterion_price:
        return False
    ebay_num = extract_float(ebay_price)
    criterion_num = extract_float(criterion_price)
    if ebay_num is None or criterion_num is None:
        return False
    return ebay_num < criterion_num


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
    films_data = await evaluator.extract(
        prompt=prompt_extract_films_from_answer(),
        template_class=FilmsCollection,
        extraction_name="films_collection"
    )

    films = films_data.films if films_data and films_data.films else []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Metacritic section
    metacritic_node = evaluator.add_sequential(
        id="metacritic_section",
        desc="Metacritic film noir filtering and selection (1940-1959, Metascore 85+, top 3)",
        parent=root,
        critical=False
    )

    # [Action Node] metacritic.com:F3:A4 - Filter by genre (Film Noir)
    mentions_film_noir = has_any_ci(answer, ['film noir', 'noir'])
    mentions_metacritic = has_any_ci(answer, ['metacritic'])
    evaluator.add_custom_node(
        result=bool(mentions_film_noir and mentions_metacritic),
        id="metacritic_genre_filter",
        desc="[Action Node] metacritic.com:F3:A4 - Filter for Film Noir genre on Metacritic",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.com:F3:A9 - Filter by year range (1940-1959)
    mentions_year_range = (has_any_ci(answer, ['1940', '1959']) or
                          has_any_ci(answer, ['1940-1959', '1940 to 1959', '1940 and 1959']))
    evaluator.add_custom_node(
        result=bool(mentions_year_range),
        id="metacritic_year_filter",
        desc="[Action Node] metacritic.com:F3:A9 - Filter for films released between 1940 and 1959",
        parent=metacritic_node,
        critical=False
    )

    # [Action Node] metacritic.com:F3:A5 - Sort by score descending
    metascores = [extract_int(f.metascore) for f in films]
    scores_sorted = scores_descending_or_equal(metascores)
    evaluator.add_custom_node(
        result=bool(scores_sorted and len(films) > 0),
        id="metacritic_sort_descending",
        desc="[Action Node] metacritic.com:F3:A5 - Sort films in descending order by Metascore",
        parent=metacritic_node,
        critical=False
    )

    # Check Metascore 85+ requirement (implicit in filtering)
    scores_85_plus = all_scores_85_or_higher(metascores)
    evaluator.add_custom_node(
        result=bool(scores_85_plus),
        id="metacritic_score_threshold",
        desc="Films have Metascore of 85 or higher",
        parent=metacritic_node,
        critical=False
    )

    # Check exactly 3 films returned
    three_films = len(films) == 3
    evaluator.add_custom_node(
        result=bool(three_films),
        id="metacritic_top_three",
        desc="Selected exactly top three films",
        parent=metacritic_node,
        critical=False
    )

    # 3.2 Criterion section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion Collection Blu-ray verification and pricing",
        parent=root,
        critical=False
    )

    mentions_criterion = has_any_ci(answer, ['criterion'])
    has_criterion_prices = any(f.criterion_bluray_price for f in films)

    # [Action Node] criterion.com:F2:A13 - Tab switching for Blu-ray format
    mentions_bluray = has_any_ci(answer, ['blu-ray', 'bluray', 'blu ray'])
    mentions_format_distinction = (has_any_ci(answer, ['non-4k', 'non-dvd', 'not 4k', 'not dvd']) or
                                  (mentions_bluray and not has_any_ci(answer, ['4k uhd', '4k ultra'])))
    evaluator.add_custom_node(
        result=bool(mentions_criterion and mentions_bluray and mentions_format_distinction),
        id="criterion_format_tab",
        desc="[Action Node] criterion.com:F2:A13 - Switch to Blu-ray format tab (non-4K, non-DVD) to check prices",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P4 - Read technical specifications for 4K restoration
    has_4k_info = any(f.has_4k_restoration is not None for f in films)
    mentions_technical_specs = has_any_ci(answer, ['technical', 'specification', 'specs', '4k digital restoration', '4k restoration'])
    evaluator.add_custom_node(
        result=bool(has_4k_info and mentions_technical_specs),
        id="criterion_4k_restoration_check",
        desc='[Perception Node] criterion.com:F2:P4 - Extract "4K digital restoration" information from technical specifications',
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P5 - Check availability status and extract price
    criterion_prices_valid = all(looks_like_price(f.criterion_bluray_price) for f in films if f.criterion_bluray_price)
    evaluator.add_custom_node(
        result=bool(has_criterion_prices and criterion_prices_valid),
        id="criterion_price_extraction",
        desc="[Perception Node] criterion.com:F2:P5 - Verify Blu-ray availability and extract official prices",
        parent=criterion_node,
        critical=False
    )

    # 3.3 eBay section
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="eBay used Blu-ray search with condition and format filters",
        parent=root,
        critical=False
    )

    mentions_ebay = has_any_ci(answer, ['ebay'])
    has_ebay_links = any(f.ebay_link for f in films)

    # [Action Node] eBay:F3:A11 - Multi-select condition filter (Very Good or better)
    mentions_condition = has_any_ci(answer, ['very good', 'condition', 'like new', 'brand new'])
    evaluator.add_custom_node(
        result=bool(mentions_ebay and mentions_condition),
        id="ebay_condition_filter",
        desc='[Action Node] eBay:F3:A11 - Filter by condition "Very Good" or better using multi-select',
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A13 - Filter by Buy It Now format
    mentions_buy_it_now = has_any_ci(answer, ['buy it now', 'buy-it-now'])
    evaluator.add_custom_node(
        result=bool(mentions_ebay and mentions_buy_it_now),
        id="ebay_buy_it_now_filter",
        desc='[Action Node] eBay:F3:A13 - Filter for "Buy It Now" format only',
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F1:P16 - Extract and compare prices
    price_comparisons_valid = True
    for film in films:
        if film.ebay_price and film.criterion_bluray_price:
            if not ebay_price_lower_than_criterion(film.ebay_price, film.criterion_bluray_price):
                price_comparisons_valid = False
                break

    has_ebay_prices = any(f.ebay_price for f in films)
    evaluator.add_custom_node(
        result=bool(has_ebay_prices and price_comparisons_valid),
        id="ebay_price_comparison",
        desc="[Perception Node] eBay:F1:P16 - Extract eBay prices and verify they are lower than Criterion prices",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F1:P1 - Verify Criterion Collection edition
    mentions_criterion_collection = has_any_ci(answer, ['criterion collection', 'criterion edition'])
    evaluator.add_custom_node(
        result=bool(mentions_ebay and mentions_criterion_collection and has_ebay_links),
        id="ebay_criterion_edition_verify",
        desc="[Perception Node] eBay:F1:P1 - Verify items are Criterion Collection editions via image/title recognition",
        parent=ebay_node,
        critical=False
    )

    # Additional checks for completeness
    valid_ebay_links = all(is_valid_url(f.ebay_link) for f in films if f.ebay_link)
    evaluator.add_custom_node(
        result=bool(has_ebay_links and valid_ebay_links),
        id="ebay_links_provided",
        desc="eBay item links are provided and valid URLs",
        parent=ebay_node,
        critical=False
    )

    # 3.4 Output completeness check
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Check all required information is present in output",
        parent=root,
        critical=False
    )

    all_titles = all(f.title for f in films)
    all_metascores = all(f.metascore is not None for f in films)
    all_criterion_prices = all(f.criterion_bluray_price for f in films)
    all_4k_info = all(f.has_4k_restoration is not None for f in films)
    all_ebay_prices = all(f.ebay_price for f in films)
    all_ebay_links = all(f.ebay_link for f in films)

    evaluator.add_custom_node(
        result=bool(all_titles and all_metascores and all_criterion_prices and
                   all_4k_info and all_ebay_prices and all_ebay_links and three_films),
        id="complete_output",
        desc="All required fields present for all three films",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
