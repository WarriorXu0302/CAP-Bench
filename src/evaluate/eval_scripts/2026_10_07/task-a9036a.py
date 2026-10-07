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
TASK_ID = "task-a9036a"
TASK_DESCRIPTION = 'As a book, film, and music enthusiast, I am currently completing my collection of original source materials for Oscar-winning films. Please help me compile information on the winning works for the Best Adapted Screenplay award at the 95th (2023), 96th (2024), and 97th (2025) Academy Awards.\n\nFirst, on IMDb, identify the winning films for these three years, along with their original book titles and authors.\n\nNext, search for the English original editions of these three books on Amazon. Filter for hardcover versions with a rating of 4.5 stars or higher, and record their prices.\n\nFinally, for collection value, for the original book of the 2025 winning film, check eBay for "First Edition" copies available for sale. Filter by "Buy It Now" mode to find the lowest-priced one.\n\nOutput: Year, Film Title, Original Book Title, Author, Amazon Hardcover Price, Amazon Rating, eBay Lowest First Edition Price (for 2025 winner only), and direct links to each product detail page.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OscarWinner(BaseModel):
    """Single Oscar Best Adapted Screenplay winner information"""
    year: Optional[str] = None
    film_title: Optional[str] = None
    book_title: Optional[str] = None
    author: Optional[str] = None
    amazon_price: Optional[str] = None
    amazon_rating: Optional[str] = None
    amazon_link: Optional[str] = None
    ebay_first_edition_price: Optional[str] = None
    ebay_link: Optional[str] = None


class AllWinners(BaseModel):
    """All three years of Oscar winners"""
    winners: List[OscarWinner] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_winners() -> str:
    return """
Extract all Oscar Best Adapted Screenplay winner information for the 95th (2023), 96th (2024), and 97th (2025) Academy Awards from the answer.

For each year, extract:
- year: the year (2023, 2024, or 2025)
- film_title: the winning film title
- book_title: the original book title
- author: the book author
- amazon_price: the Amazon hardcover price exactly as stated
- amazon_rating: the Amazon rating exactly as stated
- amazon_link: the Amazon product detail page link
- ebay_first_edition_price: the eBay first edition price (only for 2025 winner)
- ebay_link: the eBay product detail page link (only for 2025 winner)

Return a list of all winners found. If any field is missing, set it to null.
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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_year(text: Optional[str], expected_years: List[str]) -> bool:
    if not text:
        return False
    return any(year in text for year in expected_years)


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['$', 'usd', 'dollar']) or re.search(r'\d+\.\d{2}', text))


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http' in text.lower() or 'www' in text.lower() or '/' in text)


def count_years_covered(winners: List[OscarWinner]) -> int:
    years = set()
    for w in winners:
        if w.year and looks_like_year(w.year, ['2023', '2024', '2025']):
            if '2023' in w.year:
                years.add('2023')
            if '2024' in w.year:
                years.add('2024')
            if '2025' in w.year:
                years.add('2025')
    return len(years)


def get_2025_winner(winners: List[OscarWinner]) -> Optional[OscarWinner]:
    for w in winners:
        if w.year and '2025' in w.year:
            return w
    return None


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
    Restrict evaluator.verify to at most one usage.
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
    all_winners = await evaluator.extract(
        prompt=prompt_extract_all_winners(),
        template_class=AllWinners,
        extraction_name="all_oscar_winners"
    )

    winners = all_winners.winners if all_winners and all_winners.winners else []

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb Oscar Best Adapted Screenplay winners for 2023, 2024, 2025",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F2:A4 - Navigate to Awards & Events menu
    imdb_navigation_ok = has_any_ci(answer, ['imdb']) and (has_any_ci(answer, ['awards', 'oscar', 'academy awards']))
    evaluator.add_custom_node(
        result=bool(imdb_navigation_ok),
        id="imdb_action_navigate_awards",
        desc="[Action Node] imdb.com:F2:A4 - Navigate to IMDb Awards & Events section to find Oscar winners",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F2:A17 - Browse historical years (95th, 96th, 97th)
    years_covered = count_years_covered(winners)
    imdb_years_ok = years_covered >= 2  # Lenient: at least 2 out of 3 years
    evaluator.add_custom_node(
        result=bool(imdb_years_ok),
        id="imdb_action_year_browsing",
        desc="[Action Node] imdb.com:F2:A17 - Browse or filter Oscar winners for the 95th (2023), 96th (2024), and 97th (2025) ceremonies",
        parent=imdb_node,
        critical=False
    )

    # Check extraction of film, book, and author information
    films_extracted = sum(1 for w in winners if w.film_title and w.film_title.strip())
    books_extracted = sum(1 for w in winners if w.book_title and w.book_title.strip())
    authors_extracted = sum(1 for w in winners if w.author and w.author.strip())

    imdb_extraction_ok = films_extracted >= 2 and books_extracted >= 2 and authors_extracted >= 2
    evaluator.add_custom_node(
        result=bool(imdb_extraction_ok),
        id="imdb_extraction_complete",
        desc="Extracted film titles, original book titles, and authors from IMDb for multiple years",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon search for hardcover books with rating 4.5+",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A5 - Category search in Books
    amazon_search_ok = has_any_ci(answer, ['amazon']) and (has_any_ci(answer, ['book', 'books']) or books_extracted > 0)
    evaluator.add_custom_node(
        result=bool(amazon_search_ok),
        id="amazon_action_category_search",
        desc="[Action Node] Amazon:F1:A5 - Search for the books on Amazon, preferably in the Books category",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Filter by rating 4.5+
    ratings_ok = sum(1 for w in winners if w.amazon_rating and looks_like_rating(w.amazon_rating) and extract_float(w.amazon_rating) >= 4.5)
    amazon_rating_filter_ok = ratings_ok >= 2 or has_any_ci(answer, ['4.5', 'rating'])
    evaluator.add_custom_node(
        result=bool(amazon_rating_filter_ok),
        id="amazon_action_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Filter books by rating 4.5 stars or higher",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P2 - Identify hardcover format and price
    hardcover_ok = has_any_ci(answer, ['hardcover']) or sum(1 for w in winners if w.amazon_link and ci_contains(w.amazon_link, 'hardcover')) > 0
    prices_ok = sum(1 for w in winners if w.amazon_price and looks_like_price(w.amazon_price))
    amazon_format_perception_ok = hardcover_ok and prices_ok >= 2
    evaluator.add_custom_node(
        result=bool(amazon_format_perception_ok),
        id="amazon_perception_format_price",
        desc="[Perception Node] Amazon:F3:P2 - Identify hardcover format and extract prices for the books",
        parent=amazon_node,
        critical=False
    )

    # Check Amazon links
    amazon_links_ok = sum(1 for w in winners if w.amazon_link and looks_like_url(w.amazon_link, 'amazon'))
    evaluator.add_custom_node(
        result=bool(amazon_links_ok >= 2),
        id="amazon_links_provided",
        desc="Provided direct links to Amazon product detail pages",
        parent=amazon_node,
        critical=False
    )

    # 3.3 eBay section (for 2025 winner only)
    ebay_node = evaluator.add_sequential(
        id="ebay_section",
        desc="eBay search for First Edition of 2025 winner's book (Buy It Now, lowest price)",
        parent=root,
        critical=False
    )

    winner_2025 = get_2025_winner(winners)

    # [Perception Node] eBay:F3:P5 - Identify First Edition
    ebay_first_edition_ok = (has_any_ci(answer, ['first edition']) or
                             (winner_2025 and winner_2025.ebay_link and ci_contains(winner_2025.ebay_link, 'first')))
    evaluator.add_custom_node(
        result=bool(ebay_first_edition_ok),
        id="ebay_perception_first_edition",
        desc="[Perception Node] eBay:F3:P5 - Identify First Edition copies in title or description",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Filter by Buy It Now
    ebay_buy_it_now_ok = has_any_ci(answer, ['buy it now', 'buy-it-now'])
    evaluator.add_custom_node(
        result=bool(ebay_buy_it_now_ok),
        id="ebay_action_buy_it_now_filter",
        desc="[Action Node] eBay:F3:A11 - Filter by Buy It Now purchase mode (not auction)",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A18 - Sort by lowest price
    ebay_price_present = winner_2025 and winner_2025.ebay_first_edition_price and looks_like_price(winner_2025.ebay_first_edition_price)
    ebay_sort_ok = has_any_ci(answer, ['lowest', 'cheapest', 'sort', 'price']) or ebay_price_present
    evaluator.add_custom_node(
        result=bool(ebay_sort_ok),
        id="ebay_action_sort_price",
        desc="[Action Node] eBay:F3:A18 - Sort by price (lowest first) to find the cheapest First Edition",
        parent=ebay_node,
        critical=False
    )

    # Check eBay link for 2025 winner
    ebay_link_ok = winner_2025 and winner_2025.ebay_link and looks_like_url(winner_2025.ebay_link, 'ebay')
    evaluator.add_custom_node(
        result=bool(ebay_link_ok),
        id="ebay_link_provided",
        desc="Provided direct link to eBay product detail page for 2025 winner",
        parent=ebay_node,
        critical=False
    )

    # Check eBay price extraction for 2025
    evaluator.add_custom_node(
        result=bool(ebay_price_present),
        id="ebay_price_extracted",
        desc="Extracted eBay First Edition price for 2025 winner",
        parent=ebay_node,
        critical=False
    )

    # 3.4 Overall completeness check
    all_years_ok = years_covered == 3
    evaluator.add_custom_node(
        result=bool(all_years_ok),
        id="all_three_years_covered",
        desc="All three years (2023, 2024, 2025) are covered in the output",
        parent=root,
        critical=False
    )

    all_amazon_info_ok = (amazon_links_ok >= 3 and
                          sum(1 for w in winners if w.amazon_price and w.amazon_rating) >= 3)
    evaluator.add_custom_node(
        result=bool(all_amazon_info_ok),
        id="complete_amazon_info",
        desc="Complete Amazon information (price, rating, link) for all three books",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
