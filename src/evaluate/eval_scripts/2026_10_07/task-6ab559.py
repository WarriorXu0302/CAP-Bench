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
TASK_ID = "task-6ab559"
TASK_DESCRIPTION = "I am currently preparing for the 'Fast-Paced Reading Marathon' event scheduled for January 2026, and I need your assistance in compiling a booklist.\n\nPlease begin by visiting TheStoryGraph's 'Browse' or recommendation page. Utilize the filters to locate 3 novels that satisfy all the following conditions:\n1.  **Moods**: Must simultaneously include both 'Adventurous' and 'Funny'.\n2.  **Pace**: Must be 'Fast-paced'.\n3.  **Page Length**: Use the page length slider to restrict the range to under 350 pages.\n\nIdentify the three highest-rated books that meet these criteria.\n\nNext, search for each of these three books individually on OpenLibrary.org. Verify if a digital version is available for direct borrowing (indicated by a 'Borrow' button) or online reading (indicated by a 'Read' button).\n\nPlease output the following information for each book: Book Title, Author, StoryGraph Rating, StoryGraph Page Count, specific availability status on OpenLibrary (e.g., Borrow/Read/Not Found/Checked Out), and the direct links to the book's detail pages on both platforms."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    """Information for a single book extracted from the answer"""
    title: Optional[str] = None
    author: Optional[str] = None
    storygraph_rating: Optional[str] = None
    storygraph_page_count: Optional[str] = None
    openlibrary_availability: Optional[str] = None
    storygraph_url: Optional[str] = None
    openlibrary_url: Optional[str] = None


class BooksListInfo(BaseModel):
    """List of three books extracted from the answer"""
    books: List[BookInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books_from_answer() -> str:
    return """
Extract the three books reported in the answer with all their details.

For each book, return:
- title: the book title as stated
- author: the author name as stated
- storygraph_rating: the rating from StoryGraph exactly as written
- storygraph_page_count: the page count from StoryGraph exactly as written
- openlibrary_availability: the availability status on OpenLibrary (e.g., "Borrow", "Read", "Not Found", "Checked Out")
- storygraph_url: the direct link to the book's detail page on StoryGraph
- openlibrary_url: the direct link to the book's detail page on OpenLibrary

If any field is missing for a book, set it to null. Return all three books in the books list.
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


def is_valid_url(url: Optional[str], domain: str) -> bool:
    if not url:
        return False
    return domain.lower() in url.lower() and url.startswith('http')


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_page_count(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num < 350


def is_valid_availability_status(status: Optional[str]) -> bool:
    if not status:
        return False
    valid_statuses = ['borrow', 'read', 'not found', 'checked out', 'unavailable', 'available']
    return any(ci_contains(status, s) for s in valid_statuses)


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
    books_info = await evaluator.extract(
        prompt=prompt_extract_books_from_answer(),
        template_class=BooksListInfo,
        extraction_name="books_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 TheStoryGraph section
    storygraph_node = evaluator.add_sequential(
        id="storygraph_section",
        desc="TheStoryGraph filtering and book discovery",
        parent=root,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A1 - Multi-select mood filtering (Adventurous AND Funny)
    mentions_adventurous = has_any_ci(answer, ['adventurous'])
    mentions_funny = has_any_ci(answer, ['funny'])
    mentions_moods = has_any_ci(answer, ['mood', 'moods'])
    mood_filtering_ok = mentions_adventurous and mentions_funny and mentions_moods

    evaluator.add_custom_node(
        result=bool(mood_filtering_ok),
        id="storygraph_action_mood_filter",
        desc="[Action Node] thestorygraph.com:F2:A1 - Apply multi-select mood filter for both 'Adventurous' and 'Funny'",
        parent=storygraph_node,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A3 - Single-select pace filtering (Fast-paced)
    mentions_fast_paced = has_any_ci(answer, ['fast-paced', 'fast paced'])
    mentions_pace = has_any_ci(answer, ['pace'])
    pace_filtering_ok = mentions_fast_paced or (mentions_pace and has_any_ci(answer, ['fast']))

    evaluator.add_custom_node(
        result=bool(pace_filtering_ok),
        id="storygraph_action_pace_filter",
        desc="[Action Node] thestorygraph.com:F2:A3 - Apply single-select pace filter for 'Fast-paced'",
        parent=storygraph_node,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A4 - Slider range selection for page length (<350)
    mentions_page_slider = has_any_ci(answer, ['page length', 'pages', 'slider', '350'])
    all_books_under_350 = True
    if books_info and books_info.books:
        for book in books_info.books:
            if book.storygraph_page_count:
                if not looks_like_page_count(book.storygraph_page_count):
                    all_books_under_350 = False
                    break

    slider_filtering_ok = mentions_page_slider and all_books_under_350

    evaluator.add_custom_node(
        result=bool(slider_filtering_ok),
        id="storygraph_action_page_slider",
        desc="[Action Node] thestorygraph.com:F2:A4 - Use page length slider to restrict to under 350 pages",
        parent=storygraph_node,
        critical=False
    )

    # [Perception Node] thestorygraph.com:F3:P6 - Identify highest-rated books
    has_three_books = books_info and len(books_info.books) == 3
    all_have_ratings = True
    all_ratings_reasonable = True
    if has_three_books:
        for book in books_info.books:
            if not book.storygraph_rating:
                all_have_ratings = False
            elif not looks_like_rating(book.storygraph_rating):
                all_ratings_reasonable = False

    rating_identification_ok = has_three_books and all_have_ratings and all_ratings_reasonable

    evaluator.add_custom_node(
        result=bool(rating_identification_ok),
        id="storygraph_perception_highest_rated",
        desc="[Perception Node] thestorygraph.com:F3:P6 - Identify the three highest-rated books that meet criteria",
        parent=storygraph_node,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A8 - Click on book cards to get details
    all_have_storygraph_urls = True
    all_storygraph_urls_valid = True
    if has_three_books:
        for book in books_info.books:
            if not book.storygraph_url:
                all_have_storygraph_urls = False
            elif not is_valid_url(book.storygraph_url, 'thestorygraph.com'):
                all_storygraph_urls_valid = False

    card_click_ok = all_have_storygraph_urls and all_storygraph_urls_valid

    evaluator.add_custom_node(
        result=bool(card_click_ok),
        id="storygraph_action_card_click",
        desc="[Action Node] thestorygraph.com:F2:A8 - Click on book cards to access detail pages",
        parent=storygraph_node,
        critical=False
    )

    # [Perception Node] thestorygraph.com:F3:P1 - Verify mood tags on book pages
    mentions_verification = has_any_ci(answer, ['verify', 'confirm', 'check'])
    mood_tags_verified = mentions_adventurous and mentions_funny and mentions_verification

    evaluator.add_custom_node(
        result=bool(mood_tags_verified),
        id="storygraph_perception_mood_tags",
        desc="[Perception Node] thestorygraph.com:F3:P1 - Verify mood tags (Adventurous and Funny) on book detail pages",
        parent=storygraph_node,
        critical=False
    )

    # 3.2 OpenLibrary section
    openlibrary_node = evaluator.add_sequential(
        id="openlibrary_section",
        desc="OpenLibrary search and availability verification",
        parent=root,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A1 - Search type selection and search execution
    mentions_openlibrary = has_any_ci(answer, ['openlibrary', 'open library'])
    mentions_search = has_any_ci(answer, ['search'])
    all_have_openlibrary_urls = True
    all_openlibrary_urls_valid = True
    if has_three_books:
        for book in books_info.books:
            if not book.openlibrary_url:
                all_have_openlibrary_urls = False
            elif not is_valid_url(book.openlibrary_url, 'openlibrary.org'):
                all_openlibrary_urls_valid = False

    search_execution_ok = mentions_openlibrary and all_have_openlibrary_urls and all_openlibrary_urls_valid

    evaluator.add_custom_node(
        result=bool(search_execution_ok),
        id="openlibrary_action_search",
        desc="[Action Node] openlibrary.org:F1:A1 - Search for each book on OpenLibrary with appropriate search type",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A6 - Click on search results to access detail pages
    detail_page_access_ok = all_have_openlibrary_urls and all_openlibrary_urls_valid

    evaluator.add_custom_node(
        result=bool(detail_page_access_ok),
        id="openlibrary_action_result_click",
        desc="[Action Node] openlibrary.org:F1:A6 - Click on search results to access book detail pages",
        parent=openlibrary_node,
        critical=False
    )

    # [Perception Node] openlibrary.org:F1:P1 - Identify availability status (Borrow/Read buttons)
    all_have_availability = True
    all_availability_valid = True
    if has_three_books:
        for book in books_info.books:
            if not book.openlibrary_availability:
                all_have_availability = False
            elif not is_valid_availability_status(book.openlibrary_availability):
                all_availability_valid = False

    availability_detection_ok = all_have_availability and all_availability_valid

    evaluator.add_custom_node(
        result=bool(availability_detection_ok),
        id="openlibrary_perception_availability",
        desc="[Perception Node] openlibrary.org:F1:P1 - Detect and report availability status (Borrow/Read/Not Found/Checked Out)",
        parent=openlibrary_node,
        critical=False
    )

    # 3.3 Additional data completeness checks (non-prefixed)
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Data completeness and quality checks",
        parent=root,
        critical=False
    )

    # Check that all books have titles
    all_have_titles = True
    if has_three_books:
        for book in books_info.books:
            if not book.title or not book.title.strip():
                all_have_titles = False

    evaluator.add_custom_node(
        result=bool(all_have_titles),
        id="completeness_all_titles",
        desc="All three books have titles provided",
        parent=completeness_node,
        critical=False
    )

    # Check that all books have authors
    all_have_authors = True
    if has_three_books:
        for book in books_info.books:
            if not book.author or not book.author.strip():
                all_have_authors = False

    evaluator.add_custom_node(
        result=bool(all_have_authors),
        id="completeness_all_authors",
        desc="All three books have authors provided",
        parent=completeness_node,
        critical=False
    )

    # Check that all books have page counts
    all_have_page_counts = True
    if has_three_books:
        for book in books_info.books:
            if not book.storygraph_page_count:
                all_have_page_counts = False

    evaluator.add_custom_node(
        result=bool(all_have_page_counts),
        id="completeness_all_page_counts",
        desc="All three books have StoryGraph page counts provided",
        parent=completeness_node,
        critical=False
    )

    # Check that exactly three books are provided
    evaluator.add_custom_node(
        result=bool(has_three_books),
        id="completeness_three_books",
        desc="Exactly three books are provided as requested",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
