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
TASK_ID = "task-1e71cc"
TASK_DESCRIPTION = 'I want to plan my Python data engineering study for 2026.\nFirst, go to Amazon and search for "Python Data Engineering" paperback books. Use the filtering options to identify 3 candidate books that have a rating of 4 stars or higher, are priced between $30 and $60, and were published in 2024 or 2025.\nNext, go to Goodreads, look up these 3 books, and select the one with the highest "Rating Count" as the final choice.\nFinally, extract core technical keywords mentioned in the title of the ultimately selected book (e.g., PySpark, Kafka, Airflow). Then, go to StackOverflow, search for the tag corresponding to this technology, navigate to that tag\'s page, sort by "Votes," and find the question with the highest number of votes.\n\nPlease output: the titles/prices/publication years of the 3 candidate books, the Goodreads rating count of the finally selected book, the extracted keyword, and the title and link of the highest-voted StackOverflow question.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CandidateBook(BaseModel):
    """A single candidate book from Amazon"""
    title: Optional[str] = None
    price: Optional[str] = None
    publication_year: Optional[str] = None


class CandidateBooks(BaseModel):
    """3 candidate books from Amazon"""
    book1: Optional[CandidateBook] = None
    book2: Optional[CandidateBook] = None
    book3: Optional[CandidateBook] = None


class SelectedBookInfo(BaseModel):
    """Information about the finally selected book"""
    title: Optional[str] = None
    goodreads_rating_count: Optional[str] = None


class ExtractedKeyword(BaseModel):
    """Core technical keyword extracted from the selected book title"""
    keyword: Optional[str] = None


class StackOverflowQuestion(BaseModel):
    """Highest-voted StackOverflow question"""
    title: Optional[str] = None
    link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate_books() -> str:
    return """
Extract the 3 candidate books from Amazon that the user reported in their answer.

For each book, extract:
- title: the book title exactly as stated
- price: the price exactly as stated (include currency symbol if present)
- publication_year: the publication year (should be 2024 or 2025)

Return book1, book2, and book3. If any book or field is missing, set it to null.
"""


def prompt_extract_selected_book() -> str:
    return """
Extract information about the finally selected book (the one with the highest Goodreads rating count):

- title: the title of the selected book
- goodreads_rating_count: the rating count from Goodreads exactly as stated

If any field is missing, set it to null.
"""


def prompt_extract_keyword() -> str:
    return """
Extract the core technical keyword that was extracted from the selected book's title.

This should be a specific technology name like "PySpark", "Kafka", "Airflow", etc.

Return:
- keyword: the extracted technology keyword

If not present, set it to null.
"""


def prompt_extract_stackoverflow_question() -> str:
    return """
Extract the highest-voted StackOverflow question that was found:

- title: the question title
- link: the question URL

If any field is missing, set it to null.
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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'(202[4-5])', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_price_in_range(price_text: Optional[str], min_val: float = 30.0, max_val: float = 60.0) -> bool:
    if not price_text:
        return False
    price_num = extract_float(price_text)
    if price_num is None:
        return False
    return min_val <= price_num <= max_val


def is_year_valid(year_text: Optional[str]) -> bool:
    if not year_text:
        return False
    year = extract_year(year_text)
    return year in [2024, 2025]


def looks_like_rating_count(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text)


def looks_like_stackoverflow_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'stackoverflow.com')


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
        strategy=AggregationStrategy.SEQUENTIAL,
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
    candidate_books = await evaluator.extract(
        prompt=prompt_extract_candidate_books(),
        template_class=CandidateBooks,
        extraction_name="candidate_books"
    )

    selected_book = await evaluator.extract(
        prompt=prompt_extract_selected_book(),
        template_class=SelectedBookInfo,
        extraction_name="selected_book"
    )

    keyword = await evaluator.extract(
        prompt=prompt_extract_keyword(),
        template_class=ExtractedKeyword,
        extraction_name="extracted_keyword"
    )

    stackoverflow_question = await evaluator.extract(
        prompt=prompt_extract_stackoverflow_question(),
        template_class=StackOverflowQuestion,
        extraction_name="stackoverflow_question"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon - Search and filter Python Data Engineering paperback books",
        parent=root,
        critical=False
    )

    # Check if Amazon was accessed
    amazon_mentioned = has_any_ci(answer, ['amazon'])
    evaluator.add_custom_node(
        result=bool(amazon_mentioned),
        id="amazon_accessed",
        desc="Answer mentions accessing Amazon",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A17 - Price range filtering ($30-$60)
    books = [candidate_books.book1, candidate_books.book2, candidate_books.book3]
    prices_valid = all(is_price_in_range(book.price) for book in books if book)
    evaluator.add_custom_node(
        result=bool(prices_valid and any(books)),
        id="amazon_price_range_filter",
        desc="[Action Node] Amazon:F3:A17 - Apply price range filter ($30-$60) using range slider/interval selection",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Rating filter (4 stars or higher)
    rating_mentioned = has_any_ci(answer, ['4 star', '4-star', 'rating'])
    evaluator.add_custom_node(
        result=bool(rating_mentioned),
        id="amazon_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Apply rating filter (4 stars or higher) using multi-select checkbox",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F1:P23 - Publication year check (2024 or 2025)
    years_valid = all(is_year_valid(book.publication_year) for book in books if book)
    evaluator.add_custom_node(
        result=bool(years_valid and any(books)),
        id="amazon_publication_year_check",
        desc="[Perception Node] Amazon:F1:P23 - Check publication years (2024 or 2025) through cross-page comparison/list scanning",
        parent=amazon_node,
        critical=False
    )

    # Check if 3 books were identified
    three_books_found = all(book is not None for book in books)
    evaluator.add_custom_node(
        result=bool(three_books_found),
        id="amazon_three_books_identified",
        desc="Identified exactly 3 candidate books meeting all criteria",
        parent=amazon_node,
        critical=False
    )

    # 3.2 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads - Look up books and compare rating counts",
        parent=root,
        critical=False
    )

    # Check if Goodreads was accessed
    goodreads_mentioned = has_any_ci(answer, ['goodreads'])
    evaluator.add_custom_node(
        result=bool(goodreads_mentioned),
        id="goodreads_accessed",
        desc="Answer mentions accessing Goodreads",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A5 - Navigate to book details
    book_details_accessed = bool(selected_book and selected_book.goodreads_rating_count)
    evaluator.add_custom_node(
        result=bool(book_details_accessed),
        id="goodreads_book_details",
        desc="[Action Node] goodreads.com:F1:A5 - Navigate to book detail pages for the 3 candidate books",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F3:P6 - Extract and compare rating counts
    rating_count_extracted = looks_like_rating_count(selected_book.goodreads_rating_count) if selected_book else False
    highest_mentioned = has_any_ci(answer, ['highest', 'most', 'maximum', 'rating count'])
    evaluator.add_custom_node(
        result=bool(rating_count_extracted and highest_mentioned),
        id="goodreads_rating_count_comparison",
        desc="[Perception Node] goodreads.com:F3:P6 - Extract rating counts and identify the book with the highest count",
        parent=goodreads_node,
        critical=False
    )

    # 3.3 Keyword extraction
    keyword_node = evaluator.add_sequential(
        id="keyword_extraction",
        desc="Extract core technical keyword from selected book title",
        parent=root,
        critical=False
    )

    # [Perception Node] stackoverflow.com:F3:P11 - Content understanding (keyword extraction)
    keyword_extracted = bool(keyword and keyword.keyword and keyword.keyword.strip())
    evaluator.add_custom_node(
        result=bool(keyword_extracted),
        id="keyword_extraction_content_understanding",
        desc="[Perception Node] stackoverflow.com:F3:P11 - Extract core technical keyword from book title (content understanding)",
        parent=keyword_node,
        critical=False
    )

    # 3.4 StackOverflow section
    stackoverflow_node = evaluator.add_sequential(
        id="stackoverflow_section",
        desc="StackOverflow - Find highest-voted question for the technology tag",
        parent=root,
        critical=False
    )

    # Check if StackOverflow was accessed
    stackoverflow_mentioned = has_any_ci(answer, ['stackoverflow', 'stack overflow'])
    evaluator.add_custom_node(
        result=bool(stackoverflow_mentioned),
        id="stackoverflow_accessed",
        desc="Answer mentions accessing StackOverflow",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F3:A19 - Search and filter by tag
    tag_search_mentioned = has_any_ci(answer, ['tag', 'search'])
    evaluator.add_custom_node(
        result=bool(tag_search_mentioned and keyword_extracted),
        id="stackoverflow_tag_search",
        desc="[Action Node] stackoverflow.com:F3:A19 - Search for the technology tag using search/filter (Select)",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F3:A27 - Navigate to tag page
    tag_page_accessed = has_any_ci(answer, ['tag page', 'navigat'])
    evaluator.add_custom_node(
        result=bool(tag_page_accessed),
        id="stackoverflow_tag_page_navigation",
        desc="[Action Node] stackoverflow.com:F3:A27 - Navigate to the tag page by clicking on the tag",
        parent=stackoverflow_node,
        critical=False
    )

    # [Action Node] stackoverflow.com:F3:A3 - Sort by Votes
    sort_by_votes = has_any_ci(answer, ['sort', 'votes', 'highest'])
    evaluator.add_custom_node(
        result=bool(sort_by_votes),
        id="stackoverflow_sort_by_votes",
        desc="[Action Node] stackoverflow.com:F3:A3 - Sort questions by Votes using menu navigation/sort toggle",
        parent=stackoverflow_node,
        critical=False
    )

    # Check if question was found with valid link
    question_found = bool(stackoverflow_question and stackoverflow_question.title and stackoverflow_question.link)
    link_valid = looks_like_stackoverflow_link(stackoverflow_question.link) if stackoverflow_question else False
    evaluator.add_custom_node(
        result=bool(question_found and link_valid),
        id="stackoverflow_highest_voted_question",
        desc="Found the highest-voted question with valid title and StackOverflow link",
        parent=stackoverflow_node,
        critical=False
    )

    # 3.5 Overall completeness check
    all_outputs_present = (
        three_books_found and
        book_details_accessed and
        keyword_extracted and
        question_found
    )
    evaluator.add_custom_node(
        result=bool(all_outputs_present),
        id="all_required_outputs",
        desc="All required outputs present: 3 books, Goodreads rating count, keyword, and StackOverflow question",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
