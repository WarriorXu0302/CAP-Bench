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
TASK_ID = "task-5df385"
TASK_DESCRIPTION = "Our team is preparing to transition from Vue to React and would like to purchase a few physical books for the office library corner. Please help me find three React technical books on Goodreads, published after 2022, with a rating of 4.3 or higher. The content must be in-depth; avoid purely introductory books. Once you've identified suitable books, search for them on Google Books. If a preview is available, check the table of contents or main text to determine if the code examples are written using functional components + Hooks, or the older Class components. Finally, please provide me with the titles, authors, exact Goodreads ratings, and confirmation of Hooks-based examples for these three qualifying books."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ReactBook(BaseModel):
    """Details for a single React book"""
    title: Optional[str] = None
    author: Optional[str] = None
    goodreads_rating: Optional[str] = None
    publication_year: Optional[str] = None
    uses_hooks: Optional[str] = None


class BooksCollection(BaseModel):
    """Collection of React books extracted from the answer"""
    books: List[ReactBook] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books_from_answer() -> str:
    return """
Extract all React technical books mentioned in the answer that the user found on Goodreads and checked on Google Books.

For each book, extract:
- title: the book title exactly as stated
- author: the author name(s) exactly as stated
- goodreads_rating: the exact Goodreads rating mentioned (e.g., "4.5", "4.35", "4.3")
- publication_year: the year of publication if mentioned
- uses_hooks: any indication about whether the book uses functional components + Hooks (e.g., "yes", "Hooks-based", "functional components with Hooks", "Class components", "no")

If any field is missing for a book, set it to null. Return all books found in the answer.
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


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+\.\d+|\d+)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def mentions_goodreads(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['goodreads', 'good reads'])


def mentions_google_books(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['google books', 'googlebooks'])


def mentions_filtering_rating(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['4.3', '4.3 or higher', 'rating', 'rated'])


def mentions_year_filter(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['2022', 'published after 2022', 'after 2022', '2023', '2024', '2025'])


def mentions_preview_or_toc(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['preview', 'table of contents', 'toc', 'browsed', 'checked the content', 'looked at'])


def mentions_hooks_check(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['hooks', 'functional component', 'class component', 'code example', 'code style'])


def is_valid_rating(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    rating_val = extract_float(rating_text)
    if rating_val is None:
        return False
    return rating_val >= 4.3


def is_recent_year(year_text: Optional[str]) -> bool:
    if not year_text:
        return False
    year_val = extract_float(year_text)
    if year_val is None:
        return False
    return year_val > 2022


def indicates_hooks_usage(hooks_text: Optional[str]) -> bool:
    if not hooks_text:
        return False
    return has_any_ci(hooks_text, ['yes', 'hooks', 'functional', 'hook-based', 'hooks-based'])


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
    Restrict evaluator.verify to at most one usage (not used here).
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
    books_info = await evaluator.extract(
        prompt=prompt_extract_books_from_answer(),
        template_class=BooksCollection,
        extraction_name="react_books"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Find React books on Goodreads with filtering criteria",
        parent=root,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A1 - Advanced filtering (published after 2022, rating 4.3+)
    goodreads_action_filter = (mentions_goodreads(answer) and
                               mentions_filtering_rating(answer) and
                               mentions_year_filter(answer))
    evaluator.add_custom_node(
        result=bool(goodreads_action_filter),
        id="goodreads_action_advanced_filter",
        desc="[Action Node] goodreads.com:F1:A1 - Use advanced filtering or sorting to find books published after 2022 with rating 4.3+",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P6 - Extract exact ratings
    has_exact_ratings = False
    if books_info and books_info.books:
        ratings_count = sum(1 for book in books_info.books if book.goodreads_rating and extract_float(book.goodreads_rating) is not None)
        has_exact_ratings = ratings_count >= 3

    evaluator.add_custom_node(
        result=bool(has_exact_ratings),
        id="goodreads_perception_exact_ratings",
        desc="[Perception Node] goodreads.com:F2:P6 - Extract exact Goodreads ratings (hover or detail view to get precise scores)",
        parent=goodreads_node,
        critical=False
    )

    # Check for three valid books with correct criteria
    valid_books_count = 0
    if books_info and books_info.books:
        for book in books_info.books:
            has_title = bool(book.title and book.title.strip())
            has_author = bool(book.author and book.author.strip())
            has_valid_rating = is_valid_rating(book.goodreads_rating)
            has_recent_year = is_recent_year(book.publication_year) if book.publication_year else False
            if has_title and has_author and has_valid_rating:
                valid_books_count += 1

    evaluator.add_custom_node(
        result=bool(valid_books_count >= 3),
        id="goodreads_found_three_books",
        desc="Found at least three React books meeting the rating (4.3+) criteria",
        parent=goodreads_node,
        critical=False
    )

    # 3.2 Google Books section
    google_books_node = evaluator.add_sequential(
        id="google_books_section",
        desc="Search books on Google Books and check code style via preview",
        parent=root,
        critical=False
    )

    # [Action Node] books.google.com:F2:A8 - Browse preview pages
    google_books_action_browse = (mentions_google_books(answer) and
                                  mentions_preview_or_toc(answer))
    evaluator.add_custom_node(
        result=bool(google_books_action_browse),
        id="google_books_action_browse_preview",
        desc="[Action Node] books.google.com:F2:A8 - Browse table of contents or main text pages in preview",
        parent=google_books_node,
        critical=False
    )

    # [Perception Node] books.google.com:F2:P11 - Identify code style (Hooks vs Class)
    google_books_perception_hooks = mentions_hooks_check(answer)
    evaluator.add_custom_node(
        result=bool(google_books_perception_hooks),
        id="google_books_perception_code_style",
        desc="[Perception Node] books.google.com:F2:P11 - Identify whether code examples use functional components + Hooks or Class components",
        parent=google_books_node,
        critical=False
    )

    # Check hooks confirmation for books
    hooks_confirmed_count = 0
    if books_info and books_info.books:
        for book in books_info.books:
            if indicates_hooks_usage(book.uses_hooks):
                hooks_confirmed_count += 1

    evaluator.add_custom_node(
        result=bool(hooks_confirmed_count >= 3),
        id="google_books_hooks_confirmed",
        desc="Confirmed Hooks-based examples for at least three books",
        parent=google_books_node,
        critical=False
    )

    # 3.3 Final output validation
    output_node = evaluator.add_parallel(
        id="output_validation",
        desc="Final output contains all required information for three books",
        parent=root,
        critical=False
    )

    # Check completeness of output
    complete_books_count = 0
    if books_info and books_info.books:
        for book in books_info.books:
            has_all_fields = (
                bool(book.title and book.title.strip()) and
                bool(book.author and book.author.strip()) and
                bool(book.goodreads_rating and extract_float(book.goodreads_rating) is not None) and
                bool(book.uses_hooks and book.uses_hooks.strip())
            )
            if has_all_fields:
                complete_books_count += 1

    evaluator.add_custom_node(
        result=bool(complete_books_count >= 3),
        id="output_complete_info",
        desc="Output contains title, author, exact Goodreads rating, and Hooks confirmation for at least three books",
        parent=output_node,
        critical=False
    )

    # Check for in-depth content mention (avoiding purely introductory)
    mentions_indepth = has_any_ci(answer, ['in-depth', 'advanced', 'not introductory', 'beyond basics', 'deep dive'])
    evaluator.add_custom_node(
        result=bool(mentions_indepth),
        id="output_mentions_indepth",
        desc="Mentions consideration of in-depth content (avoiding purely introductory books)",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
