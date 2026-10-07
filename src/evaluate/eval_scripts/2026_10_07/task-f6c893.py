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
TASK_ID = "task-f6c893"
TASK_DESCRIPTION = 'I’m a parent of two children preparing a family course on **Ancient Greek Mythology**. Please help me create an interactive reading list that combines **books + physical artifacts**.\n\nFirst, go to **Goodreads** and browse the “Greek Mythology” topic. Under the **“Childrens”** category, find **3** popular children’s books with ratings above 4.0. To ensure data accuracy, open each book’s detail page and record the **exact number of ratings** for each title (e.g., 12,345 rather than 12k).\n\nNext, go to **OpenLibrary** and search for these three books. Use the sidebar filters to show only **English editions (Language: English)**, and confirm their current borrowing status (prioritize editions showing the blue **“Borrow”** button; if you see **“Join waitlist,” “Checked out,”** or borrowing requires login, replace that book and note the reason for replacement in the results).\n\nFinally, for one major mythological figure mentioned in each book title or description (such as Zeus, Athena, Poseidon, etc.), go to the **Getty Museum** (**getty.edu**) and find one corresponding piece of ancient art (e.g., sculpture, pottery).\n\nOutput required for each book:  \n- Title  \n- Author  \n- Exact Goodreads rating count  \n- OpenLibrary borrowing status  \n- Getty collection item name  \n- Detail-page links from all three platforms'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookEntry(BaseModel):
    """Single book entry with all required information"""
    title: Optional[str] = None
    author: Optional[str] = None
    goodreads_rating_count: Optional[str] = None
    goodreads_link: Optional[str] = None
    openlibrary_status: Optional[str] = None
    openlibrary_link: Optional[str] = None
    getty_item_name: Optional[str] = None
    getty_link: Optional[str] = None
    mythological_figure: Optional[str] = None


class ExtractedBooks(BaseModel):
    """All three books extracted from the answer"""
    book1: Optional[BookEntry] = None
    book2: Optional[BookEntry] = None
    book3: Optional[BookEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_books_from_answer() -> str:
    return """
Extract the three children's books about Greek Mythology from the answer.

For each book, extract:
- title: the book title
- author: the author name
- goodreads_rating_count: the exact number of ratings (e.g., "12,345" not "12k")
- goodreads_link: the Goodreads detail page URL
- openlibrary_status: the borrowing status (e.g., "Borrow", "Join waitlist", etc.)
- openlibrary_link: the OpenLibrary detail page URL
- getty_item_name: the name of the Getty Museum artifact
- getty_link: the Getty Museum item URL
- mythological_figure: the mythological figure mentioned (e.g., Zeus, Athena)

Return three book entries as book1, book2, and book3. If fewer than 3 books are present, set missing entries to null.
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


def looks_like_exact_count(text: Optional[str]) -> bool:
    """Check if rating count looks like exact number (not abbreviated like '12k')"""
    if not text:
        return False
    # Should contain digits
    if not re.search(r'\d', text):
        return False
    # Should NOT contain abbreviations like 'k', 'K', 'm', 'M'
    if re.search(r'\d+\s*[kKmM]\b', text):
        return False
    # Should look like a number (possibly with commas)
    if re.search(r'\d{1,3}(,\d{3})*', text):
        return True
    # Or just a plain number
    if re.search(r'^\d+$', text.replace(',', '').strip()):
        return True
    return False


def is_goodreads_detail_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'goodreads.com' in url.lower() and '/book/show/' in url.lower()


def is_openlibrary_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'openlibrary.org' in url.lower()


def is_getty_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return 'getty.edu' in url.lower()


def mentions_borrow_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['borrow', 'waitlist', 'checked out', 'available'])


def mentions_english_filter(answer: str) -> bool:
    return has_any_ci(answer, ['english', 'language: english', 'language filter'])


def mentions_childrens_category(answer: str) -> bool:
    return has_any_ci(answer, ['childrens', 'children\'s', 'children category'])


def greek_myth_figures() -> List[str]:
    return ['zeus', 'athena', 'poseidon', 'hera', 'apollo', 'artemis', 'ares',
            'aphrodite', 'hermes', 'hephaestus', 'demeter', 'hades', 'persephone',
            'dionysus', 'hestia', 'hercules', 'perseus', 'theseus', 'odysseus',
            'achilles', 'pandora', 'prometheus', 'medusa']


def mentions_mythological_figure(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(ci_contains(text, fig) for fig in greek_myth_figures())


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
        template_class=ExtractedBooks,
        extraction_name="books_information"
    )

    # Collect all books in a list for iteration
    all_books = [books_info.book1, books_info.book2, books_info.book3]

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads children's books on Greek Mythology with ratings > 4.0",
        parent=root,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A1 - Navigate to Childrens category
    childrens_category_ok = mentions_childrens_category(answer)
    evaluator.add_custom_node(
        result=bool(childrens_category_ok),
        id="goodreads_childrens_navigation",
        desc="[Action Node] goodreads.com:F1:A1 - Navigate to or filter by Childrens category under Greek Mythology topic",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F1:A5 - Click into book detail pages
    detail_pages_accessed = sum(1 for book in all_books if book and is_goodreads_detail_url(book.goodreads_link))
    evaluator.add_custom_node(
        result=bool(detail_pages_accessed >= 3),
        id="goodreads_detail_pages",
        desc="[Action Node] goodreads.com:F1:A5 - Open book detail pages to get exact rating counts (all 3 books)",
        parent=goodreads_node,
        critical=False
    )

    # [Perception Node] goodreads.com:F2:P6 - Extract exact rating counts (not abbreviated)
    exact_counts = sum(1 for book in all_books if book and looks_like_exact_count(book.goodreads_rating_count))
    evaluator.add_custom_node(
        result=bool(exact_counts >= 3),
        id="goodreads_exact_ratings",
        desc="[Perception Node] goodreads.com:F2:P6 - Extract exact rating counts (e.g., 12,345 not 12k) via hover or detail view for all 3 books",
        parent=goodreads_node,
        critical=False
    )

    # Additional check: mentions rating above 4.0
    mentions_rating_threshold = has_any_ci(answer, ['rating', '4.0', 'above 4'])
    evaluator.add_custom_node(
        result=bool(mentions_rating_threshold),
        id="goodreads_rating_threshold",
        desc="Mentions filtering or checking for ratings above 4.0",
        parent=goodreads_node,
        critical=False
    )

    # 3.2 OpenLibrary section
    openlibrary_node = evaluator.add_sequential(
        id="openlibrary_section",
        desc="OpenLibrary English editions with borrowing status verification",
        parent=root,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A1 - Search for the three books
    openlibrary_books_found = sum(1 for book in all_books if book and is_openlibrary_url(book.openlibrary_link))
    evaluator.add_custom_node(
        result=bool(openlibrary_books_found >= 3),
        id="openlibrary_search",
        desc="[Action Node] openlibrary.org:F1:A1 - Search for all three books on OpenLibrary",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A5 - Use sidebar filter for English language
    english_filter_ok = mentions_english_filter(answer)
    evaluator.add_custom_node(
        result=bool(english_filter_ok),
        id="openlibrary_english_filter",
        desc="[Action Node] openlibrary.org:F1:A5 - Use sidebar filters to show only English editions (Language: English)",
        parent=openlibrary_node,
        critical=False
    )

    # [Action Node] openlibrary.org:F1:A6 - Click into detail pages to confirm status
    openlibrary_detail_ok = sum(1 for book in all_books if book and is_openlibrary_url(book.openlibrary_link))
    evaluator.add_custom_node(
        result=bool(openlibrary_detail_ok >= 3),
        id="openlibrary_detail_pages",
        desc="[Action Node] openlibrary.org:F1:A6 - Open detail pages to confirm borrowing status for all 3 books",
        parent=openlibrary_node,
        critical=False
    )

    # [Perception Node] openlibrary.org:F1:P1 - Verify Borrow button status
    borrow_status_recorded = sum(1 for book in all_books if book and mentions_borrow_status(book.openlibrary_status))
    evaluator.add_custom_node(
        result=bool(borrow_status_recorded >= 3),
        id="openlibrary_borrow_status",
        desc="[Perception Node] openlibrary.org:F1:P1 - Record borrowing status (blue Borrow button or alternatives like Join waitlist) for all 3 books",
        parent=openlibrary_node,
        critical=False
    )

    # 3.3 Getty Museum section
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty Museum artifacts matching mythological figures from books",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F1:A2 - Navigate Getty Museum via Explore Art or search
    getty_items_found = sum(1 for book in all_books if book and is_getty_url(book.getty_link))
    evaluator.add_custom_node(
        result=bool(getty_items_found >= 3),
        id="getty_navigation",
        desc="[Action Node] getty.edu:F1:A2 - Navigate Getty Museum (via Explore Art menu or search) to find artifacts for all 3 books",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F1:P1 - Match artifacts to mythological figures
    figure_matches = sum(1 for book in all_books
                        if book and book.getty_item_name and book.mythological_figure
                        and mentions_mythological_figure(book.getty_item_name))
    evaluator.add_custom_node(
        result=bool(figure_matches >= 3),
        id="getty_figure_matching",
        desc="[Perception Node] getty.edu:F1:P1 - Identify artifacts that correspond to mythological figures mentioned in each book (all 3 books)",
        parent=getty_node,
        critical=False
    )

    # Additional check: mentions ancient art types
    mentions_art_types = has_any_ci(answer, ['sculpture', 'pottery', 'vase', 'statue', 'relief', 'artifact'])
    evaluator.add_custom_node(
        result=bool(mentions_art_types),
        id="getty_art_types",
        desc="Mentions types of ancient art (sculpture, pottery, etc.) from Getty collection",
        parent=getty_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall task completeness: all required fields for 3 books",
        parent=root,
        critical=False
    )

    # Check that all three books have complete information
    complete_books = sum(1 for book in all_books
                        if book and book.title and book.author
                        and book.goodreads_rating_count and book.goodreads_link
                        and book.openlibrary_status and book.openlibrary_link
                        and book.getty_item_name and book.getty_link)

    evaluator.add_custom_node(
        result=bool(complete_books >= 3),
        id="three_complete_books",
        desc="All three books have complete information (title, author, ratings, links from all platforms)",
        parent=completeness_node,
        critical=False
    )

    # Check that answer mentions all three platforms
    mentions_goodreads = ci_contains(answer, 'goodreads')
    mentions_openlibrary = ci_contains(answer, 'openlibrary')
    mentions_getty = ci_contains(answer, 'getty')

    evaluator.add_custom_node(
        result=bool(mentions_goodreads and mentions_openlibrary and mentions_getty),
        id="all_platforms_mentioned",
        desc="Answer mentions all three platforms (Goodreads, OpenLibrary, Getty Museum)",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
