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
TASK_ID = "task-683779"
TASK_DESCRIPTION = 'I’m a fan of Frank Herbert’s original *Dune* novel and want to know how much the newly released *Dune: Part Two* has changed it. Please do an in-depth comparison for me.\n\nFirst, go to Goodreads and find Frank Herbert’s original *Dune*. I want to see the latest discussion, so sort the reviews/comments by **“Newest.”** Find one recent, detailed long review that explicitly mentions “movie” or “film,” and note the user’s core viewpoint.\n\nThen go to Letterboxd and find *Dune: Part Two*. Browse several pages of reviews, and/or expand collapsed spoiler content. Prioritize finding **two reviews** that explicitly compare the film with the **book**. If there are fewer than two explicit book-to-film comparison reviews, you may supplement up to two total by selecting in-depth reviews that clearly discuss **adaptation trade-offs / faithfulness / omitted or changed plot points**. In your results, label which ones are direct book-film comparisons and which ones are adaptation-focused discussions.\n\nFinally, summarize the main points of disagreement between hardcore audiences on these two platforms regarding this adaptation, and list the user IDs of the reviewers you cited.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GoodreadsReviewInfo(BaseModel):
    """Information extracted about the Goodreads review"""
    sorting_mentioned: Optional[bool] = False
    review_mentions_movie_or_film: Optional[bool] = False
    user_id_or_name: Optional[str] = None
    core_viewpoint_summary: Optional[str] = None


class LetterboxdReviewsInfo(BaseModel):
    """Information extracted about Letterboxd reviews"""
    number_of_reviews_found: Optional[int] = 0
    pagination_mentioned: Optional[bool] = False
    spoiler_expansion_mentioned: Optional[bool] = False
    review_user_ids: Optional[List[str]] = Field(default_factory=list)
    direct_book_comparison_count: Optional[int] = 0
    adaptation_discussion_count: Optional[int] = 0


class ComparisonSummary(BaseModel):
    """Summary of disagreements and citations"""
    has_disagreement_summary: Optional[bool] = False
    all_cited_user_ids: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_goodreads_info() -> str:
    return """
Extract information about the Goodreads review activity from the answer:

- sorting_mentioned: true if the answer mentions sorting reviews by "Newest" or changing sort order, false otherwise.
- review_mentions_movie_or_film: true if the answer found a review that mentions "movie" or "film", false otherwise.
- user_id_or_name: the Goodreads username or user ID of the reviewer whose review was selected. Set to null if not provided.
- core_viewpoint_summary: a brief description of the reviewer's core viewpoint about the book/movie. Set to null if not provided.

If any field is missing, use the default value or null as appropriate.
"""


def prompt_extract_letterboxd_info() -> str:
    return """
Extract information about the Letterboxd review search from the answer:

- number_of_reviews_found: the total number of reviews mentioned or found (aim for 2 per task requirements).
- pagination_mentioned: true if the answer mentions browsing multiple pages or navigating through pages, false otherwise.
- spoiler_expansion_mentioned: true if the answer mentions expanding collapsed spoiler content, false otherwise.
- review_user_ids: a list of Letterboxd usernames or user IDs for the reviews cited. Empty list if none provided.
- direct_book_comparison_count: number of reviews explicitly comparing the film with the book.
- adaptation_discussion_count: number of reviews discussing adaptation trade-offs/faithfulness/omissions without explicit book comparison.

If any field is missing, use the default value.
"""


def prompt_extract_comparison_summary() -> str:
    return """
Extract the final comparison and citation information from the answer:

- has_disagreement_summary: true if the answer provides a summary of disagreements or main points of contention between audiences on the two platforms, false otherwise.
- all_cited_user_ids: a list of all user IDs/usernames cited in the answer (from both Goodreads and Letterboxd). Empty list if none provided.

If any field is missing, use the default value.
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


def mentions_site(text: Optional[str], site: str) -> bool:
    if not text:
        return False
    return ci_contains(text, site)


def mentions_sorting_or_newest(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['newest', 'sort', 'sorted', 'sorting', 'sort by']
    return has_any_ci(text, keywords)


def mentions_pagination(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['page', 'pages', 'next page', 'browse', 'browsing', 'multiple pages', 'several pages']
    return has_any_ci(text, keywords)


def mentions_spoiler_expansion(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['expand', 'collapsed', 'spoiler', 'unfold', 'show more', 'reveal']
    return has_any_ci(text, keywords)


def mentions_book_comparison(text: Optional[str]) -> bool:
    if not text:
        return False
    book_keywords = ['book', 'novel', 'source material', 'original']
    compare_keywords = ['compare', 'comparison', 'versus', 'vs', 'differ', 'different from']
    has_book = has_any_ci(text, book_keywords)
    has_compare = has_any_ci(text, compare_keywords)
    return has_book and has_compare


def mentions_adaptation_discussion(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['adaptation', 'faithful', 'faithfulness', 'omitted', 'changed', 'trade-off', 'tradeoff']
    return has_any_ci(text, keywords)


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
    goodreads_info = await evaluator.extract(
        prompt=prompt_extract_goodreads_info(),
        template_class=GoodreadsReviewInfo,
        extraction_name="goodreads_review_info"
    )

    letterboxd_info = await evaluator.extract(
        prompt=prompt_extract_letterboxd_info(),
        template_class=LetterboxdReviewsInfo,
        extraction_name="letterboxd_reviews_info"
    )

    comparison_info = await evaluator.extract(
        prompt=prompt_extract_comparison_summary(),
        template_class=ComparisonSummary,
        extraction_name="comparison_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Goodreads section
    goodreads_node = evaluator.add_sequential(
        id="goodreads_section",
        desc="Goodreads review search for Frank Herbert's Dune",
        parent=root,
        critical=False
    )

    # Check if answer mentions Goodreads and Dune
    goodreads_navigation_ok = mentions_site(answer, 'goodreads') and has_any_ci(answer, ['dune', 'frank herbert'])
    evaluator.add_custom_node(
        result=bool(goodreads_navigation_ok),
        id="goodreads_navigation",
        desc="Navigate to Goodreads and locate Frank Herbert's Dune",
        parent=goodreads_node,
        critical=False
    )

    # [Action Node] goodreads.com:F3:A7 - Sort reviews by "Newest"
    sorting_action_ok = (
        goodreads_info.sorting_mentioned or
        mentions_sorting_or_newest(answer)
    )
    evaluator.add_custom_node(
        result=bool(sorting_action_ok),
        id="goodreads_sorting_action",
        desc="[Action Node] goodreads.com:F3:A7 - Sort reviews/comments by 'Newest' using dropdown menu",
        parent=goodreads_node,
        critical=False
    )

    # Check if a review mentioning movie/film was found
    movie_mention_ok = (
        goodreads_info.review_mentions_movie_or_film or
        has_any_ci(answer, ['movie', 'film'])
    )
    evaluator.add_custom_node(
        result=bool(movie_mention_ok),
        id="goodreads_movie_mention",
        desc="Found a review that explicitly mentions 'movie' or 'film'",
        parent=goodreads_node,
        critical=False
    )

    # Check if user ID and viewpoint were captured
    user_captured_ok = bool(goodreads_info.user_id_or_name and goodreads_info.user_id_or_name.strip())
    viewpoint_captured_ok = bool(goodreads_info.core_viewpoint_summary and goodreads_info.core_viewpoint_summary.strip())
    evaluator.add_custom_node(
        result=bool(user_captured_ok and viewpoint_captured_ok),
        id="goodreads_user_and_viewpoint",
        desc="Captured the reviewer's user ID/name and their core viewpoint",
        parent=goodreads_node,
        critical=False
    )

    # 3.2 Letterboxd section
    letterboxd_node = evaluator.add_sequential(
        id="letterboxd_section",
        desc="Letterboxd review search for Dune: Part Two",
        parent=root,
        critical=False
    )

    # Check if answer mentions Letterboxd and Dune: Part Two
    letterboxd_navigation_ok = mentions_site(answer, 'letterboxd') and has_any_ci(answer, ['dune', 'part two', 'part 2'])
    evaluator.add_custom_node(
        result=bool(letterboxd_navigation_ok),
        id="letterboxd_navigation",
        desc="Navigate to Letterboxd and locate Dune: Part Two",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A16 - Browse multiple pages
    pagination_action_ok = (
        letterboxd_info.pagination_mentioned or
        mentions_pagination(answer)
    )
    evaluator.add_custom_node(
        result=bool(pagination_action_ok),
        id="letterboxd_pagination_action",
        desc="[Action Node] letterboxd.com:F3:A16 - Browse several pages of reviews",
        parent=letterboxd_node,
        critical=False
    )

    # [Action Node] letterboxd.com:F3:A15 - Expand collapsed spoiler content
    spoiler_expansion_ok = (
        letterboxd_info.spoiler_expansion_mentioned or
        mentions_spoiler_expansion(answer)
    )
    evaluator.add_custom_node(
        result=bool(spoiler_expansion_ok),
        id="letterboxd_spoiler_expansion",
        desc="[Action Node] letterboxd.com:F3:A15 - Expand collapsed spoiler content in reviews",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P7 - Find book-comparison reviews
    book_comparison_mentioned = mentions_book_comparison(answer)
    adaptation_mentioned = mentions_adaptation_discussion(answer)

    found_reviews_ok = (
        letterboxd_info.number_of_reviews_found and
        letterboxd_info.number_of_reviews_found >= 2
    )

    book_film_comparison_ok = (
        (letterboxd_info.direct_book_comparison_count and letterboxd_info.direct_book_comparison_count >= 1) or
        book_comparison_mentioned
    )

    evaluator.add_custom_node(
        result=bool(found_reviews_ok and (book_film_comparison_ok or adaptation_mentioned)),
        id="letterboxd_book_comparison_perception",
        desc="[Perception Node] letterboxd.com:F3:P7 - Find reviews that explicitly compare the film with the book or discuss adaptation trade-offs",
        parent=letterboxd_node,
        critical=False
    )

    # Check if review types are labeled correctly
    labeling_keywords = ['direct comparison', 'book-film comparison', 'adaptation-focused', 'adaptation discussion']
    labeling_ok = has_any_ci(answer, labeling_keywords)
    evaluator.add_custom_node(
        result=bool(labeling_ok),
        id="letterboxd_review_labeling",
        desc="Labels which reviews are direct book-film comparisons vs adaptation-focused discussions",
        parent=letterboxd_node,
        critical=False
    )

    # Check if Letterboxd user IDs were captured
    letterboxd_users_ok = bool(letterboxd_info.review_user_ids and len(letterboxd_info.review_user_ids) >= 1)
    evaluator.add_custom_node(
        result=bool(letterboxd_users_ok),
        id="letterboxd_user_ids",
        desc="Captured user IDs/names for the cited Letterboxd reviews",
        parent=letterboxd_node,
        critical=False
    )

    # 3.3 Summary and synthesis section
    summary_node = evaluator.add_sequential(
        id="summary_section",
        desc="Final comparison summary and citations",
        parent=root,
        critical=False
    )

    # Check if disagreements are summarized
    disagreement_summary_ok = (
        comparison_info.has_disagreement_summary or
        has_any_ci(answer, ['disagreement', 'contention', 'debate', 'different opinions', 'main points'])
    )
    evaluator.add_custom_node(
        result=bool(disagreement_summary_ok),
        id="disagreement_summary",
        desc="Summarizes main points of disagreement between audiences on the two platforms",
        parent=summary_node,
        critical=False
    )

    # Check if user IDs are listed
    all_users_listed = (
        comparison_info.all_cited_user_ids and
        len(comparison_info.all_cited_user_ids) >= 2
    )
    user_id_keywords = ['user', 'reviewer', 'username', 'user id', 'cited']
    user_listing_mentioned = has_any_ci(answer, user_id_keywords)

    evaluator.add_custom_node(
        result=bool(all_users_listed or user_listing_mentioned),
        id="user_id_listing",
        desc="Lists the user IDs/names of all reviewers cited from both platforms",
        parent=summary_node,
        critical=False
    )

    # Check cross-platform synthesis quality
    cross_platform_keywords = ['both platforms', 'two platforms', 'goodreads and letterboxd', 'hardcore audiences', 'fans']
    synthesis_ok = has_any_ci(answer, cross_platform_keywords)
    evaluator.add_custom_node(
        result=bool(synthesis_ok),
        id="cross_platform_synthesis",
        desc="Provides cross-platform synthesis comparing perspectives from both Goodreads and Letterboxd",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
