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
TASK_ID = "task-5e833f"
TASK_DESCRIPTION = "I'm interested in researching the phenomenon where film critics' opinions and audience preferences diverge significantly.\n\nFirst, navigate to the 'Movies' section on Rotten Tomatoes. Find a film currently 'In Theaters' or available 'At Home' (streaming) that meets the following criteria: a high Tomatometer score from professional critics (over 80%) but a low Audience Score (under 50%).\n\nOnce you find a suitable movie, please tell me its name and the exact Tomatometer and Audience Scores.\n\nNext, search for this film on IMDb. Go to its 'User Reviews' section, filter for reviews rated 1 or 2 stars, and sort them by 'Helpfulness'. Summarize the top-ranked lengthy review, identifying the primary reasons for audience dissatisfaction."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MovieInfo(BaseModel):
    """Movie details extracted from the answer"""
    movie_name: Optional[str] = None
    tomatometer_score: Optional[str] = None
    audience_score: Optional[str] = None


class ReviewSummary(BaseModel):
    """Review summary extracted from the answer"""
    review_summary: Optional[str] = None
    primary_reasons: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_movie_info() -> str:
    return """
Extract the movie information reported by the user from Rotten Tomatoes:

- movie_name: the name of the film found that meets the criteria
- tomatometer_score: the Tomatometer score exactly as stated (include % if present)
- audience_score: the Audience Score exactly as stated (include % if present)

If any field is missing in the answer, set it to null.
"""


def prompt_extract_review_summary() -> str:
    return """
Extract the IMDb user review summary from the answer:

- review_summary: the summary or content of the top-ranked review from IMDb
- primary_reasons: the primary reasons for audience dissatisfaction identified in the review

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


def extract_percentage(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*%?', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_high_score(score_text: Optional[str], threshold: float = 80.0) -> bool:
    score = extract_percentage(score_text)
    if score is None:
        return False
    return score > threshold


def looks_like_low_score(score_text: Optional[str], threshold: float = 50.0) -> bool:
    score = extract_percentage(score_text)
    if score is None:
        return False
    return score < threshold


def mentions_filtering_low_ratings(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    patterns = [
        r'\b1\s*(star|stars)\b',
        r'\b2\s*(star|stars)\b',
        r'\b1\s*or\s*2\s*(star|stars)\b',
        r'filter.*\b(1|2)\b',
        r'rated\s*(1|2)\b'
    ]
    return any(re.search(p, answer_text.lower()) for p in patterns)


def mentions_sort_by_helpfulness(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['helpfulness', 'helpful', 'most helpful'])


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
    movie_info = await evaluator.extract(
        prompt=prompt_extract_movie_info(),
        template_class=MovieInfo,
        extraction_name="movie_info"
    )

    review_summary = await evaluator.extract(
        prompt=prompt_extract_review_summary(),
        template_class=ReviewSummary,
        extraction_name="review_summary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Rotten Tomatoes part
    rt_node = evaluator.add_sequential(
        id="rotten_tomatoes_section",
        desc="Rotten Tomatoes - Find a film with high critic score but low audience score",
        parent=root,
        critical=False
    )

    # [Action Node] rottentomatoes.com:F2:A14 - Navigate and filter for movies meeting criteria
    rt_navigation_ok = has_any_ci(answer, ['rotten tomatoes'])
    rt_section_ok = has_any_ci(answer, ['in theaters', 'at home', 'streaming', 'movies'])

    evaluator.add_custom_node(
        result=bool(rt_navigation_ok and rt_section_ok),
        id="rt_action_navigate_and_filter",
        desc="[Action Node] rottentomatoes.com:F2:A14 - Navigate to Rotten Tomatoes Movies section and filter for films 'In Theaters' or 'At Home'",
        parent=rt_node,
        critical=False
    )

    # [Perception Node] rottentomatoes.com:F2:P2 - Identify high Tomatometer and low Audience Score
    has_movie_name = bool(movie_info and movie_info.movie_name and movie_info.movie_name.strip())
    tomatometer_high = looks_like_high_score(movie_info.tomatometer_score, threshold=80.0)
    audience_low = looks_like_low_score(movie_info.audience_score, threshold=50.0)

    evaluator.add_custom_node(
        result=bool(has_movie_name and tomatometer_high and audience_low),
        id="rt_perception_score_divergence",
        desc="[Perception Node] rottentomatoes.com:F2:P2 - Identify film with Tomatometer over 80% and Audience Score under 50%",
        parent=rt_node,
        critical=False
    )

    # Additional check: Reports exact scores
    has_exact_scores = bool(movie_info.tomatometer_score and movie_info.audience_score)
    evaluator.add_custom_node(
        result=bool(has_exact_scores),
        id="rt_reports_exact_scores",
        desc="Reports the exact Tomatometer and Audience Score values",
        parent=rt_node,
        critical=False
    )

    # 3.2 IMDb part
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb - Search for film, filter low-rated reviews, and summarize top review",
        parent=root,
        critical=False
    )

    # Action: Search for the film on IMDb
    imdb_search_ok = has_any_ci(answer, ['imdb'])
    has_movie_context = has_movie_name or has_any_ci(answer, ['film', 'movie'])

    evaluator.add_custom_node(
        result=bool(imdb_search_ok and has_movie_context),
        id="imdb_action_search",
        desc="Search for the identified film on IMDb",
        parent=imdb_node,
        critical=False
    )

    # Action: Navigate to User Reviews section
    user_reviews_ok = has_any_ci(answer, ['user reviews', 'reviews'])

    evaluator.add_custom_node(
        result=bool(user_reviews_ok),
        id="imdb_action_user_reviews",
        desc="Navigate to the User Reviews section on IMDb",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F4:A40 - Filter for 1 or 2 star reviews
    filter_low_ratings_ok = mentions_filtering_low_ratings(answer)

    evaluator.add_custom_node(
        result=bool(filter_low_ratings_ok),
        id="imdb_action_filter_low_ratings",
        desc="[Action Node] imdb.com:F4:A40 - Filter reviews for 1 or 2 star ratings",
        parent=imdb_node,
        critical=False
    )

    # Action: Sort by Helpfulness
    sort_helpfulness_ok = mentions_sort_by_helpfulness(answer)

    evaluator.add_custom_node(
        result=bool(sort_helpfulness_ok),
        id="imdb_action_sort_helpfulness",
        desc="Sort filtered reviews by Helpfulness",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F4:P25 - Summarize top-ranked review and identify reasons
    has_review_summary = bool(review_summary and review_summary.review_summary and review_summary.review_summary.strip())
    has_reasons = bool(review_summary and review_summary.primary_reasons and review_summary.primary_reasons.strip())

    evaluator.add_custom_node(
        result=bool(has_review_summary and has_reasons),
        id="imdb_perception_review_content",
        desc="[Perception Node] imdb.com:F4:P25 - Summarize the top-ranked lengthy review and identify primary reasons for dissatisfaction",
        parent=imdb_node,
        critical=False
    )

    # Additional check: Mentions "lengthy" or substantial review
    lengthy_context = has_any_ci(answer, ['lengthy', 'long', 'detailed', 'comprehensive'])
    evaluator.add_custom_node(
        result=bool(lengthy_context or has_review_summary),
        id="imdb_lengthy_review_context",
        desc="Indicates focus on a substantial/lengthy review",
        parent=imdb_node,
        critical=False
    )

    # 3.3 Overall coherence check
    coherence_node = evaluator.add_parallel(
        id="overall_coherence",
        desc="Overall task coherence and completeness",
        parent=root,
        critical=False
    )

    # Check that both platforms are addressed
    both_platforms = rt_navigation_ok and imdb_search_ok
    evaluator.add_custom_node(
        result=bool(both_platforms),
        id="coherence_both_platforms",
        desc="Answer addresses both Rotten Tomatoes and IMDb parts of the task",
        parent=coherence_node,
        critical=False
    )

    # Check that the connection between platforms is logical (same movie)
    movie_continuity = has_movie_name and (imdb_search_ok or has_any_ci(answer, ['same', 'this film', 'the film', 'the movie']))
    evaluator.add_custom_node(
        result=bool(movie_continuity),
        id="coherence_movie_continuity",
        desc="Maintains continuity - searches for the same movie on IMDb that was found on Rotten Tomatoes",
        parent=coherence_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
