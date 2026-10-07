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
TASK_ID = "task-8b2d27"
TASK_DESCRIPTION = 'I’m looking for a few “hidden gem” films worth collecting—specifically cult movies released by the Criterion Collection that are underrated by critics but beloved by fans.  \nPlease first browse the in-stock Blu-ray titles on the Criterion website and try filtering candidates by genre, preferably Horror or Sci-Fi. If there is no usable genre filter at the moment, the filter is malfunctioning, or the filtered results are too limited, continue by manually selecting clearly Horror/Sci-Fi titles (Thriller can be included as a supplement) from the visible in-stock Blu-rays on that page, and state the basis for your classification.\n\nYou need to check each film’s Rotten Tomatoes critic score (Tomatometer) and Letterboxd audience rating. Please try to find 3 films that meet both of these criteria: **Tomatometer below 60%** and **Letterboxd rating above 3.6**.  \nIf fewer than 3 films meet the strict criteria, keep **Tomatometer < 60%** as a fixed condition, then fill the remaining slots by selecting titles with the highest Letterboxd ratings, and clearly label which entries strictly meet both conditions and which are supplemental picks, along with their corresponding scores.\n\nFinally, provide the three films’ titles, directors, exact scores from both platforms, and confirm whether each title is currently marked **In Stock** on the Criterion website.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmEntry(BaseModel):
    """A single film entry extracted from the answer"""
    title: Optional[str] = None
    director: Optional[str] = None
    tomatometer: Optional[str] = None
    letterboxd_rating: Optional[str] = None
    in_stock: Optional[str] = None
    is_strict_match: Optional[bool] = None


class ExtractedFilms(BaseModel):
    """All films extracted from the answer"""
    films: List[FilmEntry] = Field(default_factory=list)
    genre_filter_attempted: Optional[bool] = None
    manual_classification_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_films_from_answer() -> str:
    return """
Extract all films mentioned in the answer that the user evaluated for the task.

For each film, extract:
- title: the film's title exactly as stated
- director: the director's name if mentioned
- tomatometer: the Rotten Tomatoes Tomatometer score exactly as stated (include % if present)
- letterboxd_rating: the Letterboxd rating exactly as stated
- in_stock: whether the film is marked as "In Stock" on Criterion (extract the exact stock status phrase if present)
- is_strict_match: true if the answer explicitly labels this film as meeting both criteria (Tomatometer < 60% AND Letterboxd > 3.6), false if labeled as supplemental/fallback, null if not specified

Also extract:
- genre_filter_attempted: true if the answer mentions attempting to use a genre filter on Criterion
- manual_classification_mentioned: true if the answer mentions manually classifying films by genre

If any field is missing, set it to null. Return empty list if no films found.
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
    # Extract first number found
    m = re.search(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_tomatometer(text: Optional[str]) -> bool:
    """Check if text looks like a valid Tomatometer score"""
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Tomatometer is 0-100%
    return 0 <= num <= 100


def looks_like_letterboxd_rating(text: Optional[str]) -> bool:
    """Check if text looks like a valid Letterboxd rating"""
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Letterboxd is typically 0-5 scale
    return 0 <= num <= 5


def tomatometer_below_60(text: Optional[str]) -> bool:
    """Check if Tomatometer score is below 60%"""
    num = extract_float(text)
    if num is None:
        return False
    return num < 60


def letterboxd_above_36(text: Optional[str]) -> bool:
    """Check if Letterboxd rating is above 3.6"""
    num = extract_float(text)
    if num is None:
        return False
    return num > 3.6


def looks_like_in_stock(text: Optional[str]) -> bool:
    """Check if text indicates in-stock status"""
    if not text:
        return False
    return has_any_ci(text, ['in stock', 'in-stock', 'available'])


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_films_from_answer(),
        template_class=ExtractedFilms,
        extraction_name="films_data"
    )

    films = extracted.films if extracted and extracted.films else []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Criterion website section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion Collection website navigation and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A1 - Format filter (Blu-ray)
    criterion_bluray_mentioned = has_any_ci(answer, ['blu-ray', 'bluray', 'blu ray'])
    criterion_website_mentioned = has_any_ci(answer, ['criterion'])

    evaluator.add_custom_node(
        result=bool(criterion_bluray_mentioned and criterion_website_mentioned),
        id="criterion_format_filter",
        desc="[Action Node] criterion.com:F1:A1 - Browse in-stock Blu-ray titles on Criterion website",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A2 - Genre filter (Horror/Sci-Fi)
    genre_filter_attempted = extracted.genre_filter_attempted if extracted else False
    horror_scifi_mentioned = has_any_ci(answer, ['horror', 'sci-fi', 'science fiction', 'thriller'])

    evaluator.add_custom_node(
        result=bool(genre_filter_attempted or (horror_scifi_mentioned and criterion_website_mentioned)),
        id="criterion_genre_filter",
        desc="[Action Node] criterion.com:F1:A2 - Filter or manually select Horror/Sci-Fi titles (Thriller as supplement)",
        parent=criterion_node,
        critical=False
    )

    # Check if manual classification basis is stated
    manual_classification_mentioned = extracted.manual_classification_mentioned if extracted else False
    classification_basis_mentioned = has_any_ci(answer, ['basis', 'classification', 'classified', 'selected', 'chose'])

    evaluator.add_custom_node(
        result=bool(manual_classification_mentioned or (horror_scifi_mentioned and classification_basis_mentioned)),
        id="criterion_classification_basis",
        desc="States the basis for manual genre classification when applicable",
        parent=criterion_node,
        critical=False
    )

    # 3.2 Cross-platform score checking
    scores_node = evaluator.add_parallel(
        id="scores_section",
        desc="Rotten Tomatoes and Letterboxd score verification",
        parent=root,
        critical=False
    )

    # Check if we have films with valid scores
    films_with_valid_scores = 0
    films_with_tomatometer = 0
    films_with_letterboxd = 0
    strict_matches = 0

    for film in films:
        has_tomatometer = looks_like_tomatometer(film.tomatometer)
        has_letterboxd = looks_like_letterboxd_rating(film.letterboxd_rating)

        if has_tomatometer:
            films_with_tomatometer += 1
        if has_letterboxd:
            films_with_letterboxd += 1
        if has_tomatometer and has_letterboxd:
            films_with_valid_scores += 1

        # Check strict match criteria
        if (has_tomatometer and tomatometer_below_60(film.tomatometer) and
            has_letterboxd and letterboxd_above_36(film.letterboxd_rating)):
            strict_matches += 1

    # [Perception Node] rottentomatoes.com:F3:P1 - Visual perception to match correct film version
    rt_mentioned = has_any_ci(answer, ['rotten tomatoes', 'tomatometer'])
    film_matching_context = has_any_ci(answer, ['year', 'director', 'version', 'criterion'])

    evaluator.add_custom_node(
        result=bool(rt_mentioned and films_with_tomatometer >= 3 and film_matching_context),
        id="rottentomatoes_visual_perception",
        desc="[Perception Node] rottentomatoes.com:F3:P1 - Visual perception to identify correct film entries matching Criterion versions",
        parent=scores_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P6 - Extract Letterboxd rating values
    letterboxd_mentioned = has_any_ci(answer, ['letterboxd'])

    evaluator.add_custom_node(
        result=bool(letterboxd_mentioned and films_with_letterboxd >= 3),
        id="letterboxd_rating_extraction",
        desc="[Perception Node] letterboxd.com:F3:P6 - Extract Letterboxd audience ratings for comparison",
        parent=scores_node,
        critical=False
    )

    # 3.3 Criteria filtering and result presentation
    results_node = evaluator.add_sequential(
        id="results_section",
        desc="Film selection based on score criteria and result presentation",
        parent=root,
        critical=False
    )

    # Check if at least 3 films are provided
    evaluator.add_custom_node(
        result=bool(len(films) >= 3),
        id="three_films_provided",
        desc="Provides at least 3 film recommendations",
        parent=results_node,
        critical=False
    )

    # Check if films meet the Tomatometer < 60% requirement
    films_meeting_tomatometer = sum(1 for f in films if tomatometer_below_60(f.tomatometer))

    evaluator.add_custom_node(
        result=bool(films_meeting_tomatometer >= 3),
        id="tomatometer_criteria",
        desc="At least 3 films have Tomatometer below 60%",
        parent=results_node,
        critical=False
    )

    # Check if strict/supplemental labeling is provided when needed
    has_labeling = any(f.is_strict_match is not None for f in films)
    labeling_keywords = has_any_ci(answer, ['strict', 'supplemental', 'meet both', 'fallback', 'fill'])

    evaluator.add_custom_node(
        result=bool(strict_matches < 3 and (has_labeling or labeling_keywords)),
        id="supplemental_labeling",
        desc="Clearly labels which films strictly meet criteria vs. supplemental picks when fewer than 3 strict matches",
        parent=results_node,
        critical=False
    )

    # Check completeness of film information
    complete_films = 0
    for film in films:
        if (film.title and film.director and
            looks_like_tomatometer(film.tomatometer) and
            looks_like_letterboxd_rating(film.letterboxd_rating)):
            complete_films += 1

    evaluator.add_custom_node(
        result=bool(complete_films >= 3),
        id="complete_film_info",
        desc="Provides complete information (title, director, both scores) for at least 3 films",
        parent=results_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A11 - Stock status verification
    films_with_stock_info = sum(1 for f in films if looks_like_in_stock(f.in_stock))
    stock_mentioned = has_any_ci(answer, ['in stock', 'in-stock', 'stock', 'available'])

    evaluator.add_custom_node(
        result=bool(stock_mentioned and films_with_stock_info >= 3),
        id="criterion_stock_verification",
        desc="[Action Node] criterion.com:F1:A11 - Confirms In Stock status for recommended films on Criterion website",
        parent=results_node,
        critical=False
    )

    # Overall quality checks
    evaluator.add_custom_node(
        result=bool(films_with_valid_scores >= 3),
        id="valid_scores_present",
        desc="At least 3 films have valid scores from both platforms",
        parent=results_node,
        critical=False
    )

    # Check if scores are presented in exact format
    exact_scores_mentioned = has_any_ci(answer, ['%', 'tomatometer', 'letterboxd'])

    evaluator.add_custom_node(
        result=bool(exact_scores_mentioned and films_with_valid_scores >= 3),
        id="exact_scores_format",
        desc="Provides exact scores from both platforms in clear format",
        parent=results_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
