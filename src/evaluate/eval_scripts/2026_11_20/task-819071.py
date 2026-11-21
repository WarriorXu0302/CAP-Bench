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
TASK_ID = "task-819071"
TASK_DESCRIPTION = 'I’m considering buying a few recently released film noir restoration Blu-rays from the Criterion Collection. First, check the Criterion website for film noir titles added in the last 3 months, filter for 4K UHD or Blu-ray formats, and try to identify 5 films (if fewer than 5 are available, record the actual count). Then, for each film, check IMDb for its rating and number of ratings, prioritizing titles with a rating above 7.5 and more than 5,000 ratings. Next, check Letterboxd for each film’s community rating and number of ratings, prioritizing titles with a rating above 4.0 and more than 2,000 ratings. If fewer than 3 films remain after applying both platform thresholds, fill the shortlist from the collected films by prioritizing how many criteria each title meets (IMDb rating/count and Letterboxd rating/count; the more conditions met, the higher the priority), and clearly mark in the output whether each condition is met along with the corresponding values.\n\nFinally, search YouTube for restoration comparison videos for each film (search: “Film Title restoration comparison”), and select comparison videos longer than 3 minutes with relatively high view counts, explaining why each was chosen. Based on restoration specs, cross-platform ratings, and comparison videos, recommend the 3 most worthwhile films to buy (if fewer than 3 candidates are available, recommend the actual number and explain why).\n\nFor each recommended film, output:\n- Title  \n- Director  \n- Criterion restoration format (4K UHD/Blu-ray)  \n- Criterion price  \n- Criterion detail page link  \n- IMDb rating  \n- IMDb rating count  \n- IMDb link  \n- Letterboxd rating  \n- Letterboxd rating count  \n- Letterboxd link  \n- YouTube comparison video title  \n- YouTube view count  \n- YouTube video link  \n- Recommendation rationale (combining restoration quality, ratings, and community reception)'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmRecommendation(BaseModel):
    """A single film recommendation extracted from the answer"""
    title: Optional[str] = None
    director: Optional[str] = None
    criterion_format: Optional[str] = None
    criterion_price: Optional[str] = None
    criterion_link: Optional[str] = None
    imdb_rating: Optional[float] = None
    imdb_rating_count: Optional[int] = None
    imdb_link: Optional[str] = None
    letterboxd_rating: Optional[float] = None
    letterboxd_rating_count: Optional[int] = None
    letterboxd_link: Optional[str] = None
    youtube_video_title: Optional[str] = None
    youtube_view_count: Optional[str] = None
    youtube_link: Optional[str] = None
    recommendation_rationale: Optional[str] = None


class FilmRecommendations(BaseModel):
    """All film recommendations extracted from the answer"""
    films: List[FilmRecommendation] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_recommendations() -> str:
    return """
Extract all film recommendations from the answer. For each recommended film, extract:
- title: film title
- director: director name
- criterion_format: format (4K UHD or Blu-ray)
- criterion_price: price as stated
- criterion_link: Criterion detail page URL
- imdb_rating: IMDb rating (as a number)
- imdb_rating_count: IMDb rating count (as an integer)
- imdb_link: IMDb page URL
- letterboxd_rating: Letterboxd rating (as a number)
- letterboxd_rating_count: Letterboxd rating count (as an integer)
- letterboxd_link: Letterboxd page URL
- youtube_video_title: YouTube comparison video title
- youtube_view_count: YouTube view count as stated
- youtube_link: YouTube video URL
- recommendation_rationale: the explanation for why this film is recommended

Also extract:
- total_count: how many films were actually recommended

If any field is missing, set it to null. If no films are present, return an empty list.
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


def is_valid_url(url: Optional[str], domain: str) -> bool:
    if not url:
        return False
    return domain.lower() in url.lower() and url.startswith('http')


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Remove commas and extract first number
    cleaned = re.sub(r'[,\s]', '', str(text))
    m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def looks_like_film_noir(answer: str) -> bool:
    return has_any_ci(answer, ['film noir', 'noir'])


def looks_like_recent_releases(answer: str) -> bool:
    return has_any_ci(answer, ['last 3 months', 'recent', 'new releases', 'recently released'])


def looks_like_format_filter(answer: str) -> bool:
    return has_any_ci(answer, ['4k uhd', 'blu-ray', 'blu ray', 'format'])


def mentions_imdb_criteria(answer: str) -> bool:
    rating_mention = has_any_ci(answer, ['7.5', '7.0', 'rating'])
    count_mention = has_any_ci(answer, ['5000', '5,000', 'ratings'])
    return rating_mention or count_mention


def mentions_letterboxd_criteria(answer: str) -> bool:
    rating_mention = has_any_ci(answer, ['4.0', '4.5', 'rating'])
    count_mention = has_any_ci(answer, ['2000', '2,000', 'ratings'])
    return rating_mention or count_mention


def mentions_youtube_criteria(answer: str) -> bool:
    return has_any_ci(answer, ['3 minutes', 'view count', 'restoration comparison', 'comparison'])


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
    recommendations = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=FilmRecommendations,
        extraction_name="film_recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Criterion Collection section
    criterion_node = evaluator.add_sequential(
        id="criterion_section",
        desc="Criterion Collection film noir discovery and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] criterion.com:F1:A2 - Genre filtering (Film Noir)
    criterion_genre_ok = looks_like_film_noir(answer) and has_any_ci(answer, ['criterion'])
    evaluator.add_custom_node(
        result=bool(criterion_genre_ok),
        id="criterion_genre_filter",
        desc="[Action Node] criterion.com:F1:A2 - Filter for Film Noir genre in Criterion Collection",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A3 - New releases filtering
    criterion_new_releases_ok = looks_like_recent_releases(answer) and has_any_ci(answer, ['criterion'])
    evaluator.add_custom_node(
        result=bool(criterion_new_releases_ok),
        id="criterion_new_releases_filter",
        desc="[Action Node] criterion.com:F1:A3 - Filter for New Releases (last 3 months)",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A1 - Format filtering (4K UHD or Blu-ray)
    criterion_format_ok = looks_like_format_filter(answer) and has_any_ci(answer, ['criterion'])
    evaluator.add_custom_node(
        result=bool(criterion_format_ok),
        id="criterion_format_filter",
        desc="[Action Node] criterion.com:F1:A1 - Filter for 4K UHD or Blu-ray formats",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F1:A9 - Click into detail pages
    has_criterion_links = any(
        is_valid_url(f.criterion_link, 'criterion')
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_criterion_links),
        id="criterion_detail_navigation",
        desc="[Action Node] criterion.com:F1:A9 - Navigate to film detail pages to get pricing and specs",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F1:P2 - Release date identification
    criterion_date_context = looks_like_recent_releases(answer)
    evaluator.add_custom_node(
        result=bool(criterion_date_context),
        id="criterion_release_date_perception",
        desc="[Perception Node] criterion.com:F1:P2 - Identify recent release dates on film cards",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F1:P9 - Card information recognition
    has_titles_directors = any(
        f.title and f.director
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_titles_directors),
        id="criterion_card_info_perception",
        desc="[Perception Node] criterion.com:F1:P9 - Extract film titles and directors from cards",
        parent=criterion_node,
        critical=False
    )

    # [Action Node] criterion.com:F2:A13 - Format tab switching
    has_formats = any(
        f.criterion_format and f.criterion_format.strip()
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_formats),
        id="criterion_format_tab_switching",
        desc="[Action Node] criterion.com:F2:A13 - Switch between format tabs to check prices",
        parent=criterion_node,
        critical=False
    )

    # [Perception Node] criterion.com:F2:P5 - Format availability recognition
    format_valid = any(
        f.criterion_format and has_any_ci(f.criterion_format, ['4k uhd', 'blu-ray', 'blu ray'])
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(format_valid),
        id="criterion_format_availability",
        desc="[Perception Node] criterion.com:F2:P5 - Recognize available formats and pricing",
        parent=criterion_node,
        critical=False
    )

    # 3.2 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb rating and popularity verification",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F1:A1 - Search type selection
    imdb_search_ok = has_any_ci(answer, ['imdb']) and any(f.title for f in recommendations.films if f)
    evaluator.add_custom_node(
        result=bool(imdb_search_ok),
        id="imdb_search_type",
        desc="[Action Node] imdb.com:F1:A1 - Search for film titles on IMDb",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F1:A26 - Click search result cards
    has_imdb_links = any(
        is_valid_url(f.imdb_link, 'imdb')
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_imdb_links),
        id="imdb_result_click",
        desc="[Action Node] imdb.com:F1:A26 - Navigate to film detail pages from search results",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P1 - Card image recognition
    imdb_title_match = has_imdb_links and has_titles_directors
    evaluator.add_custom_node(
        result=bool(imdb_title_match),
        id="imdb_card_recognition",
        desc="[Perception Node] imdb.com:F3:P1 - Match film titles in search results",
        parent=imdb_node,
        critical=False
    )

    # [Action Node] imdb.com:F3:A7 - Detail tab switching
    has_imdb_ratings = any(
        f.imdb_rating is not None and f.imdb_rating_count is not None
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_imdb_ratings),
        id="imdb_tab_navigation",
        desc="[Action Node] imdb.com:F3:A7 - Navigate detail page to extract ratings",
        parent=imdb_node,
        critical=False
    )

    # Check IMDb criteria application (7.5+, 5000+)
    imdb_criteria_applied = mentions_imdb_criteria(answer)
    evaluator.add_custom_node(
        result=bool(imdb_criteria_applied),
        id="imdb_criteria_application",
        desc="Apply IMDb filtering criteria (rating > 7.5, count > 5000)",
        parent=imdb_node,
        critical=False
    )

    # 3.3 Letterboxd section
    letterboxd_node = evaluator.add_sequential(
        id="letterboxd_section",
        desc="Letterboxd community rating verification",
        parent=root,
        critical=False
    )

    # [Action Node] letterboxd.com:F1:A5 - Search result card click
    has_letterboxd_links = any(
        is_valid_url(f.letterboxd_link, 'letterboxd')
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_letterboxd_links),
        id="letterboxd_result_click",
        desc="[Action Node] letterboxd.com:F1:A5 - Navigate to film pages from search results",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P6 - Rating data extraction
    has_letterboxd_ratings = any(
        f.letterboxd_rating is not None and f.letterboxd_rating_count is not None
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_letterboxd_ratings),
        id="letterboxd_rating_extraction",
        desc="[Perception Node] letterboxd.com:F3:P6 - Extract community ratings and counts",
        parent=letterboxd_node,
        critical=False
    )

    # [Perception Node] letterboxd.com:F3:P7 - Review content understanding
    has_rationales = any(
        f.recommendation_rationale and len(f.recommendation_rationale) > 20
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_rationales),
        id="letterboxd_review_understanding",
        desc="[Perception Node] letterboxd.com:F3:P7 - Understand community reception from reviews",
        parent=letterboxd_node,
        critical=False
    )

    # Check Letterboxd criteria application (4.0+, 2000+)
    letterboxd_criteria_applied = mentions_letterboxd_criteria(answer)
    evaluator.add_custom_node(
        result=bool(letterboxd_criteria_applied),
        id="letterboxd_criteria_application",
        desc="Apply Letterboxd filtering criteria (rating > 4.0, count > 2000)",
        parent=letterboxd_node,
        critical=False
    )

    # 3.4 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube restoration comparison video discovery",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Video card click
    has_youtube_links = any(
        is_valid_url(f.youtube_link, 'youtube') or is_valid_url(f.youtube_link, 'youtu.be')
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_youtube_links),
        id="youtube_video_click",
        desc="[Action Node] youtube.com:F1:A22 - Select restoration comparison videos",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Thumbnail understanding
    has_comparison_titles = any(
        f.youtube_video_title and has_any_ci(f.youtube_video_title, ['restoration', 'comparison', 'remaster', '4k'])
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_comparison_titles),
        id="youtube_thumbnail_recognition",
        desc="[Perception Node] youtube.com:F1:P4 - Identify restoration comparison videos from thumbnails",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F9:A4 - Duration filter
    youtube_duration_context = mentions_youtube_criteria(answer) and has_any_ci(answer, ['3 minutes', '3 min'])
    evaluator.add_custom_node(
        result=bool(youtube_duration_context),
        id="youtube_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Filter for videos longer than 3 minutes",
        parent=youtube_node,
        critical=False
    )

    # [Action Node] youtube.com:F1:A69 - Scroll for more results
    youtube_scroll_context = has_any_ci(answer, ['view count', 'high view', 'most viewed'])
    evaluator.add_custom_node(
        result=bool(youtube_scroll_context),
        id="youtube_scroll_loading",
        desc="[Action Node] youtube.com:F1:A69 - Scroll to compare view counts",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F3:P3 - View count recognition
    has_view_counts = any(
        f.youtube_view_count and extract_number(f.youtube_view_count) is not None
        for f in recommendations.films if f
    )
    evaluator.add_custom_node(
        result=bool(has_view_counts),
        id="youtube_view_count_perception",
        desc="[Perception Node] youtube.com:F3:P3 - Extract view counts for comparison videos",
        parent=youtube_node,
        critical=False
    )

    # 3.5 Final recommendations validation
    final_node = evaluator.add_parallel(
        id="final_recommendations",
        desc="Final film recommendations with complete information",
        parent=root,
        critical=False
    )

    # Check that recommendations exist
    has_recommendations = recommendations and recommendations.films and len(recommendations.films) > 0
    evaluator.add_custom_node(
        result=bool(has_recommendations),
        id="has_film_recommendations",
        desc="Answer provides film recommendations",
        parent=final_node,
        critical=False
    )

    # Check completeness of each recommendation
    if has_recommendations:
        for idx, film in enumerate(recommendations.films[:3], 1):
            film_node = evaluator.add_parallel(
                id=f"film_{idx}_completeness",
                desc=f"Film {idx} recommendation completeness",
                parent=final_node,
                critical=False
            )

            # Title and director
            evaluator.add_custom_node(
                result=bool(film.title and film.director),
                id=f"film_{idx}_basic_info",
                desc=f"Film {idx} has title and director",
                parent=film_node,
                critical=False
            )

            # Criterion info (o1, o2, o3, o4, o5)
            criterion_complete = all([
                film.criterion_format,
                film.criterion_price,
                is_valid_url(film.criterion_link, 'criterion')
            ])
            evaluator.add_custom_node(
                result=bool(criterion_complete),
                id=f"film_{idx}_criterion_info",
                desc=f"Film {idx} has Criterion format, price, and link",
                parent=film_node,
                critical=False
            )

            # IMDb info (o6, o7, o8)
            imdb_complete = all([
                film.imdb_rating is not None,
                film.imdb_rating_count is not None,
                is_valid_url(film.imdb_link, 'imdb')
            ])
            evaluator.add_custom_node(
                result=bool(imdb_complete),
                id=f"film_{idx}_imdb_info",
                desc=f"Film {idx} has IMDb rating, count, and link",
                parent=film_node,
                critical=False
            )

            # Letterboxd info (o9, o10, o11)
            letterboxd_complete = all([
                film.letterboxd_rating is not None,
                film.letterboxd_rating_count is not None,
                is_valid_url(film.letterboxd_link, 'letterboxd')
            ])
            evaluator.add_custom_node(
                result=bool(letterboxd_complete),
                id=f"film_{idx}_letterboxd_info",
                desc=f"Film {idx} has Letterboxd rating, count, and link",
                parent=film_node,
                critical=False
            )

            # YouTube info (o12, o13, o14, o15)
            youtube_complete = all([
                film.youtube_video_title,
                film.youtube_view_count,
                is_valid_url(film.youtube_link, 'youtube') or is_valid_url(film.youtube_link, 'youtu.be')
            ])
            evaluator.add_custom_node(
                result=bool(youtube_complete),
                id=f"film_{idx}_youtube_info",
                desc=f"Film {idx} has YouTube video title, view count, and link",
                parent=film_node,
                critical=False
            )

            # Recommendation rationale (o16)
            evaluator.add_custom_node(
                result=bool(film.recommendation_rationale and len(film.recommendation_rationale) > 30),
                id=f"film_{idx}_rationale",
                desc=f"Film {idx} has detailed recommendation rationale",
                parent=film_node,
                critical=False
            )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
