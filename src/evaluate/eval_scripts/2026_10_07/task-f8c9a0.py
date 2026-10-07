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
TASK_ID = "task-f8c9a0"
TASK_DESCRIPTION = "As an architecture student, I am currently writing my final paper on the Solomon R. Guggenheim Museum, designed by Frank Lloyd Wright, focusing on the controversies and current status of its 'spiral ramp' design.\n\nFirst, navigate to Wikipedia and search for the museum. Utilize the table of contents to quickly locate the 'Architecture' or 'Criticism' sections. Record its opening year and the specific criticisms made by early critics regarding 'displaying artworks on sloped walls'.\n\nNext, search for the museum on TripAdvisor. Review the 'AI Review Summary' for descriptions related to 'Atmosphere' or 'Layout'.\n\nThen, sort the reviews by 'Newest first'. Within the most recent year of reviews available, read through and filter for 3 reviews that mention 'slope' or 'dizzy'. Record the reviewers' ratings (1-5 stars) and the specific content of their comments.\n\nFinally, for a hypothetical site visit, use Google Maps to calculate the walking distance (in miles) and estimated time from The Metropolitan Museum of Art (The Met) to the Guggenheim Museum.\n\nOutput: Opening year, Wikipedia criticism summary, TripAdvisor AI summary content, ratings and content of the 3 selected reviews, walking distance between the two museums, walking time, and detailed page links for each website visited."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WikipediaInfo(BaseModel):
    """Information extracted from Wikipedia about the Guggenheim Museum"""
    opening_year: Optional[str] = None
    criticism_summary: Optional[str] = None
    wikipedia_url: Optional[str] = None


class TripAdvisorAISummary(BaseModel):
    """AI Review Summary extracted from TripAdvisor"""
    ai_summary_content: Optional[str] = None
    tripadvisor_url: Optional[str] = None


class ReviewInfo(BaseModel):
    """Individual review information"""
    rating: Optional[int] = None
    content: Optional[str] = None
    date: Optional[str] = None


class TripAdvisorReviews(BaseModel):
    """Collection of reviews mentioning slope or dizzy"""
    review_1: Optional[ReviewInfo] = None
    review_2: Optional[ReviewInfo] = None
    review_3: Optional[ReviewInfo] = None


class GoogleMapsInfo(BaseModel):
    """Distance and time information from Google Maps"""
    walking_distance_miles: Optional[str] = None
    walking_time: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_wikipedia_info() -> str:
    return """
Extract the following information about the Solomon R. Guggenheim Museum from the answer:

- opening_year: the year the museum opened (e.g., "1959", "October 1959"). If not present, set null.
- criticism_summary: the specific criticisms made by early critics regarding displaying artworks on sloped walls, exactly as described in the answer. If not present, set null.
- wikipedia_url: the Wikipedia page URL mentioned or referenced in the answer. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_tripadvisor_ai_summary() -> str:
    return """
Extract the TripAdvisor AI Review Summary information from the answer:

- ai_summary_content: the content from the AI Review Summary related to 'Atmosphere' or 'Layout', exactly as described in the answer. If not present, set null.
- tripadvisor_url: the TripAdvisor page URL mentioned or referenced in the answer. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_tripadvisor_reviews() -> str:
    return """
Extract the 3 reviews that mention 'slope' or 'dizzy' from the answer. For each review, extract:

- rating: the star rating (1-5) as an integer. If not present, set null.
- content: the specific content of the comment, exactly as written. If not present, set null.
- date: the date of the review if mentioned. If not present, set null.

Return:
- review_1: the first review (with rating, content, date)
- review_2: the second review (with rating, content, date)
- review_3: the third review (with rating, content, date)

If fewer than 3 reviews are present, set the missing ones to null.
"""


def prompt_extract_google_maps_info() -> str:
    return """
Extract the Google Maps walking route information from the answer between The Met and the Guggenheim Museum:

- walking_distance_miles: the walking distance in miles, exactly as stated. If not present, set null.
- walking_time: the estimated walking time, exactly as stated. If not present, set null.

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


def extract_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r'\b(19\d{2}|20\d{2})\b', text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_wikipedia_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return has_any_ci(url, ['wikipedia.org'])


def looks_like_tripadvisor_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return has_any_ci(url, ['tripadvisor.com'])


def is_valid_rating(rating: Optional[int]) -> bool:
    if rating is None:
        return False
    return 1 <= rating <= 5


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi'])


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour'])


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


def is_recent_review(date_text: Optional[str]) -> bool:
    """Check if the review is from recent years (2024-2025)"""
    if not date_text:
        return True  # Lenient: if no date, assume it could be recent
    year = extract_year(date_text)
    if year is None:
        return True  # Lenient
    return year >= 2024


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
    wikipedia_info = await evaluator.extract(
        prompt=prompt_extract_wikipedia_info(),
        template_class=WikipediaInfo,
        extraction_name="wikipedia_info"
    )

    tripadvisor_ai = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_ai_summary(),
        template_class=TripAdvisorAISummary,
        extraction_name="tripadvisor_ai_summary"
    )

    reviews_info = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_reviews(),
        template_class=TripAdvisorReviews,
        extraction_name="tripadvisor_reviews"
    )

    maps_info = await evaluator.extract(
        prompt=prompt_extract_google_maps_info(),
        template_class=GoogleMapsInfo,
        extraction_name="google_maps_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia research on Guggenheim Museum architecture and criticism",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia article
    wiki_nav_ok = (has_any_ci(answer, ['wikipedia']) and
                   has_any_ci(answer, ['guggenheim']))
    wiki_url_ok = looks_like_wikipedia_url(wikipedia_info.wikipedia_url) if wikipedia_info else False

    evaluator.add_custom_node(
        result=bool(wiki_nav_ok and wiki_url_ok),
        id="wikipedia_navigate_article",
        desc="[Action Node] wikipedia.org:F1:A20 - Navigate to the Guggenheim Museum Wikipedia article",
        parent=wikipedia_node,
        critical=False
    )

    # [Action Node] wikipedia.org:F4:A19 - Use table of contents to jump to section
    toc_usage_ok = has_any_ci(answer, ['table of contents', 'toc', 'architecture', 'criticism'])

    evaluator.add_custom_node(
        result=bool(toc_usage_ok),
        id="wikipedia_toc_navigation",
        desc="[Action Node] wikipedia.org:F4:A19 - Use table of contents to locate Architecture or Criticism sections",
        parent=wikipedia_node,
        critical=False
    )

    # Extract opening year
    year = extract_year(wikipedia_info.opening_year) if wikipedia_info else None
    year_ok = year is not None and 1950 <= year <= 1970  # Guggenheim opened in 1959

    evaluator.add_custom_node(
        result=bool(year_ok),
        id="wikipedia_opening_year",
        desc="Extract the museum's opening year from Wikipedia",
        parent=wikipedia_node,
        critical=False
    )

    # Extract criticism about sloped walls
    criticism_ok = (wikipedia_info and
                   wikipedia_info.criticism_summary and
                   len(wikipedia_info.criticism_summary.strip()) > 20 and
                   has_any_ci(wikipedia_info.criticism_summary, ['slope', 'wall', 'display', 'artwork', 'art']))

    evaluator.add_custom_node(
        result=bool(criticism_ok),
        id="wikipedia_criticism_content",
        desc="Extract specific criticisms about displaying artworks on sloped walls",
        parent=wikipedia_node,
        critical=False
    )

    # 3.2 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor research on current visitor experiences",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A1 - Search for museum on TripAdvisor
    ta_search_ok = (has_any_ci(answer, ['tripadvisor']) and
                    has_any_ci(answer, ['guggenheim']))
    ta_url_ok = looks_like_tripadvisor_url(tripadvisor_ai.tripadvisor_url) if tripadvisor_ai else False

    evaluator.add_custom_node(
        result=bool(ta_search_ok and ta_url_ok),
        id="tripadvisor_search_museum",
        desc="[Action Node] tripadvisor.com:F4:A1 - Search for and navigate to Guggenheim Museum on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P7 - AI Review Summary
    ai_summary_ok = (tripadvisor_ai and
                     tripadvisor_ai.ai_summary_content and
                     len(tripadvisor_ai.ai_summary_content.strip()) > 20 and
                     has_any_ci(tripadvisor_ai.ai_summary_content, ['atmosphere', 'layout', 'spiral', 'ramp', 'architecture']))

    evaluator.add_custom_node(
        result=bool(ai_summary_ok),
        id="tripadvisor_ai_summary",
        desc="[Perception Node] tripadvisor.com:F2:P7 - Extract AI Review Summary content about Atmosphere or Layout",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A5 - Sort reviews by newest first
    sort_ok = has_any_ci(answer, ['newest', 'sort', 'recent'])

    evaluator.add_custom_node(
        result=bool(sort_ok),
        id="tripadvisor_sort_reviews",
        desc="[Action Node] tripadvisor.com:F4:A5 - Sort reviews by 'Newest first'",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F2:A10 - Expand reviews to read full content
    expand_ok = has_any_ci(answer, ['read', 'expand', 'filter', 'review'])

    evaluator.add_custom_node(
        result=bool(expand_ok),
        id="tripadvisor_expand_reviews",
        desc="[Action Node] tripadvisor.com:F2:A10 - Expand and read through reviews",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P1 - Extract ratings from reviews
    reviews = [reviews_info.review_1, reviews_info.review_2, reviews_info.review_3] if reviews_info else [None, None, None]
    valid_reviews = [r for r in reviews if r is not None]

    ratings_ok = len(valid_reviews) >= 3 and all(is_valid_rating(r.rating) for r in valid_reviews[:3])

    evaluator.add_custom_node(
        result=bool(ratings_ok),
        id="tripadvisor_extract_ratings",
        desc="[Perception Node] tripadvisor.com:F2:P1 - Extract star ratings (1-5) from the 3 selected reviews",
        parent=tripadvisor_node,
        critical=False
    )

    # Check reviews mention slope or dizzy
    reviews_content_ok = (len(valid_reviews) >= 3 and
                         all(r.content and len(r.content.strip()) > 10 for r in valid_reviews[:3]) and
                         all(has_any_ci(r.content, ['slope', 'dizzy', 'ramp', 'spiral']) for r in valid_reviews[:3]))

    evaluator.add_custom_node(
        result=bool(reviews_content_ok),
        id="tripadvisor_reviews_content",
        desc="Extract 3 reviews mentioning 'slope' or 'dizzy' with their content",
        parent=tripadvisor_node,
        critical=False
    )

    # Check reviews are recent (2024-2025)
    reviews_recent_ok = len(valid_reviews) >= 3 and all(is_recent_review(r.date) for r in valid_reviews[:3])

    evaluator.add_custom_node(
        result=bool(reviews_recent_ok),
        id="tripadvisor_reviews_recent",
        desc="Reviews are from the most recent year available (2024-2025)",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps walking route between Met and Guggenheim",
        parent=root,
        critical=False
    )

    # [Action Node] google.com:maps:A5 - Enter route planning input
    maps_route_ok = (has_any_ci(answer, ['google maps', 'maps']) and
                     has_any_ci(answer, ['met', 'metropolitan']) and
                     has_any_ci(answer, ['guggenheim']))

    evaluator.add_custom_node(
        result=bool(maps_route_ok),
        id="google_maps_route_input",
        desc="[Action Node] google.com:maps:A5 - Input route from The Met to Guggenheim Museum",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] google.com:maps:P3 - Extract distance and time
    distance_ok = looks_like_distance(maps_info.walking_distance_miles) if maps_info else False
    distance_val = extract_float(maps_info.walking_distance_miles) if maps_info else None
    distance_reasonable = distance_val is not None and 0.3 <= distance_val <= 1.0  # Expected ~0.5 miles

    evaluator.add_custom_node(
        result=bool(distance_ok and distance_reasonable),
        id="google_maps_distance",
        desc="[Perception Node] google.com:maps:P3 - Extract walking distance in miles (expected ~0.4-0.6 miles)",
        parent=maps_node,
        critical=False
    )

    time_ok = looks_like_time(maps_info.walking_time) if maps_info else False
    time_val = extract_float(maps_info.walking_time) if maps_info else None
    time_reasonable = time_val is not None and 5 <= time_val <= 20  # Expected ~8-15 minutes

    evaluator.add_custom_node(
        result=bool(time_ok and time_reasonable),
        id="google_maps_time",
        desc="[Perception Node] google.com:maps:P3 - Extract walking time (expected ~8-15 minutes)",
        parent=maps_node,
        critical=False
    )

    # 3.4 Overall completeness checks (non-prefixed)
    all_urls_present = (looks_like_wikipedia_url(wikipedia_info.wikipedia_url if wikipedia_info else None) and
                       looks_like_tripadvisor_url(tripadvisor_ai.tripadvisor_url if tripadvisor_ai else None))

    evaluator.add_custom_node(
        result=bool(all_urls_present),
        id="all_page_links_provided",
        desc="Detailed page links for all visited websites are provided",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
