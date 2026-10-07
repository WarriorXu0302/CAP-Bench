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
TASK_ID = "task-4d3b21"
TASK_DESCRIPTION = "I am a Manchester United fan, planning a trip to Newcastle for an away game.\nPlease help me plan my itinerary for December 2025.\nFirst, go to ESPN to find the Manchester United away match against Newcastle United, and confirm the exact date and kick-off time.\nNext, go to TripAdvisor to find accommodation for the night of the match. Filter for 3 hotels in Newcastle upon Tyne with a rating of 4 stars or higher, and explicitly stating 'breakfast included'.\nFinally, to ensure easy access to the stadium, use Google Maps to calculate the walking route from each of these 3 hotels to St. James' Park. Only retain hotels with a walking time of 15 minutes or less (if all exceed 15 minutes, retain the one with the shortest walking time).\nOutput the match date and time, the names of the 3 hotels, their TripAdvisor ratings, price per night, whether breakfast is included, and the specific walking duration shown on Google Maps. Also, include links to the match details page and each hotel's details page."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MatchInfo(BaseModel):
    """Match details extracted from ESPN"""
    match_date: Optional[str] = None
    match_time: Optional[str] = None
    match_link: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel information from TripAdvisor and Google Maps"""
    hotel_names: Optional[List[str]] = Field(default_factory=list)
    ratings: Optional[List[str]] = Field(default_factory=list)
    prices: Optional[List[str]] = Field(default_factory=list)
    breakfast_included: Optional[List[str]] = Field(default_factory=list)
    walking_times: Optional[List[str]] = Field(default_factory=list)
    hotel_links: Optional[List[str]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_match_from_answer() -> str:
    return """
Extract the Manchester United away match information against Newcastle United from the answer.

Return:
- match_date: the exact date of the match as stated (e.g., "December 14, 2025", "14 Dec 2025"). If not present, set null.
- match_time: the kick-off time as stated (e.g., "3:00 PM", "15:00"). If not present, set null.
- match_link: the ESPN match details page URL if provided. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_hotels_from_answer() -> str:
    return """
From the answer, extract the list of hotels found on TripAdvisor and their details:

- hotel_names: list of hotel names (up to 3)
- ratings: list of ratings for each hotel (in same order)
- prices: list of prices per night for each hotel (in same order)
- breakfast_included: list of whether breakfast is included for each hotel (in same order)
- walking_times: list of walking times from each hotel to St. James' Park from Google Maps (in same order)
- hotel_links: list of TripAdvisor detail page URLs for each hotel (in same order)

Return empty lists if information is not present.
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


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    has_month = has_any_ci(text, ['january', 'february', 'march', 'april', 'may', 'june',
                                   'july', 'august', 'september', 'october', 'november', 'december',
                                   'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    has_number = contains_digits(text)
    return has_month and has_number


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'\d{1,2}:\d{2}',
        r'\d{1,2}\s*(am|pm|a\.m\.|p\.m\.)',
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def is_rating_4_or_higher(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num >= 4.0


def mentions_breakfast(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['breakfast', 'breakfast included', 'yes'])


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_unit = has_any_ci(text, ['min', 'minute', 'minutes'])
    return has_number and has_unit


def extract_walking_minutes(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:min|minute|minutes)', text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def is_within_15_minutes(text: Optional[str]) -> bool:
    mins = extract_walking_minutes(text)
    if mins is None:
        return False
    return mins <= 15


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


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
    match_info = await evaluator.extract(
        prompt=prompt_extract_match_from_answer(),
        template_class=MatchInfo,
        extraction_name="match_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotels_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 ESPN match information section
    espn_node = evaluator.add_sequential(
        id="espn_section",
        desc="ESPN match information for Manchester United away at Newcastle United",
        parent=root,
        critical=False
    )

    # [Action Node] espn.com:F5:A2 - Multi-condition filtering
    espn_filtering_ok = (has_any_ci(answer, ['espn']) and
                         has_any_ci(answer, ['manchester united', 'man united', 'man utd']) and
                         has_any_ci(answer, ['newcastle', 'newcastle united']) and
                         has_any_ci(answer, ['december', 'dec']))
    evaluator.add_custom_node(
        result=bool(espn_filtering_ok),
        id="espn_action_filtering",
        desc="[Action Node] espn.com:F5:A2 - Filter matches by teams (Manchester United away at Newcastle) and month (December 2025)",
        parent=espn_node,
        critical=False
    )

    # [Perception Node] espn.com:F5:P12 - Match date and time identification
    date_ok = looks_like_date(match_info.match_date)
    time_ok = looks_like_time(match_info.match_time)
    evaluator.add_custom_node(
        result=bool(date_ok and time_ok),
        id="espn_perception_match_datetime",
        desc="[Perception Node] espn.com:F5:P12 - Extract exact match date and kick-off time",
        parent=espn_node,
        critical=False
    )

    # Match link provided
    match_link_ok = looks_like_url(match_info.match_link)
    evaluator.add_custom_node(
        result=bool(match_link_ok),
        id="espn_match_link",
        desc="Match details page link is provided",
        parent=espn_node,
        critical=False
    )

    # 3.2 TripAdvisor hotel search section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor hotel search for Newcastle upon Tyne",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A2 - Date range selection
    tripadvisor_date_ok = (has_any_ci(answer, ['tripadvisor']) and
                           date_ok)  # Uses date from ESPN
    evaluator.add_custom_node(
        result=bool(tripadvisor_date_ok),
        id="tripadvisor_action_date",
        desc="[Action Node] tripadvisor.com:F1:A2 - Use match date from ESPN to search for accommodation on the night of the match",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-condition filtering
    has_3_hotels = hotel_info.hotel_names and len(hotel_info.hotel_names) >= 3
    all_ratings_ok = all(is_rating_4_or_higher(r) for r in (hotel_info.ratings or []) if r)
    all_breakfast_ok = all(mentions_breakfast(b) for b in (hotel_info.breakfast_included or []) if b)
    tripadvisor_filtering_ok = has_3_hotels and all_ratings_ok and all_breakfast_ok
    evaluator.add_custom_node(
        result=bool(tripadvisor_filtering_ok),
        id="tripadvisor_action_filtering",
        desc="[Action Node] tripadvisor.com:F1:A4 - Filter for hotels with rating 4+ and breakfast included, return 3 hotels",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F2:P1 - Rating identification
    ratings_present = hotel_info.ratings and len(hotel_info.ratings) >= 3
    all_valid_ratings = all(looks_like_rating(r) for r in (hotel_info.ratings or []) if r)
    evaluator.add_custom_node(
        result=bool(ratings_present and all_valid_ratings),
        id="tripadvisor_perception_rating",
        desc="[Perception Node] tripadvisor.com:F2:P1 - Extract rating values for each hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F1:P10 - Breakfast information identification
    breakfast_present = hotel_info.breakfast_included and len(hotel_info.breakfast_included) >= 3
    evaluator.add_custom_node(
        result=bool(breakfast_present),
        id="tripadvisor_perception_breakfast",
        desc="[Perception Node] tripadvisor.com:F1:P10 - Extract breakfast included status for each hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # Price information provided
    prices_present = hotel_info.prices and len(hotel_info.prices) >= 3
    evaluator.add_custom_node(
        result=bool(prices_present),
        id="tripadvisor_prices",
        desc="Price per night is provided for each hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # Hotel links provided
    links_present = hotel_info.hotel_links and len(hotel_info.hotel_links) >= 3
    all_valid_links = all(looks_like_url(link) for link in (hotel_info.hotel_links or []) if link)
    evaluator.add_custom_node(
        result=bool(links_present and all_valid_links),
        id="tripadvisor_hotel_links",
        desc="Hotel details page links are provided for each hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.3 Google Maps walking route section
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps walking route calculation from hotels to St. James' Park",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    googlemaps_route_ok = (has_any_ci(answer, ['google maps']) and
                           has_any_ci(answer, ['st. james', 'st james', 'stadium']) and
                           has_3_hotels)
    evaluator.add_custom_node(
        result=bool(googlemaps_route_ok),
        id="googlemaps_action_route",
        desc="[Action Node] maps.google.com:F2:A5 - Calculate walking routes from each of the 3 hotels to St. James' Park",
        parent=googlemaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Walking mode selection
    walking_times_present = hotel_info.walking_times and len(hotel_info.walking_times) >= 3
    all_valid_walking = all(looks_like_walking_time(w) for w in (hotel_info.walking_times or []) if w)
    evaluator.add_custom_node(
        result=bool(walking_times_present and all_valid_walking),
        id="googlemaps_action_walking",
        desc="[Action Node] maps.google.com:F2:A7 - Use walking mode to calculate route times",
        parent=googlemaps_node,
        critical=False
    )

    # Walking time filtering logic
    walking_filter_ok = False
    if hotel_info.walking_times:
        any_within_15 = any(is_within_15_minutes(w) for w in hotel_info.walking_times if w)
        if any_within_15:
            walking_filter_ok = True
        else:
            # If none within 15 min, should retain shortest
            walking_filter_ok = len(hotel_info.walking_times) >= 1
    evaluator.add_custom_node(
        result=bool(walking_filter_ok),
        id="googlemaps_walking_filter",
        desc="Hotels are filtered based on walking time criteria (15 minutes or less, or shortest if all exceed 15 minutes)",
        parent=googlemaps_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall completeness of the answer",
        parent=root,
        critical=False
    )

    # All required outputs present
    has_match_datetime = date_ok and time_ok
    has_hotels = has_3_hotels
    has_ratings_output = ratings_present
    has_prices_output = prices_present
    has_breakfast_output = breakfast_present
    has_walking_output = walking_times_present
    all_outputs_ok = (has_match_datetime and has_hotels and has_ratings_output and
                      has_prices_output and has_breakfast_output and has_walking_output)
    evaluator.add_custom_node(
        result=bool(all_outputs_ok),
        id="completeness_all_outputs",
        desc="All required information is provided (match date/time, hotel names, ratings, prices, breakfast status, walking times)",
        parent=completeness_node,
        critical=False
    )

    # Information flow verification
    flow_ok = tripadvisor_date_ok and googlemaps_route_ok
    evaluator.add_custom_node(
        result=bool(flow_ok),
        id="completeness_info_flow",
        desc="Information flows correctly from ESPN date to TripAdvisor search to Google Maps route calculation",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
