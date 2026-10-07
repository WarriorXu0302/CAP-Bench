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
TASK_ID = "task-cc9bf0"
TASK_DESCRIPTION = "I am planning to attend the 2026 F1 Singapore Grand Prix in person and need a budget plan covering flights and accommodation.\n\nFirst, go to **ESPN** to confirm the exact date of the 2026 F1 Singapore Grand Prix Sunday Race.\n\nOnce the date is obtained, go to **Skyscanner** to search for flights from London (LHR) to Singapore. Set the travel dates as 'arrival on Thursday, departure the following Monday' (based on the race week). Filter for the cheapest flight with no more than 1 layover.\n\nNext, use **Google Maps** to address accommodation. Search for 'hostels' near the 'Marina Bay Street Circuit' and identify three hostels with a rating of 4.0 or higher.\n\nFinally, use the map's route planning function to calculate the walking time from each of these three hostels to 'Singapore Flyer' (a nearby landmark).\n\n**Output:** The date of the 2026 F1 Singapore Grand Prix Sunday Race, the airline/price/number of layovers for the preferred Skyscanner flight, the names/ratings/walking times for the three chosen hostels, the link to the Skyscanner search results page, and a link to one of the Google Maps routes."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RaceDate(BaseModel):
    """Race date extracted from the answer for 2026 F1 Singapore Grand Prix"""
    race_date: Optional[str] = None


class FlightInfo(BaseModel):
    """Flight information extracted from the answer"""
    airline: Optional[str] = None
    price: Optional[str] = None
    layovers: Optional[str] = None
    skyscanner_link: Optional[str] = None


class HostelInfo(BaseModel):
    """Hostel information extracted from the answer"""
    hostel_names: Optional[List[str]] = Field(default_factory=list)
    hostel_ratings: Optional[List[str]] = Field(default_factory=list)
    walking_times: Optional[List[str]] = Field(default_factory=list)
    maps_route_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_race_date() -> str:
    return """
Extract the date of the 2026 F1 Singapore Grand Prix Sunday Race from the answer.

Return:
- race_date: the exact date as stated (e.g., "September 27, 2026" or "27/09/2026"). If not present, set null.
"""


def prompt_extract_flight_info() -> str:
    return """
Extract the flight information from the answer for the Skyscanner search results.

Return:
- airline: the airline name for the preferred flight
- price: the price of the flight exactly as stated (include currency)
- layovers: the number of layovers or stops (e.g., "0", "1", "direct", "1 stop")
- skyscanner_link: the URL link to the Skyscanner search results page

If any field is missing, set it to null.
"""


def prompt_extract_hostel_info() -> str:
    return """
Extract the hostel information from the answer for the three hostels near Marina Bay Street Circuit.

Return:
- hostel_names: a list of the three hostel names
- hostel_ratings: a list of the three hostel ratings (as strings, e.g., "4.2", "4.5")
- walking_times: a list of the three walking times from each hostel to Singapore Flyer
- maps_route_link: a URL link to one of the Google Maps routes

If any field is missing or incomplete, set appropriate defaults (empty list or null).
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
    m = re.search(r'\b(2026)\b', text)
    if m:
        return int(m.group(1))
    return None


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats containing 2026
    return extract_year(text) == 2026 and contains_digits(text)


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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for currency symbols or common currency codes
    has_currency = has_any_ci(text, ['$', '£', '€', 'usd', 'gbp', 'eur', 'sgd'])
    has_number = contains_digits(text)
    return has_currency and has_number


def looks_like_layover_count(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept "0", "1", "direct", "1 stop", "1 layover", "non-stop"
    patterns = [r'\b0\b', r'\b1\b', r'\bdirect\b', r'\bnon-stop\b', r'\b1\s*stop\b', r'\b1\s*layover\b']
    return any(re.search(p, text.lower()) for p in patterns)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    return rating is not None and 4.0 <= rating <= 5.0


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept time mentions with minutes or hours
    has_time_unit = has_any_ci(text, ['min', 'minute', 'hour', 'hr'])
    has_number = contains_digits(text)
    return has_time_unit and has_number


def is_valid_url(text: Optional[str]) -> bool:
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
    race_date_info = await evaluator.extract(
        prompt=prompt_extract_race_date(),
        template_class=RaceDate,
        extraction_name="race_date_info"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_info(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )

    hostel_info = await evaluator.extract(
        prompt=prompt_extract_hostel_info(),
        template_class=HostelInfo,
        extraction_name="hostel_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 ESPN section
    espn_node = evaluator.add_sequential(
        id="espn_section",
        desc="ESPN - Confirm 2026 F1 Singapore Grand Prix Sunday Race date",
        parent=root,
        critical=False
    )

    # [Action Node] espn.com:F5:A2 - Multi-condition filtering
    espn_filter_ok = (has_any_ci(answer, ['espn']) and
                      has_any_ci(answer, ['f1', 'formula 1', 'formula one']) and
                      has_any_ci(answer, ['2026']) and
                      has_any_ci(answer, ['singapore']))
    evaluator.add_custom_node(
        result=bool(espn_filter_ok),
        id="espn_filter_f1_2026",
        desc="[Action Node] espn.com:F5:A2 - Filter for F1 league and 2026 season Singapore race",
        parent=espn_node,
        critical=False
    )

    # [Perception Node] espn.com:F5:P12 - Schedule information recognition
    date_text_ok = looks_like_date(race_date_info.race_date)
    schedule_mention_ok = has_any_ci(answer, ['schedule', 'race date', 'sunday', 'grand prix'])
    evaluator.add_custom_node(
        result=bool(date_text_ok and schedule_mention_ok),
        id="espn_extract_race_date",
        desc="[Perception Node] espn.com:F5:P12 - Extract exact date of Sunday Race from schedule",
        parent=espn_node,
        critical=False
    )

    # 3.2 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner - Search for flights from London to Singapore",
        parent=root,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A2 - Date selection
    skyscanner_mention_ok = has_any_ci(answer, ['skyscanner'])
    date_logic_ok = has_any_ci(answer, ['thursday', 'monday']) or has_any_ci(answer, ['race week'])
    evaluator.add_custom_node(
        result=bool(skyscanner_mention_ok and date_logic_ok),
        id="skyscanner_date_selection",
        desc="[Action Node] skyscanner.com:F1:A2 - Select arrival Thursday, departure Monday based on race date",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A6 - Multi-condition filter
    layover_filter_ok = looks_like_layover_count(flight_info.layovers)
    evaluator.add_custom_node(
        result=bool(layover_filter_ok),
        id="skyscanner_layover_filter",
        desc="[Action Node] skyscanner.com:F1:A6 - Filter for flights with no more than 1 layover",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A7 - Sort operation
    cheapest_mention_ok = has_any_ci(answer, ['cheapest', 'cheap', 'lowest price', 'best price'])
    evaluator.add_custom_node(
        result=bool(cheapest_mention_ok),
        id="skyscanner_sort_cheapest",
        desc="[Action Node] skyscanner.com:F1:A7 - Sort by cheapest price",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F6:P1 - Price awareness
    price_ok = looks_like_price(flight_info.price)
    evaluator.add_custom_node(
        result=bool(price_ok),
        id="skyscanner_extract_price",
        desc="[Perception Node] skyscanner.com:F6:P1 - Extract flight price with currency",
        parent=skyscanner_node,
        critical=False
    )

    # Check for airline and Skyscanner link
    airline_ok = bool(flight_info.airline and flight_info.airline.strip())
    evaluator.add_custom_node(
        result=bool(airline_ok),
        id="skyscanner_airline_name",
        desc="Extract airline name for the preferred flight",
        parent=skyscanner_node,
        critical=False
    )

    skyscanner_link_ok = is_valid_url(flight_info.skyscanner_link)
    evaluator.add_custom_node(
        result=bool(skyscanner_link_ok),
        id="skyscanner_link_provided",
        desc="Provide link to Skyscanner search results page",
        parent=skyscanner_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps - Find hostels near Marina Bay Street Circuit and calculate walking times",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search
    maps_search_ok = (has_any_ci(answer, ['google maps', 'maps.google']) and
                      has_any_ci(answer, ['hostel', 'hostels']) and
                      has_any_ci(answer, ['marina bay', 'street circuit']))
    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="maps_search_hostels",
        desc="[Action Node] maps.google.com:F1:A2 - Search for hostels near Marina Bay Street Circuit",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Rating awareness
    three_hostels_ok = len(hostel_info.hostel_names) >= 3
    ratings_ok = all(looks_like_rating(r) for r in hostel_info.hostel_ratings) if hostel_info.hostel_ratings else False
    all_ratings_above_4 = all(extract_float(r) >= 4.0 for r in hostel_info.hostel_ratings if extract_float(r) is not None) if hostel_info.hostel_ratings else False

    evaluator.add_custom_node(
        result=bool(three_hostels_ok and ratings_ok and all_ratings_above_4),
        id="maps_filter_rating",
        desc="[Perception Node] maps.google.com:F1:P1 - Identify three hostels with rating >= 4.0",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection
    walking_mode_ok = has_any_ci(answer, ['walk', 'walking', 'on foot'])
    evaluator.add_custom_node(
        result=bool(walking_mode_ok),
        id="maps_select_walking",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking mode for route calculation",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    route_planning_ok = has_any_ci(answer, ['singapore flyer'])
    evaluator.add_custom_node(
        result=bool(route_planning_ok),
        id="maps_route_to_flyer",
        desc="[Action Node] maps.google.com:F2:A5 - Calculate routes from hostels to Singapore Flyer",
        parent=maps_node,
        critical=False
    )

    # Check walking times extracted
    walking_times_ok = (len(hostel_info.walking_times) >= 3 and
                        all(looks_like_walking_time(wt) for wt in hostel_info.walking_times))
    evaluator.add_custom_node(
        result=bool(walking_times_ok),
        id="maps_extract_walking_times",
        desc="Extract walking times from each hostel to Singapore Flyer",
        parent=maps_node,
        critical=False
    )

    # Check Google Maps route link provided
    maps_link_ok = is_valid_url(hostel_info.maps_route_link)
    evaluator.add_custom_node(
        result=bool(maps_link_ok),
        id="maps_route_link_provided",
        desc="Provide link to one of the Google Maps routes",
        parent=maps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
