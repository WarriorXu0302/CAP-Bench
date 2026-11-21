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
TASK_ID = "task-818fa1"
TASK_DESCRIPTION = 'I am planning a trip to Italy in May 2026, themed around visiting filming locations from the movie "Mission: Impossible - Dead Reckoning Part One". First, go to IMDb to find this 2023 movie and identify its main filming cities. From these, find two famous Italian tourist cities (Hint: one is the capital, and one is a \'water city\').\n\nFor the capital city (Rome), search on Google Flights for a one-way ticket from New York (JFK). The planned departure date is May 15, 2026. Use the "Date Grid" feature to check prices for surrounding dates. If May 14th or 16th is more than $50 cheaper than May 15th, select the cheaper date; otherwise, keep May 15th. Filter for the *only* non-stop flight available.\n\nFor the \'water city\' (Venice), find a hotel on Tripadvisor. Requirements: It must have the "Travelers\' Choice" badge, be a 4-star hotel, and have a nightly rate between $300-$600. Sort by "Traveler Ranked" and select the top-ranked hotel.\n\nOutput: The two city names found on IMDb, the selected departure date from Google Flights, the lowest non-stop flight price, the airline, the selected hotel name from Tripadvisor, its nightly rate, whether it has the Travelers\' Choice badge, and the respective detail page links for all three websites.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IMDbCities(BaseModel):
    """Filming location cities extracted from the answer"""
    city1: Optional[str] = None
    city2: Optional[str] = None
    imdb_url: Optional[str] = None


class GoogleFlightsInfo(BaseModel):
    """Flight information extracted from the answer"""
    departure_date: Optional[str] = None
    flight_price: Optional[str] = None
    airline: Optional[str] = None
    flights_url: Optional[str] = None


class TripadvisorHotelInfo(BaseModel):
    """Hotel information extracted from the answer"""
    hotel_name: Optional[str] = None
    nightly_rate: Optional[str] = None
    has_travelers_choice: Optional[bool] = None
    tripadvisor_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_imdb_cities() -> str:
    return """
Extract the two Italian filming location cities from the answer that the user found on IMDb for "Mission: Impossible - Dead Reckoning Part One".

Return:
- city1: the first city name (likely the capital)
- city2: the second city name (likely the water city)
- imdb_url: the IMDb detail page URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_google_flights() -> str:
    return """
Extract the Google Flights information from the answer for the flight from New York (JFK) to Rome.

Return:
- departure_date: the selected departure date (e.g., "May 15, 2026" or "2026-05-15")
- flight_price: the lowest non-stop flight price exactly as stated (include currency if present)
- airline: the airline name
- flights_url: the Google Flights detail/search page URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_tripadvisor_hotel() -> str:
    return """
Extract the Tripadvisor hotel information from the answer for Venice.

Return:
- hotel_name: the name of the selected hotel
- nightly_rate: the nightly rate exactly as stated (include currency if present)
- has_travelers_choice: true if the answer mentions the hotel has the "Travelers' Choice" badge, false otherwise
- tripadvisor_url: the Tripadvisor hotel detail page URL if provided

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
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower() or text.startswith('www.'))


def is_rome_or_similar(city: Optional[str]) -> bool:
    if not city:
        return False
    return ci_contains(city, 'rome') or ci_contains(city, 'roma')


def is_venice_or_similar(city: Optional[str]) -> bool:
    if not city:
        return False
    return ci_contains(city, 'venice') or ci_contains(city, 'venezia')


def is_may_date_2026(date_text: Optional[str]) -> bool:
    if not date_text:
        return False
    has_may = ci_contains(date_text, 'may')
    has_2026 = '2026' in date_text
    has_day = bool(re.search(r'\b(1[4-6]|14|15|16)\b', date_text))
    return has_may and has_2026 and has_day


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '$') or ci_contains(text, 'usd') or ci_contains(text, 'dollar'))


def is_price_in_range(price_text: Optional[str], min_val: float, max_val: float) -> bool:
    price = extract_float(price_text)
    if price is None:
        return False
    return min_val <= price <= max_val


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
    imdb_info = await evaluator.extract(
        prompt=prompt_extract_imdb_cities(),
        template_class=IMDbCities,
        extraction_name="imdb_cities"
    )

    flights_info = await evaluator.extract(
        prompt=prompt_extract_google_flights(),
        template_class=GoogleFlightsInfo,
        extraction_name="google_flights_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_hotel(),
        template_class=TripadvisorHotelInfo,
        extraction_name="tripadvisor_hotel_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb - Find filming locations for Mission: Impossible - Dead Reckoning Part One (2023)",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F3:A26 - Click card to enter details
    imdb_search_ok = has_any_ci(answer, ['imdb']) and has_any_ci(answer, ['mission: impossible', 'dead reckoning'])
    evaluator.add_custom_node(
        result=bool(imdb_search_ok),
        id="imdb_action_search_and_click",
        desc="[Action Node] imdb.com:F3:A26 - Search for and click on the movie card to enter detail page",
        parent=imdb_node,
        critical=False
    )

    # [Perception Node] imdb.com:F3:P1 - Image understanding (identify correct 2023 movie)
    has_2023 = '2023' in answer
    has_movie_title = has_any_ci(answer, ['mission: impossible', 'dead reckoning'])
    imdb_url_ok = looks_like_url(imdb_info.imdb_url, 'imdb.com')

    evaluator.add_custom_node(
        result=bool(has_2023 and has_movie_title and imdb_url_ok),
        id="imdb_perception_correct_movie",
        desc="[Perception Node] imdb.com:F3:P1 - Confirm correct 2023 movie via poster/year recognition and provide detail page URL",
        parent=imdb_node,
        critical=False
    )

    # Check if cities are Rome and Venice
    city1_is_rome = is_rome_or_similar(imdb_info.city1)
    city2_is_venice = is_venice_or_similar(imdb_info.city2)
    city1_is_venice = is_venice_or_similar(imdb_info.city1)
    city2_is_rome = is_rome_or_similar(imdb_info.city2)

    cities_correct = (city1_is_rome and city2_is_venice) or (city1_is_venice and city2_is_rome)

    evaluator.add_custom_node(
        result=bool(cities_correct),
        id="imdb_cities_identified",
        desc="Correctly identified Rome and Venice as the two Italian filming location cities",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights - Search for JFK to Rome flight on May 15, 2026 (or adjusted date)",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - Date range selection
    flights_search_ok = has_any_ci(answer, ['google flights']) and has_any_ci(answer, ['jfk', 'new york']) and has_any_ci(answer, ['rome'])
    date_mentioned = is_may_date_2026(flights_info.departure_date)

    evaluator.add_custom_node(
        result=bool(flights_search_ok and date_mentioned),
        id="flights_action_date_selection",
        desc="[Action Node] google.comflights:F1:A2 - Select May 15, 2026 (or nearby date) as departure date",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F4:A12 - Switch to Date Grid view
    date_grid_mentioned = has_any_ci(answer, ['date grid']) or has_any_ci(answer, ['grid view']) or has_any_ci(answer, ['surrounding dates', 'may 14', 'may 16'])

    evaluator.add_custom_node(
        result=bool(date_grid_mentioned),
        id="flights_action_date_grid",
        desc="[Action Node] google.comflights:F4:A12 - Switch to Date Grid view to compare prices",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F4:P8 - Table data understanding (price comparison)
    price_comparison_ok = has_any_ci(answer, ['14', '15', '16']) and (has_any_ci(answer, ['cheaper', 'price', 'cost']) or looks_like_price(flights_info.flight_price))

    evaluator.add_custom_node(
        result=bool(price_comparison_ok),
        id="flights_perception_price_comparison",
        desc="[Perception Node] google.comflights:F4:P8 - Compare prices across May 14, 15, 16 and select optimal date per $50 rule",
        parent=flights_node,
        critical=False
    )

    # Check if non-stop filter was applied
    nonstop_ok = has_any_ci(answer, ['non-stop', 'nonstop', 'direct flight'])

    evaluator.add_custom_node(
        result=bool(nonstop_ok),
        id="flights_nonstop_filter",
        desc="Filter applied for non-stop flights only",
        parent=flights_node,
        critical=False
    )

    # Check if all flight details are present
    has_price = looks_like_price(flights_info.flight_price)
    has_airline = bool(flights_info.airline and flights_info.airline.strip())
    has_date = bool(flights_info.departure_date and flights_info.departure_date.strip())
    has_url = looks_like_url(flights_info.flights_url, 'google.com')

    evaluator.add_custom_node(
        result=bool(has_price and has_airline and has_date and has_url),
        id="flights_complete_info",
        desc="All flight details provided: date, price, airline, and URL",
        parent=flights_node,
        critical=False
    )

    # 3.3 Tripadvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="Tripadvisor - Find 4-star hotel in Venice with Travelers' Choice badge",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Multi-condition filtering
    tripadvisor_search_ok = has_any_ci(answer, ['tripadvisor']) and has_any_ci(answer, ['venice', 'venezia'])
    four_star_ok = has_any_ci(answer, ['4-star', '4 star', 'four star'])
    price_range_ok = has_any_ci(answer, ['300', '600']) or (hotel_info.nightly_rate and is_price_in_range(hotel_info.nightly_rate, 300, 600))

    evaluator.add_custom_node(
        result=bool(tripadvisor_search_ok and four_star_ok and price_range_ok),
        id="tripadvisor_action_filters",
        desc="[Action Node] tripadvisor.com:F1:A4 - Apply filters: 4-star rating and $300-$600 price range",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A5 - Sort by Traveler Ranked
    sort_ok = has_any_ci(answer, ['traveler ranked', 'ranked', 'sort'])

    evaluator.add_custom_node(
        result=bool(sort_ok),
        id="tripadvisor_action_sort",
        desc="[Action Node] tripadvisor.com:F1:A5 - Sort by 'Traveler Ranked' to get top-ranked hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F1:P2 - Badge recognition (Travelers' Choice)
    travelers_choice_ok = bool(hotel_info.has_travelers_choice) or has_any_ci(answer, ["travelers' choice", 'travelers choice', "traveler's choice"])

    evaluator.add_custom_node(
        result=bool(travelers_choice_ok),
        id="tripadvisor_perception_badge",
        desc="[Perception Node] tripadvisor.com:F1:P2 - Verify hotel has Travelers' Choice badge",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A8 - Click card to enter details
    has_hotel_name = bool(hotel_info.hotel_name and hotel_info.hotel_name.strip())
    has_hotel_url = looks_like_url(hotel_info.tripadvisor_url, 'tripadvisor')

    evaluator.add_custom_node(
        result=bool(has_hotel_name and has_hotel_url),
        id="tripadvisor_action_click_card",
        desc="[Action Node] tripadvisor.com:F1:A8 - Click on top-ranked hotel card to view details",
        parent=tripadvisor_node,
        critical=False
    )

    # Check if all hotel details are present
    has_rate = looks_like_price(hotel_info.nightly_rate)
    rate_in_range = is_price_in_range(hotel_info.nightly_rate, 300, 600)

    evaluator.add_custom_node(
        result=bool(has_hotel_name and has_rate and rate_in_range and travelers_choice_ok and has_hotel_url),
        id="tripadvisor_complete_info",
        desc="All hotel details provided: name, nightly rate ($300-$600), Travelers' Choice badge confirmation, and URL",
        parent=tripadvisor_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
