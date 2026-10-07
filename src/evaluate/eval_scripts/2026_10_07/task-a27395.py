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
TASK_ID = "task-a27395"
TASK_DESCRIPTION = 'I’m planning a trip to Las Vegas from Thursday to Sunday on a weekend in the coming months. As an F1 fan, I must stay near the “Sphere,” since it is in the F1 track’s T-Mobile zone.\n\nFirst, find “Sphere, Las Vegas” on Google Maps, then search for nearby hotels. Select 3 hotels, and use Google Maps route planning to verify one by one that the walking time from each hotel to the Sphere is no more than 15 minutes (including 15 minutes).\n\nNext, check these 3 hotels on Booking.com for the corresponding Thursday check-in and Sunday check-out dates, and filter for hotels rated 4 stars or above only. If any hotel does not meet the star-rating or date requirements, replace it with another hotel that still meets the walking-distance requirement.\n\nFinally, use Skyscanner to search for flights from New York (JFK) to Las Vegas (LAS) on the same Thursday. I need a direct flight that lands no later than 2:00 PM (14:00) so I can check in. Sort results by price from low to high.\n\nOutput: the names of the 3 hotels, walking time (minutes) measured on Google Maps, total price on Booking.com, star rating, and for Skyscanner, the cheapest qualifying flight’s flight number, departure/arrival times, price, and details-page link.\n\nIf availability is too limited for that Thursday–Sunday weekend window, search the same Thursday–Sunday pattern in a later month and indicate the actual dates used in the output.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HotelInfo(BaseModel):
    """Information for a single hotel"""
    name: Optional[str] = None
    walking_time_minutes: Optional[str] = None
    booking_price: Optional[str] = None
    star_rating: Optional[str] = None


class HotelsData(BaseModel):
    """All three hotels extracted from the answer"""
    hotel1: Optional[HotelInfo] = None
    hotel2: Optional[HotelInfo] = None
    hotel3: Optional[HotelInfo] = None


class FlightInfo(BaseModel):
    """Flight information extracted from the answer"""
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    price: Optional[str] = None
    details_link: Optional[str] = None


class DatesUsed(BaseModel):
    """Dates information extracted from the answer"""
    thursday_date: Optional[str] = None
    sunday_date: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hotels_from_answer() -> str:
    return """
Extract information about the 3 hotels mentioned in the answer. For each hotel, extract:
- name: the hotel name
- walking_time_minutes: the walking time from the hotel to the Sphere in minutes as stated
- booking_price: the total price on Booking.com as stated (include currency if present)
- star_rating: the star rating as stated

Return hotel1, hotel2, and hotel3 objects. If any field is missing, set it to null.
"""


def prompt_extract_flight_from_answer() -> str:
    return """
Extract the flight information from the answer. Extract:
- flight_number: the flight number
- departure_time: the departure time
- arrival_time: the arrival time
- price: the flight price (include currency if present)
- details_link: the details page link/URL

If any field is missing, set it to null.
"""


def prompt_extract_dates_from_answer() -> str:
    return """
Extract the travel dates used in the answer:
- thursday_date: the Thursday check-in date
- sunday_date: the Sunday check-out date

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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_walking_time(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Walking time should be reasonable (0-15 minutes for this task)
    return 0 <= num <= 15


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', '€', '£', 'usd', 'eur', 'gbp'])


def looks_like_star_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Star rating should be 4 or above for this task
    return num >= 4


def looks_like_time(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept formats like "10:30 AM", "14:00", "2:00 PM", etc.
    patterns = [
        r'\d{1,2}:\d{2}',
        r'\d{1,2}\s*(am|pm|AM|PM)',
    ]
    return any(re.search(p, text) for p in patterns)


def arrival_before_2pm(text: Optional[str]) -> bool:
    if not text:
        return False
    # Try to extract hour
    text_lower = text.lower()
    # Look for patterns like "1:30 PM", "13:45", "11:00 AM"
    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text_lower)
    if not time_match:
        return False

    hour = int(time_match.group(1))
    minute = int(time_match.group(2))
    period = time_match.group(3)

    # Convert to 24-hour format
    if period:
        if period == 'pm' and hour != 12:
            hour += 12
        elif period == 'am' and hour == 12:
            hour = 0

    # Check if before 14:00 (2:00 PM)
    return hour < 14 or (hour == 14 and minute == 0)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://']) or has_any_ci(text, ['.com', '.net', '.org'])


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
    hotels_data = await evaluator.extract(
        prompt=prompt_extract_hotels_from_answer(),
        template_class=HotelsData,
        extraction_name="hotels_data"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_from_answer(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )

    dates_used = await evaluator.extract(
        prompt=prompt_extract_dates_from_answer(),
        template_class=DatesUsed,
        extraction_name="dates_used"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Google Maps section
    gmaps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps - Find Sphere, search nearby hotels, verify walking distances",
        parent=root,
        critical=False
    )

    # Check if Sphere was mentioned
    sphere_mention = has_any_ci(answer, ['sphere'])
    evaluator.add_custom_node(
        result=bool(sphere_mention),
        id="gmaps_sphere_mention",
        desc="Mentions finding 'Sphere, Las Vegas' on Google Maps",
        parent=gmaps_node,
        critical=False
    )

    # Check if hotels were found
    hotels_found = has_any_ci(answer, ['hotel'])
    evaluator.add_custom_node(
        result=bool(hotels_found),
        id="gmaps_hotels_search",
        desc="Mentions searching for nearby hotels on Google Maps",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input start/end points for route
    # Verify by checking if walking times are present for hotels
    hotels = [hotels_data.hotel1, hotels_data.hotel2, hotels_data.hotel3]
    walking_times_present = sum(1 for h in hotels if h and h.walking_time_minutes) >= 3

    evaluator.add_custom_node(
        result=bool(walking_times_present),
        id="gmaps_action_route_input",
        desc="[Action Node] maps.google.com:F2:A5 - Input start and end points for route planning (hotel to Sphere)",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select walking mode
    # Verify by checking if walking times match walking speed (not driving time)
    all_walking_times_valid = all(
        looks_like_walking_time(h.walking_time_minutes)
        for h in hotels if h and h.walking_time_minutes
    )

    evaluator.add_custom_node(
        result=bool(all_walking_times_valid),
        id="gmaps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking as the transportation mode",
        parent=gmaps_node,
        critical=False
    )

    # Check that 3 hotels with valid walking times are provided
    three_hotels_ok = sum(
        1 for h in hotels
        if h and h.name and h.walking_time_minutes and looks_like_walking_time(h.walking_time_minutes)
    ) >= 3

    evaluator.add_custom_node(
        result=bool(three_hotels_ok),
        id="gmaps_three_hotels_with_walking_time",
        desc="Three hotels with valid walking times (≤15 minutes) are provided",
        parent=gmaps_node,
        critical=False
    )

    # 3.2 Booking.com section
    booking_node = evaluator.add_sequential(
        id="booking_section",
        desc="Booking.com - Check hotel availability, prices, and star ratings",
        parent=root,
        critical=False
    )

    # Check if Booking.com was mentioned
    booking_mention = has_any_ci(answer, ['booking.com', 'booking'])
    evaluator.add_custom_node(
        result=bool(booking_mention),
        id="booking_mention",
        desc="Mentions checking hotels on Booking.com",
        parent=booking_node,
        critical=False
    )

    # Check if dates were specified (Thursday check-in, Sunday check-out)
    dates_ok = bool(dates_used and dates_used.thursday_date and dates_used.sunday_date)
    thursday_ok = has_any_ci(answer, ['thursday', 'thurs', 'thu'])
    sunday_ok = has_any_ci(answer, ['sunday', 'sun'])

    evaluator.add_custom_node(
        result=bool(dates_ok and thursday_ok and sunday_ok),
        id="booking_dates_specified",
        desc="Thursday check-in and Sunday check-out dates are specified",
        parent=booking_node,
        critical=False
    )

    # Check if prices are provided for hotels
    prices_ok = sum(
        1 for h in hotels
        if h and h.booking_price and looks_like_price(h.booking_price)
    ) >= 3

    evaluator.add_custom_node(
        result=bool(prices_ok),
        id="booking_prices_provided",
        desc="Total prices from Booking.com are provided for the 3 hotels",
        parent=booking_node,
        critical=False
    )

    # Check if star ratings are provided and are 4 stars or above
    star_ratings_ok = sum(
        1 for h in hotels
        if h and h.star_rating and looks_like_star_rating(h.star_rating)
    ) >= 3

    evaluator.add_custom_node(
        result=bool(star_ratings_ok),
        id="booking_star_ratings_4plus",
        desc="Star ratings (4 stars or above) are provided for the 3 hotels",
        parent=booking_node,
        critical=False
    )

    # 3.3 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner - Search for flights from JFK to LAS on Thursday",
        parent=root,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A1 - Input search parameters (JFK to LAS)
    jfk_mention = has_any_ci(answer, ['jfk', 'new york'])
    las_mention = has_any_ci(answer, ['las', 'las vegas'])

    evaluator.add_custom_node(
        result=bool(jfk_mention and las_mention),
        id="skyscanner_action_search_input",
        desc="[Action Node] skyscanner.com:F1:A1 - Input search for flights from New York (JFK) to Las Vegas (LAS)",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A2 - Date selection (Thursday)
    flight_date_ok = bool(dates_used and dates_used.thursday_date) and thursday_ok

    evaluator.add_custom_node(
        result=bool(flight_date_ok),
        id="skyscanner_action_date_selection",
        desc="[Action Node] skyscanner.com:F1:A2 - Select the Thursday departure date",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A5 - Direct flight filter (checkbox)
    direct_mention = has_any_ci(answer, ['direct', 'non-stop', 'nonstop'])

    evaluator.add_custom_node(
        result=bool(direct_mention),
        id="skyscanner_action_direct_filter",
        desc="[Action Node] skyscanner.com:F1:A5 - Apply direct/non-stop flight filter",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A6 - Arrival time filter (before 2:00 PM)
    arrival_time_valid = (flight_info and flight_info.arrival_time and
                         looks_like_time(flight_info.arrival_time) and
                         arrival_before_2pm(flight_info.arrival_time))

    evaluator.add_custom_node(
        result=bool(arrival_time_valid),
        id="skyscanner_action_arrival_filter",
        desc="[Action Node] skyscanner.com:F1:A6 - Filter for flights arriving no later than 2:00 PM (14:00)",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A7 - Sort by price (low to high)
    sort_mention = has_any_ci(answer, ['sort', 'price', 'low to high', 'cheapest', 'lowest'])

    evaluator.add_custom_node(
        result=bool(sort_mention),
        id="skyscanner_action_sort_price",
        desc="[Action Node] skyscanner.com:F1:A7 - Sort results by price from low to high",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F6:P1 - Extract flight price
    flight_price_ok = (flight_info and flight_info.price and
                      looks_like_price(flight_info.price))

    evaluator.add_custom_node(
        result=bool(flight_price_ok),
        id="skyscanner_perception_price",
        desc="[Perception Node] skyscanner.com:F6:P1 - Extract the flight price",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F6:P2 - Extract departure and arrival times
    times_ok = (flight_info and
               flight_info.departure_time and looks_like_time(flight_info.departure_time) and
               flight_info.arrival_time and looks_like_time(flight_info.arrival_time))

    evaluator.add_custom_node(
        result=bool(times_ok),
        id="skyscanner_perception_times",
        desc="[Perception Node] skyscanner.com:F6:P2 - Extract departure and arrival times",
        parent=skyscanner_node,
        critical=False
    )

    # Check if flight number is provided
    flight_number_ok = bool(flight_info and flight_info.flight_number and flight_info.flight_number.strip())

    evaluator.add_custom_node(
        result=bool(flight_number_ok),
        id="skyscanner_flight_number",
        desc="Flight number is provided",
        parent=skyscanner_node,
        critical=False
    )

    # Check if details link is provided
    details_link_ok = (flight_info and flight_info.details_link and
                      looks_like_url(flight_info.details_link))

    evaluator.add_custom_node(
        result=bool(details_link_ok),
        id="skyscanner_details_link",
        desc="Details page link/URL is provided",
        parent=skyscanner_node,
        critical=False
    )

    # 3.4 Overall output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Overall output completeness check",
        parent=root,
        critical=False
    )

    # All 3 hotels with complete information
    all_hotels_complete = sum(
        1 for h in hotels
        if h and h.name and h.walking_time_minutes and h.booking_price and h.star_rating
    ) >= 3

    evaluator.add_custom_node(
        result=bool(all_hotels_complete),
        id="output_hotels_complete",
        desc="All 3 hotels have name, walking time, Booking.com price, and star rating",
        parent=output_node,
        critical=False
    )

    # Flight information is complete
    flight_complete = (flight_info and
                      flight_info.flight_number and
                      flight_info.departure_time and
                      flight_info.arrival_time and
                      flight_info.price and
                      flight_info.details_link)

    evaluator.add_custom_node(
        result=bool(flight_complete),
        id="output_flight_complete",
        desc="Flight information is complete (flight number, times, price, details link)",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
