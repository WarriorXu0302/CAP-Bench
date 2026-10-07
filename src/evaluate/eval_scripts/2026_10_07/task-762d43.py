import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-762d43"
TASK_DESCRIPTION = "I'm a student in NYC planning an upcoming weekend trip to see the band 'The National'. First, check Spotify's concert listings to find a show scheduled for an upcoming Friday or Saturday. Select a city that seems close to New York (e.g., Philadelphia, Boston, DC). Once a target city is found, go to Google Maps and calculate the driving directions from 'New York, NY' to that city's downtown to confirm the driving distance is under 250 miles. Next, look for accommodation on Expedia: search for hotels in that city for the night of the concert (i.e., the evening of the selected Friday or Saturday). Use the interactive map view to pick a hotel that is rated 4.0 or higher and costs under $250. Finally, check Eventbrite for other 'Music' events in that city for the following day (i.e., Saturday or Sunday), specifically looking for an event marked as 'Selling fast' or 'Promoted' to round out the weekend. Output: The concert city and date, Google Maps driving distance, the chosen hotel's name, price, and rating, and the Eventbrite event name with its status tag. Provide URLs for the Spotify artist page, Google Maps route, Expedia hotel detail, and Eventbrite event."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConcertInfo(BaseModel):
    """Concert details extracted from the answer"""
    concert_city: Optional[str] = None
    concert_date: Optional[str] = None
    spotify_url: Optional[str] = None


class DrivingInfo(BaseModel):
    """Driving distance details extracted from the answer"""
    driving_distance: Optional[str] = None
    google_maps_url: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_name: Optional[str] = None
    hotel_price: Optional[str] = None
    hotel_rating: Optional[str] = None
    expedia_url: Optional[str] = None


class EventbriteInfo(BaseModel):
    """Eventbrite event details extracted from the answer"""
    event_name: Optional[str] = None
    event_status_tag: Optional[str] = None
    eventbrite_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_concert_from_answer() -> str:
    return """
Extract the concert details for 'The National' from the answer.

Return:
- concert_city: the city where the concert is scheduled (e.g., "Philadelphia", "Boston", "Washington DC"). If not present, set null.
- concert_date: the date of the concert exactly as written in the answer. If not present, set null.
- spotify_url: the Spotify artist page or concert listing URL if provided. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_driving_from_answer() -> str:
    return """
From the answer, extract the driving distance from New York, NY to the concert city.

Return:
- driving_distance: the driving distance exactly as stated (include units like "miles" or "mi" if present).
- google_maps_url: the Google Maps route URL if provided. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_hotel_from_answer() -> str:
    return """
From the answer, extract the hotel details from Expedia.

Return:
- hotel_name: the name of the selected hotel exactly as written.
- hotel_price: the price per night exactly as stated (include currency symbols if present).
- hotel_rating: the hotel rating exactly as stated (e.g., "4.2", "4.5/5").
- expedia_url: the Expedia hotel detail page URL if provided. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_eventbrite_from_answer() -> str:
    return """
From the answer, extract the Eventbrite event details for the following day.

Return:
- event_name: the name of the music event exactly as written.
- event_status_tag: the status tag for the event (e.g., "Selling fast", "Promoted") exactly as written.
- eventbrite_url: the Eventbrite event URL if provided. If not present, set null.

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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['mile', 'mi', 'km'])


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return '$' in text or has_any_ci(text, ['dollar'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def is_friday_or_saturday(date_text: Optional[str]) -> bool:
    if not date_text:
        return False
    date_lower = date_text.lower()
    # Check for explicit day names
    if 'friday' in date_lower or 'saturday' in date_lower or 'fri' in date_lower or 'sat' in date_lower:
        return True
    # Try parsing various date formats and checking day of week
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'[A-Za-z]+ \d{1,2},? \d{4}',
        r'\d{1,2} [A-Za-z]+ \d{4}'
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, date_text)
        if matches:
            return True  # Accept if contains a date pattern
    return False


def nearby_city_to_nyc(city: Optional[str]) -> bool:
    if not city:
        return False
    nearby_cities = ['philadelphia', 'boston', 'washington', 'dc', 'baltimore', 'providence',
                     'hartford', 'new haven', 'albany', 'trenton', 'newark']
    return has_any_ci(city, nearby_cities)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def distance_under_250_miles(distance_text: Optional[str]) -> bool:
    if not distance_text:
        return False
    num = extract_float(distance_text)
    if num is None:
        return False
    return num < 250


def rating_at_least_4(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    if num is None:
        return False
    return num >= 4.0


def price_under_250(price_text: Optional[str]) -> bool:
    if not price_text:
        return False
    num = extract_float(price_text)
    if num is None:
        return False
    return num < 250


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
    concert_info = await evaluator.extract(
        prompt=prompt_extract_concert_from_answer(),
        template_class=ConcertInfo,
        extraction_name="concert_info"
    )

    driving_info = await evaluator.extract(
        prompt=prompt_extract_driving_from_answer(),
        template_class=DrivingInfo,
        extraction_name="driving_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_from_answer(),
        template_class=EventbriteInfo,
        extraction_name="eventbrite_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify concert listings for 'The National'",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F9:A23 - Scroll Loading
    spotify_mentions_the_national = has_any_ci(answer, ['the national'])
    spotify_mentions_concert = has_any_ci(answer, ['concert', 'tour', 'show'])
    evaluator.add_custom_node(
        result=bool(spotify_mentions_the_national and spotify_mentions_concert),
        id="spotify_action_scroll_loading",
        desc="[Action Node] open.spotify.com:F9:A23 - Check Spotify concert listings (scroll through tour dates if needed)",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F9:P14 - List Content Understanding (Friday or Saturday)
    date_is_friday_or_saturday = is_friday_or_saturday(concert_info.concert_date)
    evaluator.add_custom_node(
        result=bool(date_is_friday_or_saturday),
        id="spotify_perception_day_of_week",
        desc="[Perception Node] open.spotify.com:F9:P14 - Identify concert scheduled for Friday or Saturday",
        parent=spotify_node,
        critical=False
    )

    # City is nearby to NYC
    city_nearby = nearby_city_to_nyc(concert_info.concert_city)
    evaluator.add_custom_node(
        result=bool(city_nearby),
        id="spotify_city_selection",
        desc="Selected city is near New York (e.g., Philadelphia, Boston, DC)",
        parent=spotify_node,
        critical=False
    )

    # Spotify URL provided
    spotify_url_ok = looks_like_url(concert_info.spotify_url)
    evaluator.add_custom_node(
        result=bool(spotify_url_ok),
        id="spotify_url_provided",
        desc="Spotify artist or tour page URL provided",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps driving directions from NYC to concert city",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Data Input
    maps_mentions_nyc = has_any_ci(answer, ['new york'])
    maps_has_destination = bool(concert_info.concert_city)
    evaluator.add_custom_node(
        result=bool(maps_mentions_nyc and maps_has_destination),
        id="maps_action_data_input",
        desc="[Action Node] maps.google.com:F2:A5 - Input driving directions from 'New York, NY' to concert city",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transport Mode Selection (driving)
    distance_looks_valid = looks_like_distance(driving_info.driving_distance)
    mentions_driving = has_any_ci(answer, ['driv', 'car'])
    evaluator.add_custom_node(
        result=bool(distance_looks_valid and mentions_driving),
        id="maps_action_transport_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode for directions",
        parent=maps_node,
        critical=False
    )

    # Distance under 250 miles constraint
    distance_under_250 = distance_under_250_miles(driving_info.driving_distance)
    evaluator.add_custom_node(
        result=bool(distance_under_250),
        id="maps_distance_constraint",
        desc="Driving distance is under 250 miles",
        parent=maps_node,
        critical=False
    )

    # Google Maps URL provided
    maps_url_ok = looks_like_url(driving_info.google_maps_url)
    evaluator.add_custom_node(
        result=bool(maps_url_ok),
        id="maps_url_provided",
        desc="Google Maps route URL provided",
        parent=maps_node,
        critical=False
    )

    # 3.3 Expedia section
    expedia_node = evaluator.add_sequential(
        id="expedia_section",
        desc="Expedia hotel search for concert night",
        parent=root,
        critical=False
    )

    # [Action Node] expedia.com:F2:A8 - Date Picker Range
    expedia_mentions_concert_date = bool(concert_info.concert_date and hotel_info.hotel_name)
    evaluator.add_custom_node(
        result=bool(expedia_mentions_concert_date),
        id="expedia_action_date_picker",
        desc="[Action Node] expedia.com:F2:A8 - Search for hotels on the night of the concert",
        parent=expedia_node,
        critical=False
    )

    # [Action Node] expedia.com:F2:A33 - Map Marker Interaction
    expedia_mentions_map = has_any_ci(answer, ['map', 'interactive'])
    hotel_in_city = bool(hotel_info.hotel_name and concert_info.concert_city)
    evaluator.add_custom_node(
        result=bool(expedia_mentions_map or hotel_in_city),
        id="expedia_action_map_interaction",
        desc="[Action Node] expedia.com:F2:A33 - Use interactive map view to select a hotel",
        parent=expedia_node,
        critical=False
    )

    # [Perception Node] expedia.com:F2:P3 - Status/Tag Perception (rating and price)
    rating_valid = looks_like_rating(hotel_info.hotel_rating)
    rating_4_plus = rating_at_least_4(hotel_info.hotel_rating)
    price_valid = looks_like_price(hotel_info.hotel_price)
    price_under_250_ok = price_under_250(hotel_info.hotel_price)
    evaluator.add_custom_node(
        result=bool(rating_valid and rating_4_plus and price_valid and price_under_250_ok),
        id="expedia_perception_rating_price",
        desc="[Perception Node] expedia.com:F2:P3 - Hotel rated 4.0+ and costs under $250",
        parent=expedia_node,
        critical=False
    )

    # Hotel details provided
    hotel_details_ok = bool(hotel_info.hotel_name and hotel_info.hotel_price and hotel_info.hotel_rating)
    evaluator.add_custom_node(
        result=bool(hotel_details_ok),
        id="expedia_hotel_details",
        desc="Hotel name, price, and rating all provided",
        parent=expedia_node,
        critical=False
    )

    # Expedia URL provided
    expedia_url_ok = looks_like_url(hotel_info.expedia_url)
    evaluator.add_custom_node(
        result=bool(expedia_url_ok),
        id="expedia_url_provided",
        desc="Expedia hotel detail page URL provided",
        parent=expedia_node,
        critical=False
    )

    # 3.4 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music event for the following day",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location Selection
    eventbrite_mentions_music = has_any_ci(answer, ['music', 'eventbrite'])
    event_in_same_city = bool(eventbrite_info.event_name and concert_info.concert_city)
    evaluator.add_custom_node(
        result=bool(eventbrite_mentions_music and event_in_same_city),
        id="eventbrite_action_location",
        desc="[Action Node] eventbrite.com:F1:A12 - Search for music events in the concert city",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Status Tag Recognition
    status_tag_present = bool(eventbrite_info.event_status_tag)
    status_tag_valid = has_any_ci(eventbrite_info.event_status_tag, ['selling fast', 'promoted'])
    evaluator.add_custom_node(
        result=bool(status_tag_present and status_tag_valid),
        id="eventbrite_perception_status_tag",
        desc="[Perception Node] eventbrite.com:F1:P1 - Event marked as 'Selling fast' or 'Promoted'",
        parent=eventbrite_node,
        critical=False
    )

    # Following day context
    mentions_following_day = has_any_ci(answer, ['following day', 'next day', 'saturday', 'sunday'])
    evaluator.add_custom_node(
        result=bool(mentions_following_day),
        id="eventbrite_following_day",
        desc="Event is for the following day (Saturday or Sunday)",
        parent=eventbrite_node,
        critical=False
    )

    # Eventbrite URL provided
    eventbrite_url_ok = looks_like_url(eventbrite_info.eventbrite_url)
    evaluator.add_custom_node(
        result=bool(eventbrite_url_ok),
        id="eventbrite_url_provided",
        desc="Eventbrite event URL provided",
        parent=eventbrite_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
