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
TASK_ID = "task-3fc3ea"
TASK_DESCRIPTION = 'I’m planning to visit the Getty Center in Los Angeles this weekend (Saturday) to see an exhibition. First, check the official website to see whether there are any exhibitions in the **Current Exhibitions** section related to **ancient history** or **medieval art**, and confirm that your selected exhibition is open to the public on the planned travel day. After choosing the most interesting one, record its name.\n\nThen, since I don’t want to spend the whole day only at the museum, search Eventbrite for Saturday afternoon or evening events near the Getty Center (in the Brentwood or Westwood area). Find one event in either the **“Music”** or **“Performing & Visual Arts”** category, preferably **free** or priced at **$50 or less**, and note its time and location.\n\nFinally, use Google Maps to plan the route: assume I depart from Santa Monica at 10:00 AM, go to the Getty Center first, and then attend that event. Along the way, use **Search along the route** to find an Italian restaurant with a rating of **4.5 or above** for dinner. If there is no Italian restaurant rated 4.5+ along the route, choose the highest-rated Italian restaurant available along the route and specify its rating.\n\nPlease compile a complete itinerary schedule and the total travel time for the full route.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GettyExhibitionInfo(BaseModel):
    """Exhibition information extracted from Getty Center website"""
    exhibition_name: Optional[str] = None
    theme_match: Optional[str] = None
    saturday_open_status: Optional[str] = None


class EventbriteEventInfo(BaseModel):
    """Event information extracted from Eventbrite search"""
    event_name: Optional[str] = None
    event_category: Optional[str] = None
    event_time: Optional[str] = None
    event_location: Optional[str] = None
    event_price: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Restaurant information found along the route"""
    restaurant_name: Optional[str] = None
    restaurant_rating: Optional[str] = None
    cuisine_type: Optional[str] = None


class ItineraryInfo(BaseModel):
    """Complete itinerary details"""
    departure_point: Optional[str] = None
    departure_time: Optional[str] = None
    first_stop: Optional[str] = None
    second_stop: Optional[str] = None
    restaurant_stop: Optional[str] = None
    total_travel_time: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_getty_exhibition() -> str:
    return """
Extract the Getty Center exhibition information from the answer:

- exhibition_name: the name of the exhibition selected from Current Exhibitions section
- theme_match: what theme it matches (ancient history or medieval art)
- saturday_open_status: whether the answer confirms the exhibition is open on Saturday

If any field is missing, set it to null.
"""


def prompt_extract_eventbrite_event() -> str:
    return """
Extract the Eventbrite event information from the answer:

- event_name: the name of the event found
- event_category: the category (Music or Performing & Visual Arts)
- event_time: the time of the event (afternoon or evening on Saturday)
- event_location: the location of the event (should be in Brentwood or Westwood area)
- event_price: the price (free or $50 or less)

If any field is missing, set it to null.
"""


def prompt_extract_restaurant() -> str:
    return """
Extract the Italian restaurant information found along the route from the answer:

- restaurant_name: the name of the restaurant
- restaurant_rating: the rating of the restaurant (should be 4.5 or above if available, otherwise the highest available)
- cuisine_type: should be Italian

If any field is missing, set it to null.
"""


def prompt_extract_itinerary() -> str:
    return """
Extract the complete itinerary information from the answer:

- departure_point: starting point (should be Santa Monica)
- departure_time: departure time (should be 10:00 AM)
- first_stop: first destination (should be Getty Center)
- second_stop: second destination (the Eventbrite event)
- restaurant_stop: the Italian restaurant for dinner
- total_travel_time: total travel time for the full route

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


def looks_like_rating(text: Optional[str], min_rating: float = 4.5) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    if rating is None:
        return False
    return rating >= min_rating or ci_contains(text, 'highest')


def looks_like_price(text: Optional[str], max_price: float = 50.0) -> bool:
    if not text:
        return False
    if ci_contains(text, 'free'):
        return True
    price = extract_float(text)
    if price is None:
        return False
    return price <= max_price


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
    getty_info = await evaluator.extract(
        prompt=prompt_extract_getty_exhibition(),
        template_class=GettyExhibitionInfo,
        extraction_name="getty_exhibition"
    )

    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_event(),
        template_class=EventbriteEventInfo,
        extraction_name="eventbrite_event"
    )

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant(),
        template_class=RestaurantInfo,
        extraction_name="restaurant"
    )

    itinerary_info = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryInfo,
        extraction_name="itinerary"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Getty Center section
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty Center exhibition research and selection",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F2:A1 - Navigate to Current Exhibitions tab
    getty_action_tab_ok = (has_any_ci(answer, ['getty', 'getty center']) and
                           has_any_ci(answer, ['current exhibitions', 'current exhibition']))
    evaluator.add_custom_node(
        result=bool(getty_action_tab_ok),
        id="getty_action_tab",
        desc="[Action Node] getty.edu:F2:A1 - Navigate to the Current Exhibitions section on Getty Center website",
        parent=getty_node,
        critical=False
    )

    # [Perception Node] getty.edu:F2:P5 - Confirm Saturday opening status
    saturday_status_ok = (getty_info and getty_info.saturday_open_status and
                          has_any_ci(getty_info.saturday_open_status, ['open', 'saturday', 'weekend']))
    evaluator.add_custom_node(
        result=bool(saturday_status_ok),
        id="getty_perception_saturday",
        desc="[Perception Node] getty.edu:F2:P5 - Confirm the selected exhibition is open to the public on Saturday",
        parent=getty_node,
        critical=False
    )

    # Check exhibition name and theme match
    exhibition_name_ok = bool(getty_info and getty_info.exhibition_name and getty_info.exhibition_name.strip())
    theme_ok = (getty_info and getty_info.theme_match and
                has_any_ci(getty_info.theme_match, ['ancient history', 'medieval art']))
    evaluator.add_custom_node(
        result=bool(exhibition_name_ok and theme_ok),
        id="getty_exhibition_details",
        desc="Exhibition name recorded and matches ancient history or medieval art theme",
        parent=getty_node,
        critical=False
    )

    # 3.2 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search near Getty Center",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A2 - Apply category filters
    eventbrite_action_filter_ok = (has_any_ci(answer, ['eventbrite']) and
                                    (has_any_ci(answer, ['music']) or
                                     has_any_ci(answer, ['performing', 'visual arts'])))
    evaluator.add_custom_node(
        result=bool(eventbrite_action_filter_ok),
        id="eventbrite_action_filter",
        desc="[Action Node] eventbrite.com:F2:A2 - Apply category filters (Music or Performing & Visual Arts) on Eventbrite",
        parent=eventbrite_node,
        critical=False
    )

    # Check event details
    event_name_ok = bool(eventbrite_info and eventbrite_info.event_name and eventbrite_info.event_name.strip())
    event_category_ok = (eventbrite_info and eventbrite_info.event_category and
                         (ci_contains(eventbrite_info.event_category, 'music') or
                          ci_contains(eventbrite_info.event_category, 'performing') or
                          ci_contains(eventbrite_info.event_category, 'visual arts')))
    event_time_ok = (eventbrite_info and eventbrite_info.event_time and
                     (ci_contains(eventbrite_info.event_time, 'afternoon') or
                      ci_contains(eventbrite_info.event_time, 'evening') or
                      ci_contains(eventbrite_info.event_time, 'saturday')))
    event_location_ok = (eventbrite_info and eventbrite_info.event_location and
                         (ci_contains(eventbrite_info.event_location, 'brentwood') or
                          ci_contains(eventbrite_info.event_location, 'westwood')))
    event_price_ok = (eventbrite_info and eventbrite_info.event_price and
                      looks_like_price(eventbrite_info.event_price))

    evaluator.add_custom_node(
        result=bool(event_name_ok and event_category_ok and event_time_ok and event_location_ok and event_price_ok),
        id="eventbrite_event_details",
        desc="Event found with correct category, time (Saturday afternoon/evening), location (Brentwood/Westwood), and price (free or ≤$50)",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps route planning with restaurant search",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A20 - Use Search along the route feature
    maps_action_search_ok = (has_any_ci(answer, ['google maps', 'maps']) and
                             has_any_ci(answer, ['search along', 'along the route', 'along route']))
    evaluator.add_custom_node(
        result=bool(maps_action_search_ok),
        id="maps_action_search_along",
        desc="[Action Node] maps.google.com:F2:A20 - Use Search along the route feature to find Italian restaurant",
        parent=maps_node,
        critical=False
    )

    # Check restaurant details
    restaurant_name_ok = bool(restaurant_info and restaurant_info.restaurant_name and restaurant_info.restaurant_name.strip())
    cuisine_ok = (restaurant_info and restaurant_info.cuisine_type and
                  ci_contains(restaurant_info.cuisine_type, 'italian'))
    rating_ok = (restaurant_info and restaurant_info.restaurant_rating and
                 (looks_like_rating(restaurant_info.restaurant_rating, 4.5) or
                  ci_contains(restaurant_info.restaurant_rating, 'highest')))

    evaluator.add_custom_node(
        result=bool(restaurant_name_ok and cuisine_ok and rating_ok),
        id="maps_restaurant_details",
        desc="Italian restaurant found with rating 4.5+ (or highest available rating specified)",
        parent=maps_node,
        critical=False
    )

    # 3.4 Itinerary compilation
    itinerary_node = evaluator.add_sequential(
        id="itinerary_section",
        desc="Complete itinerary compilation",
        parent=root,
        critical=False
    )

    # Check itinerary structure
    departure_ok = (itinerary_info and itinerary_info.departure_point and
                    ci_contains(itinerary_info.departure_point, 'santa monica'))
    time_ok = (itinerary_info and itinerary_info.departure_time and
               ci_contains(itinerary_info.departure_time, '10'))
    getty_stop_ok = (itinerary_info and itinerary_info.first_stop and
                     ci_contains(itinerary_info.first_stop, 'getty'))
    event_stop_ok = bool(itinerary_info and itinerary_info.second_stop and itinerary_info.second_stop.strip())
    restaurant_stop_ok = bool(itinerary_info and itinerary_info.restaurant_stop and itinerary_info.restaurant_stop.strip())
    total_time_ok = bool(itinerary_info and itinerary_info.total_travel_time and contains_digits(itinerary_info.total_travel_time))

    evaluator.add_custom_node(
        result=bool(departure_ok and time_ok and getty_stop_ok and event_stop_ok and restaurant_stop_ok and total_time_ok),
        id="itinerary_complete",
        desc="Complete itinerary compiled with all stops (Santa Monica → Getty Center → Restaurant → Event) and total travel time",
        parent=itinerary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
