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
TASK_ID = "task-ede7c1"
TASK_DESCRIPTION = 'I plan to attend a music event in Austin on a specific weekend next month.\n\nPlease start by searching Eventbrite for music events happening in Austin on that weekend. Identify a popular event or one with fast-selling tickets, and then determine its name, specific location, and date.\n\nOnce the event is selected, use its location as the central point to search Airbnb for accommodations within walking distance. I require properties with good reviews, ideally host-managed, and choose the option that offers the best value for money.\n\nFollowing this, based on the event date, search for flights from New York (JFK) to Austin (AUS). Ensure the arrival time allows ample time before the event begins, and preferably find a direct flight.\n\nFinally, compile the key details for this complete itinerary (event, accommodation, and flights) and provide an estimated total cost.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from Eventbrite"""
    event_name: Optional[str] = None
    event_location: Optional[str] = None
    event_date: Optional[str] = None
    popularity_indicator: Optional[str] = None


class AccommodationInfo(BaseModel):
    """Accommodation details extracted from Airbnb"""
    property_name: Optional[str] = None
    distance_from_event: Optional[str] = None
    review_info: Optional[str] = None
    host_type: Optional[str] = None
    price: Optional[str] = None


class FlightInfo(BaseModel):
    """Flight details extracted from Skyscanner"""
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    flight_type: Optional[str] = None
    airline: Optional[str] = None
    price: Optional[str] = None


class TotalCost(BaseModel):
    """Total estimated cost"""
    total_cost_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the Eventbrite event details from the answer:

- event_name: the name of the selected music event
- event_location: the specific location/venue of the event
- event_date: the date of the event
- popularity_indicator: any mention of "Selling Fast", "Sold Out", "popular", or similar indicators

If any field is missing, set it to null.
"""


def prompt_extract_accommodation_from_answer() -> str:
    return """
Extract the Airbnb accommodation details from the answer:

- property_name: the name or description of the selected property
- distance_from_event: any mention of walking distance or proximity to the event location
- review_info: information about reviews or ratings
- host_type: whether it's host-managed or mentions host details
- price: the price or cost of the accommodation

If any field is missing, set it to null.
"""


def prompt_extract_flight_from_answer() -> str:
    return """
Extract the flight details from Skyscanner from the answer:

- departure_time: the departure time from JFK
- arrival_time: the arrival time at AUS
- flight_type: whether it's direct/nonstop or has connections
- airline: the airline name if mentioned
- price: the flight price

If any field is missing, set it to null.
"""


def prompt_extract_total_cost_from_answer() -> str:
    return """
Extract the total estimated cost for the complete itinerary from the answer:

- total_cost_text: the total cost estimation that combines event, accommodation, and flights

If not present, set it to null.
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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar', 'price', 'cost'])


def mentions_popularity(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['selling fast', 'sold out', 'popular', 'high demand', 'tickets going fast'])


def mentions_walking_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['walking distance', 'walkable', 'walk to', 'steps from', 'nearby', 'close to'])


def mentions_reviews(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['review', 'rating', 'rated', 'stars'])


def mentions_direct_flight(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['direct', 'nonstop', 'non-stop'])


def mentions_arrival_timing(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['arrive', 'arrival', 'before event', 'ample time', 'enough time'])


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
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_from_answer(),
        template_class=EventInfo,
        extraction_name="event_info"
    )

    accommodation_info = await evaluator.extract(
        prompt=prompt_extract_accommodation_from_answer(),
        template_class=AccommodationInfo,
        extraction_name="accommodation_info"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_from_answer(),
        template_class=FlightInfo,
        extraction_name="flight_info"
    )

    total_cost = await evaluator.extract(
        prompt=prompt_extract_total_cost_from_answer(),
        template_class=TotalCost,
        extraction_name="total_cost"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music event search in Austin",
        parent=root,
        critical=False
    )

    # Check if Eventbrite was used and Austin mentioned
    eventbrite_used = has_any_ci(answer, ['eventbrite']) and has_any_ci(answer, ['austin'])
    evaluator.add_custom_node(
        result=bool(eventbrite_used),
        id="eventbrite_search_action",
        desc="Search Eventbrite for music events in Austin",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Identify popularity/fast-selling indicators
    popularity_ok = (mentions_popularity(event_info.popularity_indicator) or
                     mentions_popularity(answer))
    evaluator.add_custom_node(
        result=bool(popularity_ok),
        id="eventbrite_popularity_perception",
        desc="[Perception Node] eventbrite.com:F1:P1 - Identify event popularity indicators (Selling Fast, Sold Out, etc.)",
        parent=eventbrite_node,
        critical=False
    )

    # Event details extracted
    event_details_ok = (bool(event_info.event_name and event_info.event_name.strip()) and
                        bool(event_info.event_location and event_info.event_location.strip()) and
                        bool(event_info.event_date and event_info.event_date.strip()))
    evaluator.add_custom_node(
        result=bool(event_details_ok),
        id="eventbrite_event_details",
        desc="Extract event name, specific location, and date",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search near event location",
        parent=root,
        critical=False
    )

    # Check if Airbnb was used
    airbnb_used = has_any_ci(answer, ['airbnb'])
    evaluator.add_custom_node(
        result=bool(airbnb_used),
        id="airbnb_search_action",
        desc="Search Airbnb for accommodations",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A38 - Map search centered on event location
    map_search_ok = (mentions_walking_distance(accommodation_info.distance_from_event) or
                     mentions_walking_distance(answer) or
                     has_any_ci(answer, ['map', 'location', 'centered']))
    evaluator.add_custom_node(
        result=bool(map_search_ok),
        id="airbnb_map_search_action",
        desc="[Action Node] airbnb.com:F4:A38 - Use map search or location-based filtering centered on event venue",
        parent=airbnb_node,
        critical=False
    )

    # Walking distance mentioned
    walking_distance_ok = mentions_walking_distance(answer)
    evaluator.add_custom_node(
        result=bool(walking_distance_ok),
        id="airbnb_walking_distance",
        desc="Identify properties within walking distance of the event",
        parent=airbnb_node,
        critical=False
    )

    # Good reviews mentioned
    reviews_ok = (mentions_reviews(accommodation_info.review_info) or
                  mentions_reviews(answer))
    evaluator.add_custom_node(
        result=bool(reviews_ok),
        id="airbnb_good_reviews",
        desc="Consider properties with good reviews/ratings",
        parent=airbnb_node,
        critical=False
    )

    # Host-managed mentioned
    host_managed_ok = has_any_ci(answer, ['host', 'hosted', 'host-managed'])
    evaluator.add_custom_node(
        result=bool(host_managed_ok),
        id="airbnb_host_managed",
        desc="Consider host-managed properties",
        parent=airbnb_node,
        critical=False
    )

    # Value for money / price mentioned
    accommodation_price_ok = looks_like_price(accommodation_info.price) or has_any_ci(answer, ['value', 'affordable', 'best price'])
    evaluator.add_custom_node(
        result=bool(accommodation_price_ok),
        id="airbnb_value_selection",
        desc="Select property offering best value for money",
        parent=airbnb_node,
        critical=False
    )

    # 3.3 Skyscanner section
    skyscanner_node = evaluator.add_sequential(
        id="skyscanner_section",
        desc="Skyscanner flight search from JFK to AUS",
        parent=root,
        critical=False
    )

    # Check if Skyscanner was used and route mentioned
    skyscanner_used = (has_any_ci(answer, ['skyscanner', 'flight', 'flights']) and
                       has_any_ci(answer, ['jfk', 'new york']) and
                       has_any_ci(answer, ['aus', 'austin']))
    evaluator.add_custom_node(
        result=bool(skyscanner_used),
        id="skyscanner_search_action",
        desc="Search flights from JFK to AUS based on event date",
        parent=skyscanner_node,
        critical=False
    )

    # [Perception Node] skyscanner.com:F1:P2 - Verify arrival time compatibility with event
    timing_ok = (mentions_arrival_timing(answer) or
                 (bool(flight_info.arrival_time) and bool(event_info.event_date)))
    evaluator.add_custom_node(
        result=bool(timing_ok),
        id="skyscanner_arrival_timing_perception",
        desc="[Perception Node] skyscanner.com:F1:P2 - Verify arrival time allows ample time before event begins",
        parent=skyscanner_node,
        critical=False
    )

    # [Action Node] skyscanner.com:F1:A5 - Filter for direct flights
    direct_filter_ok = (mentions_direct_flight(flight_info.flight_type) or
                        mentions_direct_flight(answer))
    evaluator.add_custom_node(
        result=bool(direct_filter_ok),
        id="skyscanner_direct_flight_action",
        desc="[Action Node] skyscanner.com:F1:A5 - Filter or select direct/nonstop flights",
        parent=skyscanner_node,
        critical=False
    )

    # Flight details extracted
    flight_details_ok = (bool(flight_info.departure_time) or bool(flight_info.arrival_time))
    evaluator.add_custom_node(
        result=bool(flight_details_ok),
        id="skyscanner_flight_details",
        desc="Extract flight timing details",
        parent=skyscanner_node,
        critical=False
    )

    # Flight price extracted
    flight_price_ok = looks_like_price(flight_info.price)
    evaluator.add_custom_node(
        result=bool(flight_price_ok),
        id="skyscanner_flight_price",
        desc="Extract flight price",
        parent=skyscanner_node,
        critical=False
    )

    # 3.4 Final compilation section
    compilation_node = evaluator.add_parallel(
        id="compilation_section",
        desc="Compile complete itinerary with total cost",
        parent=root,
        critical=False
    )

    # Itinerary summary provided
    itinerary_ok = (bool(event_info.event_name) and
                    bool(accommodation_info.property_name or accommodation_info.price) and
                    bool(flight_info.arrival_time or flight_info.price))
    evaluator.add_custom_node(
        result=bool(itinerary_ok),
        id="itinerary_compiled",
        desc="Complete itinerary details compiled (event, accommodation, flights)",
        parent=compilation_node,
        critical=False
    )

    # Total cost estimation provided
    total_cost_ok = (looks_like_price(total_cost.total_cost_text) or
                     has_any_ci(answer, ['total cost', 'total price', 'estimated cost', 'overall cost']))
    evaluator.add_custom_node(
        result=bool(total_cost_ok),
        id="total_cost_provided",
        desc="Estimated total cost for complete itinerary provided",
        parent=compilation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
