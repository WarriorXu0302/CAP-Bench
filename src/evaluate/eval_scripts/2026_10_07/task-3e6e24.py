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
TASK_ID = "task-3e6e24"
TASK_DESCRIPTION = "I am planning to fly from San Francisco to Austin this coming Friday to Sunday to attend a music festival. Please help me plan a complete weekend itinerary:\n\nFirst, go to Eventbrite and search for music festival events in Austin for this coming Friday to Sunday. Find a large electronic or rock music festival and note its start time, end time, and specific venue address.\n\nThen, use Google Flights to search for round-trip flights from San Francisco to Austin. The outbound flight must arrive in Austin at least 4 hours before the festival begins. The return flight should depart on the morning following the festival's end. Only consider direct flights, with a price under $500.\n\nNext, go to Airbnb to find accommodation near the festival venue. Check-in on this coming Friday and check-out on the following Monday. Use Google Maps to verify that the walking distance from the accommodation address to the festival venue is indeed within 2 miles. Filter for listings with a rating of 4.5 or higher and a price under $150 per night.\n\nFinally, use Yelp to search for restaurants near the accommodation address. Find three restaurants with a rating of 4 stars or higher, a price level of $$ or below, and that are currently open.\n\nOutput: Festival name, start time, end time, venue address, Eventbrite event page link; Outbound flight number, departure time, arrival time, price; Return flight number, departure time, arrival time, price, Google Flights booking link; Accommodation name, address, rating, price per night, walking distance to festival venue, Airbnb listing link, Google Maps distance verification screenshot link; Names, addresses, ratings, price levels, operating hours, and Yelp page links for the three restaurants."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FestivalInfo(BaseModel):
    """Festival details extracted from the answer"""
    name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    venue_address: Optional[str] = None
    eventbrite_link: Optional[str] = None


class OutboundFlight(BaseModel):
    """Outbound flight details extracted from the answer"""
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    price: Optional[str] = None


class ReturnFlight(BaseModel):
    """Return flight details extracted from the answer"""
    flight_number: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_time: Optional[str] = None
    price: Optional[str] = None
    booking_link: Optional[str] = None


class Accommodation(BaseModel):
    """Accommodation details extracted from the answer"""
    name: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[str] = None
    price_per_night: Optional[str] = None
    walking_distance: Optional[str] = None
    airbnb_link: Optional[str] = None
    maps_verification_link: Optional[str] = None


class Restaurant(BaseModel):
    """Restaurant details extracted from the answer"""
    name: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[str] = None
    price_level: Optional[str] = None
    hours: Optional[str] = None
    yelp_link: Optional[str] = None


class RestaurantsInfo(BaseModel):
    """All three restaurants extracted from the answer"""
    restaurant_1: Optional[Restaurant] = None
    restaurant_2: Optional[Restaurant] = None
    restaurant_3: Optional[Restaurant] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_festival() -> str:
    return """
Extract the festival information from the answer:

- name: the festival name
- start_time: the festival start time
- end_time: the festival end time
- venue_address: the specific venue address in Austin
- eventbrite_link: the Eventbrite event page link

If any field is missing, set it to null.
"""


def prompt_extract_outbound_flight() -> str:
    return """
Extract the outbound flight information (San Francisco to Austin) from the answer:

- flight_number: the flight number
- departure_time: the departure time
- arrival_time: the arrival time in Austin
- price: the ticket price

If any field is missing, set it to null.
"""


def prompt_extract_return_flight() -> str:
    return """
Extract the return flight information (Austin to San Francisco) from the answer:

- flight_number: the flight number
- departure_time: the departure time from Austin
- arrival_time: the arrival time
- price: the ticket price
- booking_link: the Google Flights booking link

If any field is missing, set it to null.
"""


def prompt_extract_accommodation() -> str:
    return """
Extract the accommodation information from the answer:

- name: the accommodation name
- address: the accommodation address
- rating: the accommodation rating
- price_per_night: the price per night
- walking_distance: the walking distance to the festival venue
- airbnb_link: the Airbnb listing link
- maps_verification_link: the Google Maps distance verification screenshot link

If any field is missing, set it to null.
"""


def prompt_extract_restaurants() -> str:
    return """
Extract information for all three restaurants mentioned in the answer:

For each restaurant (restaurant_1, restaurant_2, restaurant_3):
- name: the restaurant name
- address: the restaurant address
- rating: the restaurant rating
- price_level: the price level ($ or $$)
- hours: the operating hours
- yelp_link: the Yelp page link

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


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.'])


def is_austin_address(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'austin')


def is_march_14_16(text: Optional[str]) -> bool:
    """Check if date is in March 14-16 range"""
    if not text:
        return False
    return has_any_ci(text, ['march 14', 'march 15', 'march 16', '3/14', '3/15', '3/16', 'mar 14', 'mar 15', 'mar 16'])


def is_electronic_or_rock(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['electronic', 'edm', 'rock', 'music festival'])


def is_direct_flight(answer: str) -> bool:
    return has_any_ci(answer, ['direct', 'nonstop', 'non-stop']) or not has_any_ci(answer, ['stop', 'layover', 'connection'])


def total_price_under_500(outbound_price: Optional[str], return_price: Optional[str]) -> bool:
    ob = extract_float(outbound_price)
    ret = extract_float(return_price)
    if ob is None or ret is None:
        return False
    return (ob + ret) < 500


def rating_at_least(rating_text: Optional[str], threshold: float) -> bool:
    rating = extract_float(rating_text)
    if rating is None:
        return False
    return rating >= threshold


def price_under(price_text: Optional[str], threshold: float) -> bool:
    price = extract_float(price_text)
    if price is None:
        return False
    return price < threshold


def distance_under_2_miles(distance_text: Optional[str]) -> bool:
    dist = extract_float(distance_text)
    if dist is None:
        return False
    return dist <= 2.0


def is_valid_price_level(price_level: Optional[str]) -> bool:
    if not price_level:
        return False
    return price_level.strip() in ['$', '$$']


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
    festival_info = await evaluator.extract(
        prompt=prompt_extract_festival(),
        template_class=FestivalInfo,
        extraction_name="festival_info"
    )

    outbound_flight = await evaluator.extract(
        prompt=prompt_extract_outbound_flight(),
        template_class=OutboundFlight,
        extraction_name="outbound_flight"
    )

    return_flight = await evaluator.extract(
        prompt=prompt_extract_return_flight(),
        template_class=ReturnFlight,
        extraction_name="return_flight"
    )

    accommodation = await evaluator.extract(
        prompt=prompt_extract_accommodation(),
        template_class=Accommodation,
        extraction_name="accommodation"
    )

    restaurants = await evaluator.extract(
        prompt=prompt_extract_restaurants(),
        template_class=RestaurantsInfo,
        extraction_name="restaurants"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music festival search in Austin",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date selector operation
    date_ok = is_march_14_16(festival_info.start_time) or is_march_14_16(festival_info.end_time)
    evaluator.add_custom_node(
        result=bool(date_ok),
        id="eventbrite_date_selector",
        desc="[Action Node] eventbrite.com:F1:A1 - Select event dates for March 14-16, 2026",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location dropdown selection
    location_ok = is_austin_address(festival_info.venue_address)
    evaluator.add_custom_node(
        result=bool(location_ok),
        id="eventbrite_location_selector",
        desc="[Action Node] eventbrite.com:F1:A12 - Select Austin as the city location",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A13 - Category navigation
    category_ok = is_electronic_or_rock(festival_info.name)
    evaluator.add_custom_node(
        result=bool(category_ok),
        id="eventbrite_category_navigation",
        desc="[Action Node] eventbrite.com:F1:A13 - Navigate to music category for electronic or rock festivals",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Event status label recognition
    festival_name_ok = bool(festival_info.name and festival_info.name.strip())
    eventbrite_link_ok = looks_like_url(festival_info.eventbrite_link)
    evaluator.add_custom_node(
        result=bool(festival_name_ok and eventbrite_link_ok),
        id="eventbrite_event_status",
        desc="[Perception Node] eventbrite.com:F1:P1 - Identify large festival scale and status label",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - Event card click
    complete_info_ok = bool(festival_info.start_time and festival_info.end_time and festival_info.venue_address)
    evaluator.add_custom_node(
        result=bool(complete_info_ok),
        id="eventbrite_card_click",
        desc="[Action Node] eventbrite.com:F3:A8 - Click event card to view start time, end time, and venue address details",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P6 - Venue feature extraction
    venue_detailed_ok = bool(festival_info.venue_address and len(festival_info.venue_address.strip()) > 10)
    evaluator.add_custom_node(
        result=bool(venue_detailed_ok),
        id="eventbrite_venue_extraction",
        desc="[Perception Node] eventbrite.com:F3:P6 - Extract complete navigable venue address",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights round-trip search from San Francisco to Austin",
        parent=root,
        critical=False
    )

    # [Action Node] google.comflights:F1:A1 - Origin and destination input
    flight_route_ok = has_any_ci(answer, ['san francisco', 'sfo']) and has_any_ci(answer, ['austin', 'aus'])
    evaluator.add_custom_node(
        result=bool(flight_route_ok),
        id="flights_origin_destination",
        desc="[Action Node] google.comflights:F1:A1 - Input San Francisco as origin and Austin as destination",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - Date selection
    outbound_date_ok = is_march_14_16(outbound_flight.departure_time)
    return_date_ok = has_any_ci(return_flight.departure_time, ['march 17', '3/17', 'mar 17'])
    evaluator.add_custom_node(
        result=bool(outbound_date_ok and return_date_ok),
        id="flights_date_selection",
        desc="[Action Node] google.comflights:F1:A2 - Select outbound March 14 and return March 17 dates",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - Stops filter (nonstop only)
    direct_flight_ok = is_direct_flight(answer)
    evaluator.add_custom_node(
        result=bool(direct_flight_ok),
        id="flights_stops_filter",
        desc="[Action Node] google.comflights:F3:A5 - Filter for nonstop/direct flights only",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F3:A5 - Price range filter
    total_price_ok = total_price_under_500(outbound_flight.price, return_flight.price)
    evaluator.add_custom_node(
        result=bool(total_price_ok),
        id="flights_price_filter",
        desc="[Action Node] google.comflights:F3:A5 - Set price filter under $500 total",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F6:A14 - Flight details expansion
    arrival_time_ok = bool(outbound_flight.arrival_time)
    evaluator.add_custom_node(
        result=bool(arrival_time_ok),
        id="flights_details_expansion",
        desc="[Action Node] google.comflights:F6:A14 - Expand flight details to verify arrival is 4+ hours before festival start",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F6:P1 - Flight time data recognition
    flight_data_complete = bool(
        outbound_flight.flight_number and outbound_flight.departure_time and
        outbound_flight.arrival_time and outbound_flight.price and
        return_flight.flight_number and return_flight.departure_time and
        return_flight.arrival_time and return_flight.price
    )
    evaluator.add_custom_node(
        result=bool(flight_data_complete),
        id="flights_time_data_recognition",
        desc="[Perception Node] google.comflights:F6:P1 - Identify complete flight times and prices for both flights",
        parent=flights_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search near festival venue",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Destination search filter
    accommodation_near_venue = bool(accommodation.address and is_austin_address(accommodation.address))
    evaluator.add_custom_node(
        result=bool(accommodation_near_venue),
        id="airbnb_destination_search",
        desc="[Action Node] airbnb.com:F1:A10 - Search for accommodation near festival venue address",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Check-in/check-out date selection
    dates_in_link = bool(accommodation.airbnb_link and ('march' in accommodation.airbnb_link.lower() or '2026' in accommodation.airbnb_link.lower()))
    evaluator.add_custom_node(
        result=bool(dates_in_link or has_any_ci(answer, ['march 14', 'march 17', 'check-in', 'check-out'])),
        id="airbnb_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Select check-in March 14 and check-out March 17 dates",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price slider filter
    price_ok = price_under(accommodation.price_per_night, 150)
    evaluator.add_custom_node(
        result=bool(price_ok),
        id="airbnb_price_filter",
        desc="[Action Node] airbnb.com:F2:A13 - Set price slider to filter under $150 per night",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A34 - Filter drawer operation
    rating_ok = rating_at_least(accommodation.rating, 4.5)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="airbnb_filter_drawer",
        desc="[Action Node] airbnb.com:F2:A34 - Open filter panel and set rating to 4.5 or higher",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F2:P10 - Listing information understanding
    listing_info_complete = bool(accommodation.rating and accommodation.price_per_night)
    evaluator.add_custom_node(
        result=bool(listing_info_complete and rating_ok and price_ok),
        id="airbnb_listing_understanding",
        desc="[Perception Node] airbnb.com:F2:P10 - Understand listing rating and price meet filter criteria",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps distance verification from accommodation to festival venue",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    distance_verified = distance_under_2_miles(accommodation.walking_distance)
    maps_link_ok = looks_like_url(accommodation.maps_verification_link)
    evaluator.add_custom_node(
        result=bool(distance_verified and maps_link_ok),
        id="maps_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Input accommodation and venue addresses for route planning",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection
    walking_mode_ok = has_any_ci(answer, ['walking', 'walk'])
    evaluator.add_custom_node(
        result=bool(walking_mode_ok),
        id="maps_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking transportation mode",
        parent=maps_node,
        critical=False
    )

    # 3.5 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp restaurant search near accommodation",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Business keyword and location search
    yelp_search_ok = has_any_ci(answer, ['yelp', 'restaurant'])
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_search_restaurants",
        desc="[Action Node] yelp.com:F1:A1 - Search for restaurants near accommodation address",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-condition filtering
    r1 = restaurants.restaurant_1
    r2 = restaurants.restaurant_2
    r3 = restaurants.restaurant_3

    three_restaurants_ok = bool(r1 and r2 and r3)
    all_ratings_ok = (rating_at_least(r1.rating if r1 else None, 4.0) and
                      rating_at_least(r2.rating if r2 else None, 4.0) and
                      rating_at_least(r3.rating if r3 else None, 4.0))
    all_prices_ok = (is_valid_price_level(r1.price_level if r1 else None) and
                     is_valid_price_level(r2.price_level if r2 else None) and
                     is_valid_price_level(r3.price_level if r3 else None))

    evaluator.add_custom_node(
        result=bool(three_restaurants_ok and all_ratings_ok and all_prices_ok),
        id="yelp_multi_filter",
        desc="[Action Node] yelp.com:F1:A2 - Apply filters: rating 4+ stars, price level $$ or below, currently open",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Basic business information perception
    all_basic_info_ok = bool(
        r1 and r1.name and r1.rating and r1.price_level and
        r2 and r2.name and r2.rating and r2.price_level and
        r3 and r3.name and r3.rating and r3.price_level
    )
    evaluator.add_custom_node(
        result=bool(all_basic_info_ok),
        id="yelp_basic_info_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Perceive basic information (rating, price level) for three restaurants",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P2 - Operating status label recognition
    all_hours_ok = bool(
        r1 and r1.hours and r2 and r2.hours and r3 and r3.hours
    )
    evaluator.add_custom_node(
        result=bool(all_hours_ok),
        id="yelp_operating_status",
        desc="[Perception Node] yelp.com:F1:P2 - Identify operating status showing currently open",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F2:A9 - Business card click
    all_details_ok = bool(
        r1 and r1.address and r1.hours and r1.yelp_link and
        r2 and r2.address and r2.hours and r2.yelp_link and
        r3 and r3.address and r3.hours and r3.yelp_link
    )
    evaluator.add_custom_node(
        result=bool(all_details_ok),
        id="yelp_card_click",
        desc="[Action Node] yelp.com:F2:A9 - Click restaurant cards to view detailed information",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P7 - Operating hours perception
    complete_hours_ok = bool(
        r1 and r1.hours and len(r1.hours.strip()) > 5 and
        r2 and r2.hours and len(r2.hours.strip()) > 5 and
        r3 and r3.hours and len(r3.hours.strip()) > 5
    )
    evaluator.add_custom_node(
        result=bool(complete_hours_ok),
        id="yelp_hours_perception",
        desc="[Perception Node] yelp.com:F2:P7 - Perceive complete operating hours schedule from business details",
        parent=yelp_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
