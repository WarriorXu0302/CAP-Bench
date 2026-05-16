import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-13f79e"
TASK_DESCRIPTION = """
I'm a jazz enthusiast currently in London, planning to go to New York to catch a live performance. Please first go to Eventbrite and help me find jazz performances in New York that still have tickets available for a weekend in the coming weeks, for example, from next Friday to next Sunday. Don't just pick the first results; scroll down or browse a few pages. Choose an event that looks interesting and still has tickets available. Note its name, specific performance time, and venue.

Based on this performance time, go to Skyscanner and search for round-trip flights from London (LHR) to New York. I want direct flights only; if none are available, choose the flight with the shortest travel time.

Finally, go to Expedia and look for hotels in New York for those performance days. Find one with a rating of 4 stars or higher that includes free breakfast.

Finally, compile and present the selected event, flight information, and hotel recommendation.
"""

NYC_AIRPORT_KEYWORDS = ["new york", "nyc", "jfk", "ewr", "lga", "newark", "la guardia", "laGuardia"]
LHR_KEYWORDS = ["lhr", "heathrow", "london heathrow", "london (lhr)"]

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Selected Eventbrite event details"""
    name: Optional[str] = None
    performance_date: Optional[str] = None
    performance_time: Optional[str] = None
    venue: Optional[str] = None
    day_of_week: Optional[str] = None
    availability_phrase: Optional[str] = None


class FlightInfo(BaseModel):
    """Selected Skyscanner flight details"""
    origin: Optional[str] = None
    destination: Optional[str] = None
    outbound_date: Optional[str] = None
    return_date: Optional[str] = None
    is_direct_claimed: Optional[bool] = None
    stops_description: Optional[str] = None
    travel_time_outbound: Optional[str] = None
    travel_time_return: Optional[str] = None
    airline: Optional[str] = None
    filter_notes: Optional[str] = None


class HotelInfo(BaseModel):
    """Selected Expedia hotel details"""
    name: Optional[str] = None
    rating_text: Optional[str] = None
    rating_value: Optional[float] = 0.0
    breakfast_included: Optional[bool] = None
    checkin_date: Optional[str] = None
    checkout_date: Optional[str] = None
    location: Optional[str] = None


class Selections(BaseModel):
    """Aggregated selections"""
    event: Optional[EventInfo] = None
    flight: Optional[FlightInfo] = None
    hotel: Optional[HotelInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
    From the assistant's answer, extract the single chosen Eventbrite jazz event in New York.

    Extract:
    - name: the event name
    - performance_date: the date string for the performance (e.g., 'Fri, May 10, 2026')
    - performance_time: the time string for the performance (e.g., '8:00 PM')
    - venue: the venue name (e.g., 'Blue Note, New York')
    - day_of_week: the stated day of week if present (e.g., 'Friday', 'Saturday', 'Sunday')
    - availability_phrase: the phrase indicating tickets are still available (e.g., 'Tickets available', 'Buy tickets', 'Available'), or null if not provided

    If any field is not explicitly provided, set it to null.
    Return a single JSON object.
    """


def prompt_extract_flight_from_answer() -> str:
    return """
    From the assistant's answer, extract the selected Skyscanner round-trip flight information.

    Extract:
    - origin: the departure airport or city (prefer 'LHR' or 'Heathrow' if present)
    - destination: the arrival airport or city (e.g., 'JFK', 'New York')
    - outbound_date: the outbound flight date string
    - return_date: the return flight date string
    - is_direct_claimed: true if the answer claims direct or nonstop flights; false if it explicitly says none available; null if unclear
    - stops_description: any description of stops (e.g., 'nonstop', '0 stops', '1 stop in Reykjavik')
    - travel_time_outbound: total outbound travel time if provided
    - travel_time_return: total return travel time if provided
    - airline: the airline(s) if provided
    - filter_notes: any mention of applying 'direct flights only' filter or choosing 'shortest travel time' if direct not available

    If any field is missing, set it to null.
    Return a single JSON object.
    """


def prompt_extract_hotel_from_answer() -> str:
    return """
    From the assistant's answer, extract the Expedia hotel details that match the performance days.

    Extract:
    - name: the hotel name
    - rating_text: the rating text provided (e.g., '4.5-star', '4.2/5', '4-star hotel')
    - rating_value: a numeric rating; if multiple are present, prefer star rating out of 5; if only text like '4-star', set 4.0
    - breakfast_included: true if 'free breakfast' or similar is included; false otherwise
    - checkin_date: the check-in date string
    - checkout_date: the check-out date string
    - location: the city or area, e.g., 'New York'

    Return a single JSON object.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def contains_any(text: Optional[str], keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def mentions_weekend_or_days(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(d in t for d in ["weekend", "friday", "saturday", "sunday", "fri", "sat", "sun"])


def mentions_scrolling_or_pagination(text: Optional[str]) -> bool:
    return contains_any(text, [
        "scroll", "scrolled", "scrolling", "scroll down", "load more", "browse", "browsed",
        "next page", "page 2", "page two", "previous page", "pagination", "more results"
    ])


def mentions_tickets_available(text: Optional[str]) -> bool:
    # Positive indicators of availability and not sold out
    positive = any(kw in (text or "").lower() for kw in [
        "tickets available", "available", "buy tickets", "get tickets", "book now",
        "reserve now", "on sale", "select tickets"
    ])
    not_sold_out = "sold out" not in (text or "").lower()
    return positive and not_sold_out


def parse_rating_value(rating_text: Optional[str], fallback_value: Optional[float] = None) -> Optional[float]:
    if rating_text is None:
        return fallback_value
    t = rating_text.lower()
    # Look for forms like "4.5/5", "4.2 out of 5", "4-star", "4 stars"
    m = re.search(r'(\d+(\.\d+)?)\s*/\s*5', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'(\d+(\.\d+)?)\s*out of\s*5', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'(\d+(\.\d+)?)\s*-\s*star', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r'(\d+(\.\d+)?)\s*star', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Sometimes "4-star hotel" or "4 stars"
    m = re.search(r'(\d+(\.\d+)?)\s*stars?', t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # As a lenient fallback, any first number <= 5
    m = re.search(r'(\d+(\.\d+)?)', t)
    if m:
        try:
            val = float(m.group(1))
            if 0.0 < val <= 5.0:
                return val
        except Exception:
            pass
    return fallback_value


def is_direct_or_nonstop(text: Optional[str]) -> bool:
    return contains_any(text, ["direct", "nonstop", "0 stop", "0-stop", "non-stop", "non stop"])


def mentions_shortest_time_choice(text: Optional[str]) -> bool:
    return contains_any(text, ["shortest travel time", "shortest duration", "fastest", "least travel time"])


def mentions_round_trip(text: Optional[str]) -> bool:
    return contains_any(text, ["round trip", "round-trip", "return flight", "return ticket"])


def route_mentions_lhr_to_nyc(origin: Optional[str], destination: Optional[str], answer: str) -> bool:
    # Check origin/destination fields first, fallback to answer text
    origin_text = (origin or "") + " " + answer
    dest_text = (destination or "") + " " + answer
    origin_ok = contains_any(origin_text, LHR_KEYWORDS) or contains_any(answer, LHR_KEYWORDS)
    dest_ok = contains_any(dest_text, NYC_AIRPORT_KEYWORDS) or contains_any(answer, NYC_AIRPORT_KEYWORDS)
    return origin_ok and dest_ok


def alignment_mentions(answer: str) -> bool:
    # Check that the user linked steps: flights based on performance time; hotels for those days
    return contains_any(answer, [
        "based on the performance time",
        "matching the performance date",
        "for those performance days",
        "for those days",
        "same dates as the performance",
        "using the performance dates",
        "aligned with the event date",
        "coinciding with the event"
    ])


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

    # -------- 1. Extract structured information from the answer ---------- #
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_from_answer(),
        template_class=EventInfo,
        extraction_name="selected_event"
    )

    flight_info = await evaluator.extract(
        prompt=prompt_extract_flight_from_answer(),
        template_class=FlightInfo,
        extraction_name="selected_flight"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_from_answer(),
        template_class=HotelInfo,
        extraction_name="selected_hotel"
    )

    # Safely post-process hotel rating
    if hotel_info:
        hotel_info.rating_value = parse_rating_value(hotel_info.rating_text, fallback_value=hotel_info.rating_value or 0.0)

    # -------- 2. Build evaluation tree ----------------------------------- #
    # Event verification
    event_node = evaluator.add_sequential(
        id="event_selection",
        desc="Verify Eventbrite event selection and details",
        parent=root,
        critical=False
    )

    # 2.1 [Action Node] Eventbrite date/weekend filtering (eventbrite.com:F1:A6)
    event_weekend_node = evaluator.add_custom_node(
        result=mentions_weekend_or_days(answer) or mentions_weekend_or_days((event_info.day_of_week or "")) or contains_any(answer, ["next weekend", "upcoming weekend", "coming weeks"]),
        id="event_weekend_filter",
        desc="[Action Node] eventbrite.com:F1:A6 - Applied/considered weekend date filtering (e.g., Friday–Sunday, upcoming weekends)",
        parent=event_node,
        critical=False
    )

    # 2.2 [Action Node] Eventbrite scrolling/pagination (eventbrite.com:F1:A12)
    event_scroll_node = evaluator.add_custom_node(
        result=mentions_scrolling_or_pagination(answer),
        id="event_scroll_pagination",
        desc="[Action Node] eventbrite.com:F1:A12 - Scrolled or browsed multiple results/pages before choosing event",
        parent=event_node,
        critical=False
    )

    # 2.3 [Perception Node] Eventbrite ticket availability (eventbrite.com:F1:P1)
    availability_text = (event_info.availability_phrase or "") + " " + answer
    event_availability_node = evaluator.add_custom_node(
        result=mentions_tickets_available(availability_text),
        id="event_ticket_availability",
        desc="[Perception Node] eventbrite.com:F1:P1 - Recognized ticket availability (not sold out) for the chosen event",
        parent=event_node,
        critical=False
    )

    # 2.4 Event details presence checks
    event_name_node = evaluator.add_custom_node(
        result=bool(event_info and event_info.name),
        id="event_name_present",
        desc="Event name is provided",
        parent=event_node,
        critical=False
    )

    event_time_node = evaluator.add_custom_node(
        result=bool(event_info and (event_info.performance_time or event_info.performance_date)),
        id="event_time_present",
        desc="Event performance date/time is provided",
        parent=event_node,
        critical=False
    )

    event_venue_node = evaluator.add_custom_node(
        result=bool(event_info and event_info.venue),
        id="event_venue_present",
        desc="Event venue is provided",
        parent=event_node,
        critical=False
    )

    # Flight verification
    flight_node = evaluator.add_sequential(
        id="flight_selection",
        desc="Verify Skyscanner flight selection and constraints",
        parent=root,
        critical=False
    )

    # 3.1 Route LHR -> NYC (lenient)
    route_ok = route_mentions_lhr_to_nyc(flight_info.origin if flight_info else None,
                                         flight_info.destination if flight_info else None,
                                         answer)
    flight_route_node = evaluator.add_custom_node(
        result=route_ok,
        id="flight_route_lhr_nyc",
        desc="Flight route is between London (LHR/Heathrow) and New York (JFK/EWR/LGA)",
        parent=flight_node,
        critical=False
    )

    # 3.2 [Action Node] Skyscanner direct flights only (skyscanner.com:F1:A5) with fallback to shortest time
    direct_claimed_text = ""
    if flight_info and flight_info.filter_notes:
        direct_claimed_text += " " + flight_info.filter_notes
    if flight_info and flight_info.stops_description:
        direct_claimed_text += " " + flight_info.stops_description
    direct_ok = is_direct_or_nonstop(direct_claimed_text or answer) or \
                ((flight_info.is_direct_claimed is False) and mentions_shortest_time_choice((flight_info.filter_notes or "") + " " + answer))
    flight_direct_node = evaluator.add_custom_node(
        result=bool(direct_ok),
        id="flight_direct_or_shortest",
        desc="[Action Node] skyscanner.com:F1:A5 - Applied 'Direct flights only' filter; if none found, chose shortest travel time",
        parent=flight_node,
        critical=False
    )

    # 3.3 Round-trip and dates present (lenient)
    roundtrip_ok = mentions_round_trip(answer) or (bool(flight_info and flight_info.outbound_date and flight_info.return_date))
    flight_roundtrip_node = evaluator.add_custom_node(
        result=roundtrip_ok,
        id="flight_roundtrip_dates",
        desc="Round-trip structure and outbound/return dates provided",
        parent=flight_node,
        critical=False
    )

    # Hotel verification
    hotel_node = evaluator.add_sequential(
        id="hotel_selection",
        desc="Verify Expedia hotel selection and constraints",
        parent=root,
        critical=False
    )

    # 4.1 Location is New York (lenient)
    hotel_loc_ok = contains_any((hotel_info.location or "") + " " + answer, ["new york", "nyc", "manhattan", "brooklyn", "queens", "bronx"])
    hotel_location_node = evaluator.add_custom_node(
        result=hotel_loc_ok,
        id="hotel_location_nyc",
        desc="Hotel is in New York",
        parent=hotel_node,
        critical=False
    )

    # 4.2 [Perception Node] Expedia free breakfast amenity (expedia.com:F2:P3)
    breakfast_ok = bool(hotel_info and hotel_info.breakfast_included) or contains_any(answer, ["free breakfast", "complimentary breakfast", "breakfast included"])
    hotel_breakfast_node = evaluator.add_custom_node(
        result=breakfast_ok,
        id="hotel_free_breakfast",
        desc="[Perception Node] expedia.com:F2:P3 - Recognized 'Free breakfast' amenity for the selected hotel",
        parent=hotel_node,
        critical=False
    )

    # 4.3 Rating ≥ 4 stars
    rating_value = (hotel_info.rating_value if hotel_info else 0.0) or parse_rating_value(hotel_info.rating_text if hotel_info else None, fallback_value=0.0)
    rating_ok = (rating_value or 0.0) >= 4.0
    hotel_rating_node = evaluator.add_custom_node(
        result=bool(rating_ok),
        id="hotel_rating_check",
        desc="Hotel has rating of 4 stars or higher",
        parent=hotel_node,
        critical=False
    )

    # 4.4 Dates present (lenient)
    hotel_dates_ok = bool(hotel_info and hotel_info.checkin_date and hotel_info.checkout_date)
    hotel_dates_node = evaluator.add_custom_node(
        result=hotel_dates_ok,
        id="hotel_dates_present",
        desc="Hotel check-in and check-out dates are provided",
        parent=hotel_node,
        critical=False
    )

    # Integration checks (information flow alignment)
    integration_node = evaluator.add_sequential(
        id="integration_flow",
        desc="Verify information flow alignment across Eventbrite → Skyscanner → Expedia",
        parent=root,
        critical=False
    )

    # 5.1 Flights linked to performance date (lenient: check linking phrases)
    flights_aligned_node = evaluator.add_custom_node(
        result=alignment_mentions(answer),
        id="flights_linked_to_event",
        desc="Flights appear chosen based on the performance date (linking phrases present)",
        parent=integration_node,
        critical=False
    )

    # 5.2 Hotels linked to performance days (lenient: check linking phrases)
    hotels_aligned_node = evaluator.add_custom_node(
        result=alignment_mentions(answer),
        id="hotels_linked_to_event",
        desc="Hotel dates appear chosen based on the performance days (linking phrases present)",
        parent=integration_node,
        critical=False
    )

    # Compilation & presentation
    compile_node = evaluator.add_sequential(
        id="final_compilation",
        desc="Verify final compiled presentation includes event, flight, and hotel",
        parent=root,
        critical=False
    )

    event_present = bool(event_info and (event_info.name or event_info.venue or event_info.performance_date or event_info.performance_time))
    flight_present = bool(flight_info and (flight_info.outbound_date or flight_info.return_date or flight_info.origin or flight_info.destination))
    hotel_present = bool(hotel_info and (hotel_info.name or hotel_info.rating_text or hotel_info.breakfast_included))

    compilation_event_node = evaluator.add_custom_node(
        result=event_present,
        id="compiled_event_present",
        desc="Compiled summary includes the selected event details",
        parent=compile_node,
        critical=False
    )

    compilation_flight_node = evaluator.add_custom_node(
        result=flight_present,
        id="compiled_flight_present",
        desc="Compiled summary includes the selected flight details",
        parent=compile_node,
        critical=False
    )

    compilation_hotel_node = evaluator.add_custom_node(
        result=hotel_present,
        id="compiled_hotel_present",
        desc="Compiled summary includes the hotel recommendation",
        parent=compile_node,
        critical=False
    )

    # -------- 3. Return structured result -------------------------------- #
    return evaluator.get_summary()