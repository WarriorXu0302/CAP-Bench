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
TASK_ID = "task-bb2e9f"
TASK_DESCRIPTION = 'I’m planning a trip to San Francisco next month to research AI technology trends. First, please find an in-person Conference or Seminar on Eventbrite related to “AI Agents” or “Generative AI,” with a ticket price of no more than $500. If there are no qualifying events next month, then look for the earliest qualifying in-person event within the following two months. After identifying a target event, search Reddit for the organizer’s name or the event series name and quickly check whether there have been any serious negative reviews in the past year (e.g., “scam” or “badly organized”). If the overall sentiment is positive or neutral, continue.\n\nNext, help me arrange accommodation. On Airbnb, find an “Entire home” with check-in on the day before the event starts, for a total stay of 3 nights. The property must be within a 2-mile walking distance of the event venue (be sure to verify walking distance using Google Maps), and the combined total of the event ticket plus the 3-night stay must not exceed $2,000. Output the event name, date, ticket price, organizer and Reddit reputation summary, Airbnb listing name, total accommodation cost, and the walking distance from the Airbnb to the venue as shown on Google Maps.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from the answer"""
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    ticket_price: Optional[str] = None
    organizer: Optional[str] = None
    event_location: Optional[str] = None


class RedditReputationInfo(BaseModel):
    """Reddit reputation check results"""
    reputation_summary: Optional[str] = None
    negative_reviews_found: Optional[bool] = None


class AirbnbInfo(BaseModel):
    """Airbnb listing details"""
    listing_name: Optional[str] = None
    total_accommodation_cost: Optional[str] = None
    checkin_date: Optional[str] = None
    nights: Optional[int] = None


class WalkingDistanceInfo(BaseModel):
    """Google Maps walking distance verification"""
    walking_distance: Optional[str] = None


class BudgetInfo(BaseModel):
    """Combined budget calculation"""
    total_cost: Optional[str] = None
    within_budget: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_info() -> str:
    return """
Extract the Eventbrite event details from the answer:

- event_name: the full name of the event
- event_date: the date when the event takes place
- ticket_price: the ticket price (include currency symbol if present)
- organizer: the name of the event organizer or organizing entity
- event_location: the venue or location where the event takes place

If any field is missing, set it to null.
"""


def prompt_extract_reddit_reputation() -> str:
    return """
Extract the Reddit reputation check results from the answer:

- reputation_summary: a summary of what was found about the organizer's reputation on Reddit
- negative_reviews_found: true if serious negative reviews were found, false otherwise, null if not checked

If information is missing, set fields to null.
"""


def prompt_extract_airbnb_info() -> str:
    return """
Extract the Airbnb accommodation details from the answer:

- listing_name: the name/title of the Airbnb listing
- total_accommodation_cost: the total cost for the stay (include currency if present)
- checkin_date: the check-in date
- nights: the number of nights (should be 3)

If any field is missing, set it to null.
"""


def prompt_extract_walking_distance() -> str:
    return """
Extract the Google Maps walking distance verification from the answer:

- walking_distance: the walking distance from the Airbnb to the event venue as shown on Google Maps

If missing, set to null.
"""


def prompt_extract_budget_info() -> str:
    return """
Extract the budget calculation from the answer:

- total_cost: the combined total of event ticket and accommodation
- within_budget: whether the total is within the $2,000 budget (true/false)

If missing, set to null.
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
    m = re.findall(r'(\d+(?:,\d+)?(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'usd', 'dollar'])


def is_within_500(price_text: Optional[str]) -> bool:
    val = extract_float(price_text)
    if val is None:
        return False
    return val <= 500


def looks_like_walking_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km', 'meter', 'feet', 'ft', 'walk'])


def is_within_2_miles(distance_text: Optional[str]) -> bool:
    if not distance_text:
        return False
    val = extract_float(distance_text)
    if val is None:
        return False
    if ci_contains(distance_text, 'km'):
        val = val * 0.621371
    return val <= 2.0


def is_within_2000_budget(ticket_text: Optional[str], accom_text: Optional[str]) -> bool:
    ticket = extract_float(ticket_text)
    accom = extract_float(accom_text)
    if ticket is None or accom is None:
        return False
    return (ticket + accom) <= 2000


def mentions_january_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    return (has_any_ci(text, ['january 2026', 'jan 2026', '2026-01', '01/2026']) or
            (has_any_ci(text, ['january', 'jan']) and has_any_ci(text, ['2026'])))


def mentions_san_francisco(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['san francisco', 'sf', 'san fran'])


def mentions_ai_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['ai agent', 'generative ai', 'gen ai', 'artificial intelligence'])


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
    Restrict evaluator.verify to at most one usage (we'll not use it here).
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
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_info(),
        template_class=EventInfo,
        extraction_name="event_info"
    )

    reddit_info = await evaluator.extract(
        prompt=prompt_extract_reddit_reputation(),
        template_class=RedditReputationInfo,
        extraction_name="reddit_reputation"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_info(),
        template_class=AirbnbInfo,
        extraction_name="airbnb_info"
    )

    distance_info = await evaluator.extract(
        prompt=prompt_extract_walking_distance(),
        template_class=WalkingDistanceInfo,
        extraction_name="walking_distance"
    )

    budget_info = await evaluator.extract(
        prompt=prompt_extract_budget_info(),
        template_class=BudgetInfo,
        extraction_name="budget_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite event search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date range filtering (January 2026)
    date_in_jan_2026 = mentions_january_2026(event_info.event_date if event_info else None) or mentions_january_2026(answer)
    evaluator.add_custom_node(
        result=bool(date_in_jan_2026),
        id="eventbrite_date_range",
        desc="[Action Node] eventbrite.com:F1:A1 - Filter events by date range (January 2026 or following two months)",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location filtering (San Francisco)
    location_sf = mentions_san_francisco(event_info.event_location if event_info else None) or mentions_san_francisco(answer)
    evaluator.add_custom_node(
        result=bool(location_sf),
        id="eventbrite_location_filter",
        desc="[Action Node] eventbrite.com:F1:A12 - Filter events by location (San Francisco)",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - List content understanding (AI-related keywords)
    ai_topic_ok = mentions_ai_keywords(event_info.event_name if event_info else None) or mentions_ai_keywords(answer)
    evaluator.add_custom_node(
        result=bool(ai_topic_ok),
        id="eventbrite_topic_understanding",
        desc="[Perception Node] eventbrite.com:F1:P3 - Identify events related to 'AI Agents' or 'Generative AI' from listing",
        parent=eventbrite_node,
        critical=False
    )

    # Check ticket price <= $500
    ticket_price_ok = is_within_500(event_info.ticket_price if event_info else None)
    evaluator.add_custom_node(
        result=bool(ticket_price_ok),
        id="eventbrite_price_check",
        desc="Verify ticket price is no more than $500",
        parent=eventbrite_node,
        critical=False
    )

    # Check in-person format mentioned
    in_person_ok = has_any_ci(answer, ['in-person', 'in person', 'physical', 'venue', 'location'])
    evaluator.add_custom_node(
        result=bool(in_person_ok),
        id="eventbrite_in_person",
        desc="Verify event is in-person (not virtual)",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 Reddit reputation check section
    reddit_node = evaluator.add_sequential(
        id="reddit_section",
        desc="Reddit reputation verification for event organizer",
        parent=root,
        critical=False
    )

    # [Action Node] reddit.com:F1:A1 - Search for organizer
    organizer_searched = (event_info and event_info.organizer and
                         has_any_ci(answer, ['reddit']) and
                         ci_contains(answer, event_info.organizer))
    evaluator.add_custom_node(
        result=bool(organizer_searched),
        id="reddit_search_organizer",
        desc="[Action Node] reddit.com:F1:A1 - Search Reddit for the organizer's name or event series",
        parent=reddit_node,
        critical=False
    )

    # [Perception Node] reddit.com:F1:P1 - Sentiment/content analysis
    reputation_checked = (reddit_info and reddit_info.reputation_summary and
                         has_any_ci(reddit_info.reputation_summary, ['positive', 'neutral', 'negative', 'scam', 'organized']))
    evaluator.add_custom_node(
        result=bool(reputation_checked),
        id="reddit_sentiment_analysis",
        desc="[Perception Node] reddit.com:F1:P1 - Analyze sentiment and check for serious negative reviews in the past year",
        parent=reddit_node,
        critical=False
    )

    # 3.3 Airbnb accommodation section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search and selection",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A2 - Date selection (day before event, 3 nights)
    checkin_correct = (airbnb_info and airbnb_info.nights == 3 and
                      has_any_ci(answer, ['day before', 'before the event', '3 nights', 'three nights']))
    evaluator.add_custom_node(
        result=bool(checkin_correct),
        id="airbnb_date_selection",
        desc="[Action Node] airbnb.com:F1:A2 - Select check-in date (day before event) and duration (3 nights)",
        parent=airbnb_node,
        critical=False
    )

    # Check "Entire home" property type
    entire_home_ok = has_any_ci(answer, ['entire home', 'entire place'])
    evaluator.add_custom_node(
        result=bool(entire_home_ok),
        id="airbnb_property_type",
        desc="Verify property is 'Entire home' type",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P1 - Price awareness (combined budget check)
    budget_ok = is_within_2000_budget(event_info.ticket_price if event_info else None,
                                      airbnb_info.total_accommodation_cost if airbnb_info else None)
    evaluator.add_custom_node(
        result=bool(budget_ok),
        id="airbnb_budget_check",
        desc="[Perception Node] airbnb.com:F1:P1 - Verify combined event ticket and accommodation cost is under $2,000",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Google Maps distance verification section
    maps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps walking distance verification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    maps_used = has_any_ci(answer, ['google maps', 'maps.google', 'walking distance'])
    evaluator.add_custom_node(
        result=bool(maps_used),
        id="googlemaps_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Input Airbnb and event venue addresses for walking route",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F2:P3 - Extract distance
    distance_extracted = looks_like_walking_distance(distance_info.walking_distance if distance_info else None)
    evaluator.add_custom_node(
        result=bool(distance_extracted),
        id="googlemaps_distance_extraction",
        desc="[Perception Node] maps.google.com:F2:P3 - Extract walking distance value from Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P8 - Verify distance is within 2 miles
    within_2miles = is_within_2_miles(distance_info.walking_distance if distance_info else None)
    evaluator.add_custom_node(
        result=bool(within_2miles),
        id="distance_within_2miles",
        desc="[Perception Node] eventbrite.com:F3:P8 - Verify walking distance is within 2 miles of event venue",
        parent=maps_node,
        critical=False
    )

    # 3.5 Output completeness check
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required information is present in the output",
        parent=root,
        critical=False
    )

    has_event_name = bool(event_info and event_info.event_name)
    evaluator.add_custom_node(
        result=has_event_name,
        id="output_event_name",
        desc="Output includes event name",
        parent=output_node,
        critical=False
    )

    has_event_date = bool(event_info and event_info.event_date)
    evaluator.add_custom_node(
        result=has_event_date,
        id="output_event_date",
        desc="Output includes event date",
        parent=output_node,
        critical=False
    )

    has_ticket_price = bool(event_info and event_info.ticket_price)
    evaluator.add_custom_node(
        result=has_ticket_price,
        id="output_ticket_price",
        desc="Output includes ticket price",
        parent=output_node,
        critical=False
    )

    has_organizer = bool(event_info and event_info.organizer)
    evaluator.add_custom_node(
        result=has_organizer,
        id="output_organizer",
        desc="Output includes organizer name",
        parent=output_node,
        critical=False
    )

    has_reddit_summary = bool(reddit_info and reddit_info.reputation_summary)
    evaluator.add_custom_node(
        result=has_reddit_summary,
        id="output_reddit_reputation",
        desc="Output includes Reddit reputation summary",
        parent=output_node,
        critical=False
    )

    has_airbnb_name = bool(airbnb_info and airbnb_info.listing_name)
    evaluator.add_custom_node(
        result=has_airbnb_name,
        id="output_airbnb_name",
        desc="Output includes Airbnb listing name",
        parent=output_node,
        critical=False
    )

    has_accom_cost = bool(airbnb_info and airbnb_info.total_accommodation_cost)
    evaluator.add_custom_node(
        result=has_accom_cost,
        id="output_accommodation_cost",
        desc="Output includes total accommodation cost",
        parent=output_node,
        critical=False
    )

    has_walking_distance = bool(distance_info and distance_info.walking_distance)
    evaluator.add_custom_node(
        result=has_walking_distance,
        id="output_walking_distance",
        desc="Output includes walking distance from Google Maps",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
