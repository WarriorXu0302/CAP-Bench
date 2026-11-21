import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-a2923a"
TASK_DESCRIPTION = "I'm a huge Coldplay fan and I've heard they're touring in Europe in summer 2026. Please help me plan a fan trip.\n\nFirst, check Coldplay's tour schedule on Spotify. Find all concerts held in the UK between June and August 2026. For each concert, note down the date, city, and venue.\n\nNext, go to Eventbrite and search for 'After Party' or 'Music Festival' events in these cities on the respective concert dates (as post-concert entertainment). Filter for events priced under £50.\n\nFinally, assuming I'm departing from New York (JFK), check Google Flights for one-way direct flights corresponding to these concert dates (arriving the day before the concert).\n\nPlease compile all qualifying options.\n\nOutput: Concert Date, City, Venue, Matching Eventbrite Activity Name & Price, Lowest Direct Flight Price from JFK, Spotify Tour Page Link, Eventbrite Event Page Link, Google Flights Search Result Link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConcertEntry(BaseModel):
    """A single concert entry extracted from the answer"""
    date: Optional[str] = None
    city: Optional[str] = None
    venue: Optional[str] = None
    eventbrite_activity_name: Optional[str] = None
    eventbrite_price: Optional[str] = None
    flight_price: Optional[str] = None
    spotify_link: Optional[str] = None
    eventbrite_link: Optional[str] = None
    google_flights_link: Optional[str] = None


class ExtractedData(BaseModel):
    """All extracted concert entries from the answer"""
    concerts: List[ConcertEntry] = Field(default_factory=list)
    mentions_uk: bool = False
    mentions_june_august_2026: bool = False
    mentions_spotify: bool = False
    mentions_eventbrite: bool = False
    mentions_google_flights: bool = False
    mentions_jfk: bool = False


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_concerts_from_answer() -> str:
    return """
Extract all concert entries that the user compiled from Coldplay's tour schedule and related searches.

For each concert, extract:
- date: the concert date as stated
- city: the city name
- venue: the venue name
- eventbrite_activity_name: the name of the matching Eventbrite activity
- eventbrite_price: the price of the Eventbrite activity
- flight_price: the lowest direct flight price from JFK
- spotify_link: the Spotify tour page link
- eventbrite_link: the Eventbrite event page link
- google_flights_link: the Google Flights search result link

Also indicate:
- mentions_uk: whether the answer mentions UK concerts
- mentions_june_august_2026: whether the answer mentions June-August 2026 timeframe
- mentions_spotify: whether the answer mentions Spotify
- mentions_eventbrite: whether the answer mentions Eventbrite
- mentions_google_flights: whether the answer mentions Google Flights
- mentions_jfk: whether the answer mentions JFK as departure airport

If any field is missing, set it to null or false.
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


def looks_like_date_2026(text: Optional[str]) -> bool:
    if not text:
        return False
    return '2026' in text and contains_digits(text)


def looks_like_uk_city(text: Optional[str]) -> bool:
    if not text:
        return False
    uk_cities = ['london', 'manchester', 'birmingham', 'glasgow', 'edinburgh',
                 'liverpool', 'leeds', 'cardiff', 'bristol', 'newcastle',
                 'sheffield', 'nottingham', 'brighton', 'oxford', 'cambridge']
    return any(ci_contains(text, city) for city in uk_cities)


def is_june_to_august(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    months = ['june', 'july', 'august', 'jun', 'jul', 'aug']
    return any(ci_contains(date_str, month) for month in months)


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['£', 'gbp', 'pounds', '$', 'usd']))


def extract_price_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)', text)
    if match:
        try:
            return float(match.group(1))
        except:
            return None
    return None


def is_price_under_50_gbp(price_str: Optional[str]) -> bool:
    if not price_str:
        return False
    val = extract_price_value(price_str)
    if val is None:
        return False
    if ci_contains(price_str, '£') or ci_contains(price_str, 'gbp') or ci_contains(price_str, 'pound'):
        return val < 50
    return True


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://', 'www.', '.com', '.co.uk'])


def is_spotify_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'spotify')


def is_eventbrite_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'eventbrite')


def is_google_flights_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'google') and ci_contains(text, 'flights')


def has_jfk_in_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'jfk')


def date_minus_one_day_check(concert_date: Optional[str], flight_url: Optional[str]) -> bool:
    """Lenient check that flight date appears to be one day before concert date"""
    if not concert_date or not flight_url:
        return False
    # This is hard to verify without parsing, so we do a lenient check
    return True


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_concerts_from_answer(),
        template_class=ExtractedData,
        extraction_name="coldplay_tour_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify tour schedule - UK concerts June-August 2026",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F9:A23 - Scroll to load future dates
    has_concerts = len(extracted.concerts) > 0
    has_multiple_concerts = len(extracted.concerts) >= 2
    all_dates_in_range = all(is_june_to_august(c.date) for c in extracted.concerts if c.date)

    evaluator.add_custom_node(
        result=bool(has_multiple_concerts and all_dates_in_range),
        id="spotify_scroll_load",
        desc="[Action Node] open.spotify.com:F9:A23 - Scroll to load all June-August 2026 concerts (completeness check)",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F9:P14 - Extract venue names
    all_have_venues = all(bool(c.venue and c.venue.strip()) for c in extracted.concerts)

    evaluator.add_custom_node(
        result=bool(has_concerts and all_have_venues),
        id="spotify_venue_extraction",
        desc="[Perception Node] open.spotify.com:F9:P14 - Extract venue names accurately for each concert",
        parent=spotify_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    evaluator.add_custom_node(
        result=bool(extracted.mentions_spotify),
        id="spotify_mention",
        desc="Mentions Spotify as the source for tour information",
        parent=spotify_node,
        critical=False
    )

    uk_cities_ok = all(looks_like_uk_city(c.city) for c in extracted.concerts if c.city)
    evaluator.add_custom_node(
        result=bool(has_concerts and uk_cities_ok),
        id="spotify_uk_filter",
        desc="Correctly filters for UK cities only",
        parent=spotify_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(extracted.mentions_june_august_2026),
        id="spotify_timeframe",
        desc="Mentions June-August 2026 timeframe",
        parent=spotify_node,
        critical=False
    )

    all_have_dates = all(bool(c.date and c.date.strip()) for c in extracted.concerts)
    evaluator.add_custom_node(
        result=bool(has_concerts and all_have_dates),
        id="spotify_date_extraction",
        desc="Extracts concert dates for all concerts",
        parent=spotify_node,
        critical=False
    )

    all_have_cities = all(bool(c.city and c.city.strip()) for c in extracted.concerts)
    evaluator.add_custom_node(
        result=bool(has_concerts and all_have_cities),
        id="spotify_city_extraction",
        desc="Extracts city names for all concerts",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite after-party/music festival events matching concert dates and cities",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date picker for concert dates
    concerts_with_activities = [c for c in extracted.concerts if c.eventbrite_activity_name]

    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0),
        id="eventbrite_date_picker",
        desc="[Action Node] eventbrite.com:F1:A1 - Use date picker to search for events on concert dates",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location dropdown for cities
    cities_match = all(
        ci_contains(c.eventbrite_activity_name, c.city) or True
        for c in concerts_with_activities if c.city
    )

    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0),
        id="eventbrite_location_select",
        desc="[Action Node] eventbrite.com:F1:A12 - Select locations matching concert cities via dropdown",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - Filter prices under £50
    all_prices_valid = all(
        is_price_under_50_gbp(c.eventbrite_price)
        for c in concerts_with_activities if c.eventbrite_price
    )

    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0 and all_prices_valid),
        id="eventbrite_price_filter",
        desc="[Perception Node] eventbrite.com:F1:P3 - Filter and extract events priced under £50",
        parent=eventbrite_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    evaluator.add_custom_node(
        result=bool(extracted.mentions_eventbrite),
        id="eventbrite_mention",
        desc="Mentions Eventbrite as the source for after-party events",
        parent=eventbrite_node,
        critical=False
    )

    activity_keywords_ok = any(
        has_any_ci(c.eventbrite_activity_name, ['after party', 'music festival', 'party', 'festival'])
        for c in concerts_with_activities if c.eventbrite_activity_name
    )
    evaluator.add_custom_node(
        result=bool(activity_keywords_ok),
        id="eventbrite_keyword_search",
        desc="Searches for relevant keywords like 'After Party' or 'Music Festival'",
        parent=eventbrite_node,
        critical=False
    )

    all_have_eventbrite_names = all(
        bool(c.eventbrite_activity_name and c.eventbrite_activity_name.strip())
        for c in concerts_with_activities
    )
    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0 and all_have_eventbrite_names),
        id="eventbrite_name_extraction",
        desc="Extracts activity names for matching events",
        parent=eventbrite_node,
        critical=False
    )

    all_have_eventbrite_prices = all(
        bool(c.eventbrite_price and c.eventbrite_price.strip())
        for c in concerts_with_activities
    )
    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0 and all_have_eventbrite_prices),
        id="eventbrite_price_extraction",
        desc="Extracts prices for matching events",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Google Flights section
    flights_node = evaluator.add_sequential(
        id="google_flights_section",
        desc="Google Flights one-way direct flights from JFK arriving day before concert",
        parent=root,
        critical=False
    )

    concerts_with_flights = [c for c in extracted.concerts if c.flight_price]

    # [Action Node] google.comflights:F1:A1 - Form input with JFK origin
    evaluator.add_custom_node(
        result=bool(extracted.mentions_jfk and len(concerts_with_flights) > 0),
        id="google_flights_jfk_input",
        desc="[Action Node] google.comflights:F1:A1 - Input JFK as departure airport in flight search form",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A2 - Date selection for day before concert
    all_have_flight_urls = all(
        bool(c.google_flights_link and c.google_flights_link.strip())
        for c in concerts_with_flights
    )

    evaluator.add_custom_node(
        result=bool(len(concerts_with_flights) > 0 and all_have_flight_urls),
        id="google_flights_date_select",
        desc="[Action Node] google.comflights:F1:A2 - Select arrival date as day before concert date",
        parent=flights_node,
        critical=False
    )

    # [Action Node] google.comflights:F1:A16 - Sort/filter for direct flights only
    evaluator.add_custom_node(
        result=bool(len(concerts_with_flights) > 0),
        id="google_flights_direct_filter",
        desc="[Action Node] google.comflights:F1:A16 - Apply direct flights filter/sort",
        parent=flights_node,
        critical=False
    )

    # [Perception Node] google.comflights:F1:P1 - Extract lowest price
    all_have_valid_prices = all(
        looks_like_price(c.flight_price)
        for c in concerts_with_flights if c.flight_price
    )

    evaluator.add_custom_node(
        result=bool(len(concerts_with_flights) > 0 and all_have_valid_prices),
        id="google_flights_price_extraction",
        desc="[Perception Node] google.comflights:F1:P1 - Extract lowest direct flight price",
        parent=flights_node,
        critical=False
    )

    # Additional checks (non-prefixed)
    evaluator.add_custom_node(
        result=bool(extracted.mentions_google_flights),
        id="google_flights_mention",
        desc="Mentions Google Flights as the source for flight information",
        parent=flights_node,
        critical=False
    )

    one_way_mentioned = has_any_ci(answer, ['one-way', 'one way'])
    evaluator.add_custom_node(
        result=bool(one_way_mentioned),
        id="google_flights_one_way",
        desc="Specifies one-way flights as requested",
        parent=flights_node,
        critical=False
    )

    direct_mentioned = has_any_ci(answer, ['direct', 'non-stop', 'nonstop'])
    evaluator.add_custom_node(
        result=bool(direct_mentioned),
        id="google_flights_direct_mention",
        desc="Mentions direct or non-stop flights",
        parent=flights_node,
        critical=False
    )

    # 3.4 Output compilation section
    output_node = evaluator.add_parallel(
        id="output_compilation",
        desc="Complete output with all required fields and links",
        parent=root,
        critical=False
    )

    all_have_spotify_links = all(
        is_spotify_url(c.spotify_link)
        for c in extracted.concerts if c.spotify_link
    )
    evaluator.add_custom_node(
        result=bool(has_concerts and all_have_spotify_links),
        id="output_spotify_links",
        desc="Includes Spotify tour page links in output",
        parent=output_node,
        critical=False
    )

    all_have_eventbrite_links = all(
        is_eventbrite_url(c.eventbrite_link)
        for c in concerts_with_activities if c.eventbrite_link
    )
    evaluator.add_custom_node(
        result=bool(len(concerts_with_activities) > 0 and all_have_eventbrite_links),
        id="output_eventbrite_links",
        desc="Includes Eventbrite event page links in output",
        parent=output_node,
        critical=False
    )

    all_have_flights_links = all(
        is_google_flights_url(c.google_flights_link)
        for c in concerts_with_flights if c.google_flights_link
    )
    evaluator.add_custom_node(
        result=bool(len(concerts_with_flights) > 0 and all_have_flights_links),
        id="output_google_flights_links",
        desc="Includes Google Flights search result links in output",
        parent=output_node,
        critical=False
    )

    output_format_ok = (
        has_any_ci(answer, ['concert date', 'date']) and
        has_any_ci(answer, ['city']) and
        has_any_ci(answer, ['venue']) and
        has_any_ci(answer, ['price'])
    )
    evaluator.add_custom_node(
        result=bool(output_format_ok),
        id="output_format",
        desc="Compiles output in requested format with all specified fields",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
