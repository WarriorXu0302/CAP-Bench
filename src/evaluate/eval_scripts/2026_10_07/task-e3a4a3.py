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
TASK_ID = "task-e3a4a3"
TASK_DESCRIPTION = 'I plan to attend an electronic music festival and would like to find a complete travel itinerary.\n\nFirst, on Spotify, identify 3 popular electronic music artists (Techno or House genres) with recent tour schedules. Record their names and tour information.\n\nNext, search on Eventbrite for performances by these artists in US West Coast cities (Los Angeles, San Francisco, Seattle) during an upcoming period suitable for festivals. Find 2 well-rated festival events with tickets still available.\n\nThen, on Airbnb, search for accommodations within 5 miles of the event venues. Filter for listings with a rating of 4.5 stars or higher and priced between $150-$300 per night. Identify 3 suitable options.\n\nFinally, use Google Maps to verify the actual distance and driving time (by car) from each accommodation to its corresponding festival venue, confirming that it is indeed within 5 miles and has a reasonable commute time.\n\nOutput for each event: Artist Name, Festival Name, Date and Time, Venue Name and Address, Ticket Price, Eventbrite Link.\nOutput for each accommodation: Listing Name, Price/Night, Rating, Actual Distance (miles) to Venue, Driving Time, Airbnb Link, Google Maps Route Link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SpotifyArtists(BaseModel):
    """Artists extracted from Spotify"""
    artist1_name: Optional[str] = None
    artist1_tour_info: Optional[str] = None
    artist2_name: Optional[str] = None
    artist2_tour_info: Optional[str] = None
    artist3_name: Optional[str] = None
    artist3_tour_info: Optional[str] = None


class EventbriteEvents(BaseModel):
    """Festival events extracted from Eventbrite"""
    event1_artist: Optional[str] = None
    event1_name: Optional[str] = None
    event1_datetime: Optional[str] = None
    event1_venue_name: Optional[str] = None
    event1_venue_address: Optional[str] = None
    event1_ticket_price: Optional[str] = None
    event1_link: Optional[str] = None
    event2_artist: Optional[str] = None
    event2_name: Optional[str] = None
    event2_datetime: Optional[str] = None
    event2_venue_name: Optional[str] = None
    event2_venue_address: Optional[str] = None
    event2_ticket_price: Optional[str] = None
    event2_link: Optional[str] = None


class AirbnbListings(BaseModel):
    """Accommodations extracted from Airbnb"""
    listing1_name: Optional[str] = None
    listing1_price: Optional[str] = None
    listing1_rating: Optional[str] = None
    listing1_distance: Optional[str] = None
    listing1_drive_time: Optional[str] = None
    listing1_link: Optional[str] = None
    listing1_maps_link: Optional[str] = None
    listing2_name: Optional[str] = None
    listing2_price: Optional[str] = None
    listing2_rating: Optional[str] = None
    listing2_distance: Optional[str] = None
    listing2_drive_time: Optional[str] = None
    listing2_link: Optional[str] = None
    listing2_maps_link: Optional[str] = None
    listing3_name: Optional[str] = None
    listing3_price: Optional[str] = None
    listing3_rating: Optional[str] = None
    listing3_distance: Optional[str] = None
    listing3_drive_time: Optional[str] = None
    listing3_link: Optional[str] = None
    listing3_maps_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_spotify_artists() -> str:
    return """
Extract the 3 electronic music artists (Techno or House genres) with tour schedules from the answer.

Return:
- artist1_name: name of the first artist
- artist1_tour_info: tour information for the first artist
- artist2_name: name of the second artist
- artist2_tour_info: tour information for the second artist
- artist3_name: name of the third artist
- artist3_tour_info: tour information for the third artist

If any field is missing, set it to null.
"""


def prompt_extract_eventbrite_events() -> str:
    return """
Extract the 2 festival events from Eventbrite with all requested details from the answer.

Return for each event:
- eventN_artist: artist name
- eventN_name: festival name
- eventN_datetime: date and time
- eventN_venue_name: venue name
- eventN_venue_address: venue address
- eventN_ticket_price: ticket price
- eventN_link: Eventbrite link

If any field is missing, set it to null.
"""


def prompt_extract_airbnb_listings() -> str:
    return """
Extract the 3 accommodation listings from Airbnb with all requested details from the answer.

Return for each listing:
- listingN_name: listing name
- listingN_price: price per night
- listingN_rating: rating
- listingN_distance: actual distance to venue in miles
- listingN_drive_time: driving time
- listingN_link: Airbnb link
- listingN_maps_link: Google Maps route link

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


def looks_like_url(text: Optional[str], domain: str = None) -> bool:
    if not text:
        return False
    url_pattern = r'https?://'
    has_url = bool(re.search(url_pattern, text))
    if domain:
        return has_url and ci_contains(text, domain)
    return has_url


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '$') or ci_contains(text, 'dollar'))


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km'])


def looks_like_time_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour', 'hr'])


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
    spotify_info = await evaluator.extract(
        prompt=prompt_extract_spotify_artists(),
        template_class=SpotifyArtists,
        extraction_name="spotify_artists"
    )

    eventbrite_info = await evaluator.extract(
        prompt=prompt_extract_eventbrite_events(),
        template_class=EventbriteEvents,
        extraction_name="eventbrite_events"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_listings(),
        template_class=AirbnbListings,
        extraction_name="airbnb_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify - Identify 3 electronic music artists with tour schedules",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F2:A2 - Navigate to electronic music category
    spotify_mentions_electronic = has_any_ci(answer, ['electronic', 'techno', 'house'])
    spotify_has_artists = (spotify_info.artist1_name and spotify_info.artist2_name and spotify_info.artist3_name)
    evaluator.add_custom_node(
        result=bool(spotify_mentions_electronic and spotify_has_artists),
        id="spotify_action_category_navigation",
        desc="[Action Node] open.spotify.com:F2:A2 - Navigate to electronic music category and identify artists",
        parent=spotify_node,
        critical=False
    )

    # [Action Node] open.spotify.com:F9:A16 - View concert information
    has_tour_info = (spotify_info.artist1_tour_info and spotify_info.artist2_tour_info and spotify_info.artist3_tour_info)
    evaluator.add_custom_node(
        result=bool(has_tour_info),
        id="spotify_action_view_tour_dates",
        desc="[Action Node] open.spotify.com:F9:A16 - View tour dates and concert information for artists",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F9:P5 - Identify concert visual information
    evaluator.add_custom_node(
        result=bool(has_tour_info),
        id="spotify_perception_concert_visuals",
        desc="[Perception Node] open.spotify.com:F9:P5 - Identify concert promotional images and activity type",
        parent=spotify_node,
        critical=False
    )

    # [Perception Node] open.spotify.com:F9:P14 - Understand concert list information
    tour_info_structured = has_tour_info and (
        has_any_ci(answer, ['tour', 'date', 'venue', 'city']) or
        any([ci_contains(str(info), 'tour') or ci_contains(str(info), 'date')
             for info in [spotify_info.artist1_tour_info, spotify_info.artist2_tour_info, spotify_info.artist3_tour_info]
             if info])
    )
    evaluator.add_custom_node(
        result=bool(tour_info_structured),
        id="spotify_perception_tour_list",
        desc="[Perception Node] open.spotify.com:F9:P14 - Understand tour dates list with date, venue, and city information",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite - Find 2 festival events on US West Coast",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date range filtering
    has_event_dates = bool(eventbrite_info.event1_datetime and eventbrite_info.event2_datetime)
    evaluator.add_custom_node(
        result=bool(has_event_dates),
        id="eventbrite_action_date_filter",
        desc="[Action Node] eventbrite.com:F1:A1 - Select date range in date picker for festival period",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location selection
    west_coast_cities = ['los angeles', 'san francisco', 'seattle', 'la', 'sf']
    has_west_coast_venue = any([
        has_any_ci(eventbrite_info.event1_venue_address, west_coast_cities),
        has_any_ci(eventbrite_info.event2_venue_address, west_coast_cities)
    ])
    evaluator.add_custom_node(
        result=bool(has_west_coast_venue),
        id="eventbrite_action_location_select",
        desc="[Action Node] eventbrite.com:F1:A12 - Select West Coast cities in location dropdown",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P4 - Identify date status
    evaluator.add_custom_node(
        result=bool(has_event_dates),
        id="eventbrite_perception_date_status",
        desc="[Perception Node] eventbrite.com:F1:P4 - Identify selectable dates and highlight status in calendar",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A2 - Event format filtering
    has_festival_keywords = has_any_ci(answer, ['festival', 'fest', 'music festival'])
    evaluator.add_custom_node(
        result=bool(has_festival_keywords),
        id="eventbrite_action_format_filter",
        desc="[Action Node] eventbrite.com:F2:A2 - Select Festival format in Format filter",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - Click event card to view details
    has_eventbrite_links = bool(eventbrite_info.event1_link and eventbrite_info.event2_link)
    links_valid = (looks_like_url(eventbrite_info.event1_link, 'eventbrite') and
                   looks_like_url(eventbrite_info.event2_link, 'eventbrite'))
    evaluator.add_custom_node(
        result=bool(has_eventbrite_links and links_valid),
        id="eventbrite_action_click_card",
        desc="[Action Node] eventbrite.com:F3:A8 - Click event cards to view details page",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P1 - Identify event status markers
    mentions_tickets_available = has_any_ci(answer, ['ticket', 'available', 'not sold out'])
    evaluator.add_custom_node(
        result=bool(mentions_tickets_available),
        id="eventbrite_perception_status_markers",
        desc="[Perception Node] eventbrite.com:F3:P1 - Identify event status markers like Going fast, Sold Out",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F3:P2 - Identify event image content
    evaluator.add_custom_node(
        result=bool(has_festival_keywords),
        id="eventbrite_perception_image_content",
        desc="[Perception Node] eventbrite.com:F3:P2 - Identify music activity from event cover images",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - Understand event list information
    has_complete_event_info = bool(
        eventbrite_info.event1_name and eventbrite_info.event1_datetime and
        eventbrite_info.event1_venue_name and eventbrite_info.event2_name and
        eventbrite_info.event2_datetime and eventbrite_info.event2_venue_name
    )
    evaluator.add_custom_node(
        result=bool(has_complete_event_info),
        id="eventbrite_perception_list_info",
        desc="[Perception Node] eventbrite.com:F1:P3 - Understand comprehensive event information from list",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb - Find 3 accommodations near festival venues",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date range selection
    has_listing_dates = has_event_dates
    evaluator.add_custom_node(
        result=bool(has_listing_dates),
        id="airbnb_action_date_select",
        desc="[Action Node] airbnb.com:F1:A6 - Select check-in and check-out dates matching festival dates",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Search destination filtering
    has_venue_addresses = bool(eventbrite_info.event1_venue_address or eventbrite_info.event2_venue_address)
    evaluator.add_custom_node(
        result=bool(has_venue_addresses),
        id="airbnb_action_destination_search",
        desc="[Action Node] airbnb.com:F1:A10 - Enter venue address or name in destination input",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price range slider
    has_prices = bool(airbnb_info.listing1_price and airbnb_info.listing2_price and airbnb_info.listing3_price)
    prices_valid = all([
        extract_float(p) is not None and 150 <= extract_float(p) <= 300
        for p in [airbnb_info.listing1_price, airbnb_info.listing2_price, airbnb_info.listing3_price]
        if p and extract_float(p) is not None
    ])
    evaluator.add_custom_node(
        result=bool(has_prices and prices_valid),
        id="airbnb_action_price_slider",
        desc="[Action Node] airbnb.com:F2:A13 - Set price range slider to $150-$300 per night",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A34 - Open filter drawer
    mentions_filters = has_any_ci(answer, ['filter', 'rating', 'price'])
    evaluator.add_custom_node(
        result=bool(mentions_filters),
        id="airbnb_action_open_filter",
        desc="[Action Node] airbnb.com:F2:A34 - Click filter button to open filter panel",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A35 - Set filter conditions in drawer
    has_ratings = bool(airbnb_info.listing1_rating and airbnb_info.listing2_rating and airbnb_info.listing3_rating)
    ratings_valid = all([
        extract_float(r) is not None and extract_float(r) >= 4.5
        for r in [airbnb_info.listing1_rating, airbnb_info.listing2_rating, airbnb_info.listing3_rating]
        if r and extract_float(r) is not None
    ])
    evaluator.add_custom_node(
        result=bool(has_ratings and ratings_valid),
        id="airbnb_action_set_filters",
        desc="[Action Node] airbnb.com:F2:A35 - Set rating and price filters in filter drawer",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F3:A19 - Click listing card to view details
    has_airbnb_links = bool(airbnb_info.listing1_link and airbnb_info.listing2_link and airbnb_info.listing3_link)
    links_valid = all([
        looks_like_url(link, 'airbnb')
        for link in [airbnb_info.listing1_link, airbnb_info.listing2_link, airbnb_info.listing3_link]
        if link
    ])
    evaluator.add_custom_node(
        result=bool(has_airbnb_links and links_valid),
        id="airbnb_action_click_listing",
        desc="[Action Node] airbnb.com:F3:A19 - Click listing cards to view detailed information",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F3:P1 - Identify listing image features
    has_listing_names = bool(airbnb_info.listing1_name and airbnb_info.listing2_name and airbnb_info.listing3_name)
    evaluator.add_custom_node(
        result=bool(has_listing_names),
        id="airbnb_perception_image_features",
        desc="[Perception Node] airbnb.com:F3:P1 - Identify housing type and decor from listing images",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F2:P3 - Identify listing status markers
    evaluator.add_custom_node(
        result=bool(has_ratings),
        id="airbnb_perception_status_markers",
        desc="[Perception Node] airbnb.com:F2:P3 - Identify rating, Guest Favorite and other markers on listing cards",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F2:P10 - Understand listing list information
    has_complete_listing_info = bool(has_listing_names and has_prices and has_ratings)
    evaluator.add_custom_node(
        result=bool(has_complete_listing_info),
        id="airbnb_perception_list_info",
        desc="[Perception Node] airbnb.com:F2:P10 - Understand comprehensive listing information from list",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A36 - Map zoom and pan
    has_distances = bool(airbnb_info.listing1_distance and airbnb_info.listing2_distance and airbnb_info.listing3_distance)
    evaluator.add_custom_node(
        result=bool(has_distances),
        id="airbnb_action_map_zoom",
        desc="[Action Node] airbnb.com:F4:A36 - Adjust map view to see listings within 5 mile radius",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F4:P16 - Understand listing map position
    distances_valid = all([
        looks_like_distance(d) and extract_float(d) is not None and extract_float(d) <= 5
        for d in [airbnb_info.listing1_distance, airbnb_info.listing2_distance, airbnb_info.listing3_distance]
        if d
    ])
    evaluator.add_custom_node(
        result=bool(has_distances and distances_valid),
        id="airbnb_perception_map_position",
        desc="[Perception Node] airbnb.com:F4:P16 - Understand listing position relative to venue on map",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Google Maps section
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps - Verify actual distance and driving time",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input origin and destination
    has_maps_links = bool(airbnb_info.listing1_maps_link and airbnb_info.listing2_maps_link and airbnb_info.listing3_maps_link)
    maps_links_valid = all([
        looks_like_url(link, 'google.com/maps')
        for link in [airbnb_info.listing1_maps_link, airbnb_info.listing2_maps_link, airbnb_info.listing3_maps_link]
        if link
    ])
    evaluator.add_custom_node(
        result=bool(has_maps_links and maps_links_valid),
        id="googlemaps_action_input_route",
        desc="[Action Node] maps.google.com:F2:A5 - Input accommodation and venue addresses in route planner",
        parent=googlemaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select transportation mode
    has_drive_times = bool(airbnb_info.listing1_drive_time and airbnb_info.listing2_drive_time and airbnb_info.listing3_drive_time)
    drive_times_valid = all([
        looks_like_time_duration(t)
        for t in [airbnb_info.listing1_drive_time, airbnb_info.listing2_drive_time, airbnb_info.listing3_drive_time]
        if t
    ])
    mentions_driving = has_any_ci(answer, ['driv', 'car'])
    evaluator.add_custom_node(
        result=bool(has_drive_times and drive_times_valid and mentions_driving),
        id="googlemaps_action_select_driving",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode for transportation",
        parent=googlemaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Place name search
    evaluator.add_custom_node(
        result=bool(has_venue_addresses),
        id="googlemaps_action_place_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search venue names and addresses for location",
        parent=googlemaps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P12 - Identify map markers
    evaluator.add_custom_node(
        result=bool(has_maps_links),
        id="googlemaps_perception_map_markers",
        desc="[Perception Node] maps.google.com:F3:P12 - Identify origin and destination markers on map",
        parent=googlemaps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
