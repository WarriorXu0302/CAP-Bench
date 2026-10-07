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
TASK_ID = "task-dcdac4"
TASK_DESCRIPTION = "I plan to follow my favorite electronic music artist Peggy Gou's tour during an upcoming summer season, aiming to plan a comprehensive music festival trip.\n\nFirst, help me confirm Peggy Gou's artist page on Spotify to see if she has listed any tour information or social media links.\n\nThen, search Eventbrite for her North American performances during an upcoming summer season (e.g., June-August). Find 3 dates in different cities, with ticket prices ranging from $50 to $150 per event.\n\nNext, use Google Maps to pinpoint the exact locations of these 3 performance venues. Plan a walking route from each venue to its surrounding area (ensuring accommodation and dining options are available within a 2 km radius).\n\nThen, using each venue's address as the center, search Airbnb for accommodation options. Filter for listings within 2 km, offering free cancellation, rated 4.5 stars or higher, and priced under $150 per night. Find 2 alternative options for each city.\n\nFinally, on Yelp, search for Late Night Dining or Live Music restaurants/bars within a 20-minute walk of each venue. Filter for places rated 4 stars or higher, open past midnight, and with a price range of $$-$$$. Find 2 recommendations for each venue.\n\nOutput a complete itinerary table including: City Name, Performance Date, Eventbrite Event Link, Venue Name, Venue Address, Ticket Price Range, 2 Airbnb Listings (including listing name, price, rating, cancellation policy, Airbnb link), 2 Yelp Recommended Restaurants (including restaurant name, rating, price level, operating hours, Yelp link), and the Google Maps location link for each venue."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event information for each city"""
    city_name: Optional[str] = None
    performance_date: Optional[str] = None
    eventbrite_link: Optional[str] = None
    venue_name: Optional[str] = None
    venue_address: Optional[str] = None
    ticket_price_range: Optional[str] = None
    google_maps_link: Optional[str] = None


class AirbnbListing(BaseModel):
    """Airbnb listing details"""
    listing_name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    cancellation_policy: Optional[str] = None
    airbnb_link: Optional[str] = None


class YelpRestaurant(BaseModel):
    """Yelp restaurant details"""
    restaurant_name: Optional[str] = None
    rating: Optional[str] = None
    price_level: Optional[str] = None
    operating_hours: Optional[str] = None
    yelp_link: Optional[str] = None


class ItineraryData(BaseModel):
    """Complete itinerary data extracted from answer"""
    events: Optional[List[EventInfo]] = Field(default_factory=list)
    airbnb_listings_by_city: Optional[Dict[str, List[AirbnbListing]]] = Field(default_factory=dict)
    yelp_restaurants_by_city: Optional[Dict[str, List[YelpRestaurant]]] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary() -> str:
    return """
Extract the complete itinerary information from the answer for Peggy Gou's tour.

Return:
- events: a list of event information containing city_name, performance_date, eventbrite_link, venue_name, venue_address, ticket_price_range, and google_maps_link for each of the 3 cities
- airbnb_listings_by_city: a dictionary mapping city names to lists of 2 Airbnb listings, each containing listing_name, price, rating, cancellation_policy, and airbnb_link
- yelp_restaurants_by_city: a dictionary mapping city names to lists of 2 Yelp restaurants, each containing restaurant_name, rating, price_level, operating_hours, and yelp_link

If any field is missing, set it to null or empty list/dict as appropriate.
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


def looks_like_north_american_city(city: Optional[str]) -> bool:
    if not city:
        return False
    # Simple heuristic: any city name is acceptable
    return len(city.strip()) > 0


def looks_like_summer_date(date: Optional[str]) -> bool:
    if not date:
        return False
    # Check for June, July, August or months 6, 7, 8
    return has_any_ci(date, ['june', 'july', 'august', 'jun', 'jul', 'aug', '/6/', '/7/', '/8/', '-06-', '-07-', '-08-'])


def price_in_range(price_text: Optional[str], min_val: float, max_val: float) -> bool:
    if not price_text:
        return False
    num = extract_float(price_text)
    if num is None:
        return False
    return min_val <= num <= max_val


def rating_meets_threshold(rating_text: Optional[str], threshold: float) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    if num is None:
        return False
    return num >= threshold


def is_free_cancellation(policy: Optional[str]) -> bool:
    if not policy:
        return False
    return has_any_ci(policy, ['free', 'cancellation'])


def is_dollar_sign_level(price_level: Optional[str], expected_signs: List[str]) -> bool:
    if not price_level:
        return False
    return any(signs in price_level for signs in expected_signs)


def is_open_past_midnight(hours: Optional[str]) -> bool:
    if not hours:
        return False
    # Check for times past 12:00 AM or midnight
    return has_any_ci(hours, ['midnight', '12:00 am', '1:00 am', '2:00 am', '00:', '01:', '02:'])


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
    itinerary = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryData,
        extraction_name="itinerary_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Spotify section
    spotify_node = evaluator.add_sequential(
        id="spotify_section",
        desc="Spotify artist page confirmation for Peggy Gou",
        parent=root,
        critical=False
    )

    # [Action Node] open.spotify.com:F3:A9
    spotify_action_ok = has_any_ci(answer, ['spotify']) and has_any_ci(answer, ['peggy gou'])
    evaluator.add_custom_node(
        result=bool(spotify_action_ok),
        id="spotify_action_artist_page",
        desc="[Action Node] open.spotify.com:F3:A9 - Navigate to Peggy Gou's artist page on Spotify",
        parent=spotify_node,
        critical=False
    )

    # 3.2 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite search for 3 North American performances",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date range filter
    events = itinerary.events if itinerary and itinerary.events else []
    has_summer_dates = any(looks_like_summer_date(e.performance_date) for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_summer_dates),
        id="eventbrite_action_date_filter",
        desc="[Action Node] eventbrite.com:F1:A1 - Filter events by summer date range (June-August)",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A12 - Location dropdown filter
    has_north_american_cities = any(looks_like_north_american_city(e.city_name) for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_north_american_cities and len(events) >= 3),
        id="eventbrite_action_location_filter",
        desc="[Action Node] eventbrite.com:F1:A12 - Filter by North American location",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F2:A3 - Price filter
    has_valid_price_range = any(price_in_range(e.ticket_price_range, 50, 150) for e in events if e and e.ticket_price_range)
    evaluator.add_custom_node(
        result=bool(has_valid_price_range),
        id="eventbrite_action_price_filter",
        desc="[Action Node] eventbrite.com:F2:A3 - Filter by price range $50-$150",
        parent=eventbrite_node,
        critical=False
    )

    # [Action Node] eventbrite.com:F3:A8 - Click event card
    has_venue_names = any(e.venue_name for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_venue_names),
        id="eventbrite_action_event_details",
        desc="[Action Node] eventbrite.com:F3:A8 - Click event cards to view venue details",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P1 - Event status awareness
    has_eventbrite_links = any(e.eventbrite_link for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_eventbrite_links),
        id="eventbrite_perception_event_status",
        desc="[Perception Node] eventbrite.com:F1:P1 - Identify available (non-sold-out) events",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - Event list content understanding
    has_complete_event_info = len(events) >= 3 and all(
        e.city_name and e.performance_date and e.ticket_price_range
        for e in events[:3] if e
    )
    evaluator.add_custom_node(
        result=bool(has_complete_event_info),
        id="eventbrite_perception_event_list",
        desc="[Perception Node] eventbrite.com:F1:P3 - Extract city, date, and price information from event list",
        parent=eventbrite_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps venue location and walking route planning",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search
    has_venue_addresses = any(e.venue_address for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_venue_addresses),
        id="maps_action_venue_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for venue locations",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    mentions_walking_route = has_any_ci(answer, ['walking', 'route', '2 km'])
    evaluator.add_custom_node(
        result=bool(mentions_walking_route),
        id="maps_action_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Plan walking routes from venues",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select walking mode
    evaluator.add_custom_node(
        result=bool(mentions_walking_route),
        id="maps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking transportation mode",
        parent=maps_node,
        critical=False
    )

    has_maps_links = any(e.google_maps_link for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_maps_links),
        id="maps_output_venue_links",
        desc="Output includes Google Maps links for venues",
        parent=maps_node,
        critical=False
    )

    # 3.4 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb accommodation search with filters",
        parent=root,
        critical=False
    )

    airbnb_listings = itinerary.airbnb_listings_by_city if itinerary and itinerary.airbnb_listings_by_city else {}
    all_airbnb_listings = [listing for listings in airbnb_listings.values() for listing in listings if listing]

    # [Action Node] airbnb.com:F1:A6 - Date selection
    has_date_context = any(e.performance_date for e in events if e)
    evaluator.add_custom_node(
        result=bool(has_date_context),
        id="airbnb_action_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Select dates matching performance dates",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Destination search
    has_venue_based_search = has_venue_addresses and len(all_airbnb_listings) > 0
    evaluator.add_custom_node(
        result=bool(has_venue_based_search),
        id="airbnb_action_destination_search",
        desc="[Action Node] airbnb.com:F1:A10 - Search using venue addresses as center points",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A38 - Map-based location filtering
    mentions_2km_radius = has_any_ci(answer, ['2 km', '2km', 'within 2'])
    evaluator.add_custom_node(
        result=bool(mentions_2km_radius),
        id="airbnb_action_map_location",
        desc="[Action Node] airbnb.com:F1:A38 - Use map to verify listings within 2 km radius",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Free cancellation filter
    has_free_cancel = any(is_free_cancellation(l.cancellation_policy) for l in all_airbnb_listings)
    evaluator.add_custom_node(
        result=bool(has_free_cancel),
        id="airbnb_action_cancellation_filter",
        desc="[Action Node] airbnb.com:F2:A11 - Filter for free cancellation",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price slider
    has_price_under_150 = any(price_in_range(l.price, 0, 150) for l in all_airbnb_listings if l.price)
    evaluator.add_custom_node(
        result=bool(has_price_under_150),
        id="airbnb_action_price_slider",
        desc="[Action Node] airbnb.com:F2:A13 - Set price filter to under $150/night",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A34 - Open filter drawer
    has_multiple_filters = has_free_cancel and has_price_under_150
    evaluator.add_custom_node(
        result=bool(has_multiple_filters),
        id="airbnb_action_open_filters",
        desc="[Action Node] airbnb.com:F2:A34 - Open filter panel for multiple criteria",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A35 - Apply multiple filters in drawer
    has_rating_filter = any(rating_meets_threshold(l.rating, 4.5) for l in all_airbnb_listings if l.rating)
    evaluator.add_custom_node(
        result=bool(has_rating_filter and has_free_cancel and has_price_under_150),
        id="airbnb_action_drawer_filters",
        desc="[Action Node] airbnb.com:F2:A35 - Apply rating, cancellation, and price filters",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F3:A19 - Click listing card
    has_detailed_info = any(l.cancellation_policy and l.airbnb_link for l in all_airbnb_listings)
    evaluator.add_custom_node(
        result=bool(has_detailed_info),
        id="airbnb_action_listing_details",
        desc="[Action Node] airbnb.com:F3:A19 - Click listing cards to view details",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P3 - Guest Favorite and rating awareness
    evaluator.add_custom_node(
        result=bool(has_rating_filter),
        id="airbnb_perception_rating_status",
        desc="[Perception Node] airbnb.com:F1:P3 - Identify listings with 4.5+ star ratings",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F2:P10 - Listing content understanding
    has_complete_airbnb_info = len(all_airbnb_listings) >= 6 and all(
        l.listing_name and l.price and l.rating and l.cancellation_policy
        for l in all_airbnb_listings[:6]
    )
    evaluator.add_custom_node(
        result=bool(has_complete_airbnb_info),
        id="airbnb_perception_listing_content",
        desc="[Perception Node] airbnb.com:F2:P10 - Extract comprehensive listing information",
        parent=airbnb_node,
        critical=False
    )

    # 3.5 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp restaurant search with filters",
        parent=root,
        critical=False
    )

    yelp_restaurants = itinerary.yelp_restaurants_by_city if itinerary and itinerary.yelp_restaurants_by_city else {}
    all_yelp_restaurants = [r for restaurants in yelp_restaurants.values() for r in restaurants if r]

    # [Action Node] yelp.com:F1:A1 - Business keyword search
    has_late_night_or_live_music = has_any_ci(answer, ['late night', 'live music'])
    evaluator.add_custom_node(
        result=bool(has_late_night_or_live_music and len(all_yelp_restaurants) > 0),
        id="yelp_action_keyword_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for Late Night Dining or Live Music venues",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-criteria filter panel
    has_yelp_filters = has_any_ci(answer, ['4 star', 'midnight', '$$'])
    evaluator.add_custom_node(
        result=bool(has_yelp_filters),
        id="yelp_action_filter_panel",
        desc="[Action Node] yelp.com:F1:A2 - Open and apply rating, hours, and price filters",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A9 - Click business card
    has_operating_hours = any(r.operating_hours for r in all_yelp_restaurants)
    evaluator.add_custom_node(
        result=bool(has_operating_hours),
        id="yelp_action_business_details",
        desc="[Action Node] yelp.com:F1:A9 - Click business cards to view operating hours",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Business basic info awareness
    has_rating_4_plus = any(rating_meets_threshold(r.rating, 4.0) for r in all_yelp_restaurants if r.rating)
    has_dollar_signs = any(is_dollar_sign_level(r.price_level, ['$$', '$$$']) for r in all_yelp_restaurants if r.price_level)
    evaluator.add_custom_node(
        result=bool(has_rating_4_plus and has_dollar_signs),
        id="yelp_perception_basic_info",
        desc="[Perception Node] yelp.com:F1:P1 - Identify ratings and price levels on business cards",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P7 - Operating hours awareness
    has_midnight_hours = any(is_open_past_midnight(r.operating_hours) for r in all_yelp_restaurants if r.operating_hours)
    evaluator.add_custom_node(
        result=bool(has_midnight_hours),
        id="yelp_perception_hours",
        desc="[Perception Node] yelp.com:F2:P7 - Verify restaurants open past midnight",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P8 - Amenity/service awareness
    evaluator.add_custom_node(
        result=bool(has_late_night_or_live_music),
        id="yelp_perception_amenities",
        desc="[Perception Node] yelp.com:F2:P8 - Identify Late Night Dining or Live Music tags",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P14 - Price level awareness
    evaluator.add_custom_node(
        result=bool(has_dollar_signs),
        id="yelp_perception_price_level",
        desc="[Perception Node] yelp.com:F2:P14 - Recognize $$-$$$ price level indicators",
        parent=yelp_node,
        critical=False
    )

    # 3.6 Output completeness checks
    output_node = evaluator.add_parallel(
        id="output_section",
        desc="Complete itinerary table output verification",
        parent=root,
        critical=False
    )

    has_3_cities = len(events) >= 3
    evaluator.add_custom_node(
        result=bool(has_3_cities),
        id="output_three_cities",
        desc="Output includes 3 different cities",
        parent=output_node,
        critical=False
    )

    has_6_airbnb = len(all_airbnb_listings) >= 6
    evaluator.add_custom_node(
        result=bool(has_6_airbnb),
        id="output_six_airbnb",
        desc="Output includes 2 Airbnb listings per city (6 total)",
        parent=output_node,
        critical=False
    )

    has_6_yelp = len(all_yelp_restaurants) >= 6
    evaluator.add_custom_node(
        result=bool(has_6_yelp),
        id="output_six_yelp",
        desc="Output includes 2 Yelp restaurants per city (6 total)",
        parent=output_node,
        critical=False
    )

    has_table_format = has_any_ci(answer, ['table', '|', 'city', 'date', 'venue'])
    evaluator.add_custom_node(
        result=bool(has_table_format),
        id="output_table_format",
        desc="Output is formatted as a comprehensive itinerary table",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
