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
TASK_ID = "task-a8b347"
TASK_DESCRIPTION = "I'm planning a 'record store road trip' on the US West Coast in Spring 2026, following the route of Los Angeles → San Francisco → Portland to dig for vinyl records. I mainly collect jazz and rock records and prefer not to drive more than 400 miles per day.\n\nFirst, use Yelp to find 3-5 independent record stores in each of Los Angeles, San Francisco, and Portland, specializing in vinyl records with a rating of 4.5 or higher. For each store, note its name, address, rating, and operating hours. Additionally, review the comments to identify any mentioned stylistic characteristics or specialties.\n\nNext, use Google Maps to plan the driving routes from Los Angeles to San Francisco and from San Francisco to Portland, ensuring each leg is within 400 miles. Within each city, mark the locations of the identified record stores and optimize the visiting order, taking into account operating hours and distances between stores.\n\nAfterward, search on Airbnb for accommodation. Find one listing in San Francisco and one in Portland. Each should have a rating of 4.2 or higher, be located near a concentration of record stores, be available for booking on the travel dates, and cost no more than $150 per night.\n\nFinally, search Discogs using the names of these record stores to identify which ones have a seller page where online inventory can be previewed.\n\n**Output:**\n\n*   For each record store: name, address, Yelp rating, operating hours, specialty categories mentioned in Yelp reviews, and a link to its Yelp business page.\n*   For the Los Angeles → San Francisco and San Francisco → Portland routes: driving distance and estimated time, and a Google Maps route link.\n*   For multi-store visits within each city: a description of the optimized visiting order and a saved Google Maps route.\n*   For each Airbnb listing in San Francisco and Portland: name, price, address, rating, and Airbnb listing link.\n*   A list of record stores with a Discogs Seller page, along with their Discogs store links."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RecordStore(BaseModel):
    """Single record store information"""
    name: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    operating_hours: Optional[str] = None
    specialty: Optional[str] = None
    yelp_link: Optional[str] = None


class RecordStoresInfo(BaseModel):
    """All record stores found across three cities"""
    los_angeles_stores: List[RecordStore] = Field(default_factory=list)
    san_francisco_stores: List[RecordStore] = Field(default_factory=list)
    portland_stores: List[RecordStore] = Field(default_factory=list)


class DrivingRoute(BaseModel):
    """Driving route between cities"""
    from_city: Optional[str] = None
    to_city: Optional[str] = None
    distance_text: Optional[str] = None
    time_text: Optional[str] = None
    route_link: Optional[str] = None


class CityRouteInfo(BaseModel):
    """Within-city multi-store route information"""
    city: Optional[str] = None
    visiting_order_description: Optional[str] = None
    route_link: Optional[str] = None


class AirbnbListing(BaseModel):
    """Airbnb accommodation listing"""
    city: Optional[str] = None
    name: Optional[str] = None
    price: Optional[str] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    listing_link: Optional[str] = None


class DiscogsStore(BaseModel):
    """Record store with Discogs seller page"""
    store_name: Optional[str] = None
    discogs_link: Optional[str] = None


class FullTripInfo(BaseModel):
    """Complete trip information extraction"""
    la_to_sf_route: Optional[DrivingRoute] = None
    sf_to_portland_route: Optional[DrivingRoute] = None
    la_city_route: Optional[CityRouteInfo] = None
    sf_city_route: Optional[CityRouteInfo] = None
    portland_city_route: Optional[CityRouteInfo] = None
    sf_airbnb: Optional[AirbnbListing] = None
    portland_airbnb: Optional[AirbnbListing] = None
    discogs_stores: List[DiscogsStore] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_record_stores() -> str:
    return """
Extract all record stores mentioned in the answer from Los Angeles, San Francisco, and Portland.

For each store, extract:
- name: the store name
- address: full address
- rating: numerical rating (e.g., 4.5, 4.7)
- operating_hours: operating hours text
- specialty: any specialty categories or characteristics mentioned from reviews (e.g., 'jazz vinyl specialist', 'rock collection')
- yelp_link: Yelp business page URL if present

Group them by city into los_angeles_stores, san_francisco_stores, and portland_stores arrays.
If any field is missing for a store, set it to null.
"""


def prompt_extract_trip_info() -> str:
    return """
Extract the following trip planning information from the answer:

1. la_to_sf_route: Los Angeles to San Francisco driving route
   - from_city, to_city, distance_text, time_text, route_link

2. sf_to_portland_route: San Francisco to Portland driving route
   - from_city, to_city, distance_text, time_text, route_link

3. la_city_route, sf_city_route, portland_city_route: Within-city multi-store routes
   - city, visiting_order_description, route_link

4. sf_airbnb: San Francisco accommodation
   - city, name, price, address, rating, listing_link

5. portland_airbnb: Portland accommodation
   - city, name, price, address, rating, listing_link

6. discogs_stores: List of stores with Discogs seller pages
   - store_name, discogs_link

If any field is missing, set it to null or empty list.
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


def extract_distance_miles(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Extract number that might be miles
    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def count_stores_in_city(stores: List[RecordStore]) -> int:
    """Count valid stores (those with at least a name)"""
    return sum(1 for s in stores if s and s.name)


def all_ratings_above_threshold(stores: List[RecordStore], threshold: float) -> bool:
    """Check if all stores with ratings meet the threshold"""
    if not stores:
        return False
    rated_stores = [s for s in stores if s and s.rating is not None]
    if not rated_stores:
        return False
    return all(s.rating >= threshold for s in rated_stores)


def has_valid_link(link: Optional[str], domain: str) -> bool:
    """Check if link is valid and contains domain"""
    if not link:
        return False
    return domain.lower() in link.lower() and ('http://' in link.lower() or 'https://' in link.lower())


def extract_price_value(price_text: Optional[str]) -> Optional[float]:
    """Extract numeric price value"""
    if not price_text:
        return None
    # Remove common currency symbols and extract number
    cleaned = re.sub(r'[$€£¥,]', '', price_text)
    nums = re.findall(r'(\d+(?:\.\d+)?)', cleaned)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


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
    stores_info = await evaluator.extract(
        prompt=prompt_extract_record_stores(),
        template_class=RecordStoresInfo,
        extraction_name="record_stores_info"
    )

    trip_info = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=FullTripInfo,
        extraction_name="full_trip_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Yelp record stores section
    yelp_section = evaluator.add_sequential(
        id="yelp_record_stores",
        desc="Yelp record store search and information extraction across three cities",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Keyword search for vinyl record stores
    yelp_search_ok = (
        has_any_ci(answer, ['yelp']) and
        has_any_ci(answer, ['vinyl', 'record store', 'record shop'])
    )
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_keyword_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for vinyl record stores using keywords on Yelp",
        parent=yelp_section,
        critical=False
    )

    # Check store counts per city (o1 verification)
    la_count = count_stores_in_city(stores_info.los_angeles_stores if stores_info else [])
    sf_count = count_stores_in_city(stores_info.san_francisco_stores if stores_info else [])
    portland_count = count_stores_in_city(stores_info.portland_stores if stores_info else [])

    store_count_ok = (3 <= la_count <= 5) and (3 <= sf_count <= 5) and (3 <= portland_count <= 5)
    evaluator.add_custom_node(
        result=bool(store_count_ok),
        id="yelp_store_count",
        desc="Found 3-5 record stores in each of the three cities (Los Angeles, San Francisco, Portland)",
        parent=yelp_section,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-condition filtering (rating 4.5+)
    all_stores = []
    if stores_info:
        all_stores.extend(stores_info.los_angeles_stores or [])
        all_stores.extend(stores_info.san_francisco_stores or [])
        all_stores.extend(stores_info.portland_stores or [])

    rating_filter_ok = all_ratings_above_threshold(all_stores, 4.5)
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),
        id="yelp_rating_filter",
        desc="[Action Node] yelp.com:F1:A2 - Apply rating filter to show only stores with 4.5+ ratings",
        parent=yelp_section,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Basic business info perception (o1, o2, o3)
    has_names = all(s.name for s in all_stores if s)
    has_addresses = sum(1 for s in all_stores if s and s.address) >= len(all_stores) * 0.8
    has_ratings = sum(1 for s in all_stores if s and s.rating is not None) >= len(all_stores) * 0.8

    basic_info_ok = has_names and has_addresses and has_ratings and len(all_stores) >= 9
    evaluator.add_custom_node(
        result=bool(basic_info_ok),
        id="yelp_basic_info_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Extract store names, addresses, and ratings from business cards",
        parent=yelp_section,
        critical=False
    )

    # [Action Node] yelp.com:F1:A9 - Click into business details
    has_hours = sum(1 for s in all_stores if s and s.operating_hours) >= len(all_stores) * 0.7
    evaluator.add_custom_node(
        result=bool(has_hours),
        id="yelp_click_details",
        desc="[Action Node] yelp.com:F1:A9 - Click into business detail pages to get complete information",
        parent=yelp_section,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P7 - Operating hours perception (o4)
    evaluator.add_custom_node(
        result=bool(has_hours),
        id="yelp_hours_perception",
        desc="[Perception Node] yelp.com:F2:P7 - Extract operating hours from business detail pages",
        parent=yelp_section,
        critical=False
    )

    # [Action Node] yelp.com:F3:A10 - Switch to Reviews tab
    reviews_mentioned = has_any_ci(answer, ['review', 'comment', 'specialty', 'specialties'])
    evaluator.add_custom_node(
        result=bool(reviews_mentioned),
        id="yelp_reviews_tab",
        desc="[Action Node] yelp.com:F3:A10 - Navigate to Reviews tab to read customer comments",
        parent=yelp_section,
        critical=False
    )

    # [Perception Node] yelp.com:F3:P3 - Review content understanding (o5)
    has_specialties = sum(1 for s in all_stores if s and s.specialty) >= len(all_stores) * 0.5
    evaluator.add_custom_node(
        result=bool(has_specialties),
        id="yelp_specialty_perception",
        desc="[Perception Node] yelp.com:F3:P3 - Extract specialty categories and characteristics from review content",
        parent=yelp_section,
        critical=False
    )

    # Yelp links present
    has_yelp_links = sum(1 for s in all_stores if s and has_valid_link(s.yelp_link, 'yelp')) >= len(all_stores) * 0.7
    evaluator.add_custom_node(
        result=bool(has_yelp_links),
        id="yelp_links_present",
        desc="Provide Yelp business page links for the stores",
        parent=yelp_section,
        critical=False
    )

    # 3.2 Google Maps routing section
    maps_section = evaluator.add_sequential(
        id="google_maps_routing",
        desc="Google Maps route planning for inter-city travel and within-city store visits",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input start and end points for routes (o7, o8)
    has_la_sf = trip_info and trip_info.la_to_sf_route and trip_info.la_to_sf_route.distance_text
    has_sf_portland = trip_info and trip_info.sf_to_portland_route and trip_info.sf_to_portland_route.distance_text

    evaluator.add_custom_node(
        result=bool(has_la_sf and has_sf_portland),
        id="maps_input_endpoints",
        desc="[Action Node] maps.google.com:F2:A5 - Input start and destination points for LA→SF and SF→Portland routes",
        parent=maps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode
    driving_mode_ok = has_any_ci(answer, ['driv', 'car', 'route'])
    evaluator.add_custom_node(
        result=bool(driving_mode_ok),
        id="maps_driving_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving as the transportation mode",
        parent=maps_section,
        critical=False
    )

    # Verify distance constraint (400 miles per day)
    la_sf_dist = extract_distance_miles(trip_info.la_to_sf_route.distance_text if trip_info and trip_info.la_to_sf_route else None)
    sf_portland_dist = extract_distance_miles(trip_info.sf_to_portland_route.distance_text if trip_info and trip_info.sf_to_portland_route else None)

    distance_ok = (la_sf_dist and la_sf_dist <= 400) and (sf_portland_dist and sf_portland_dist <= 400)
    evaluator.add_custom_node(
        result=bool(distance_ok),
        id="maps_distance_constraint",
        desc="Verify both route legs are within 400 miles daily driving limit",
        parent=maps_section,
        critical=False
    )

    # Route timing information present
    has_timing = (
        trip_info and
        trip_info.la_to_sf_route and trip_info.la_to_sf_route.time_text and
        trip_info.sf_to_portland_route and trip_info.sf_to_portland_route.time_text
    )
    evaluator.add_custom_node(
        result=bool(has_timing),
        id="maps_route_timing",
        desc="Provide estimated driving time for both inter-city routes",
        parent=maps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Search and mark store locations
    city_routes_exist = (
        trip_info and
        (trip_info.la_city_route or trip_info.sf_city_route or trip_info.portland_city_route)
    )
    evaluator.add_custom_node(
        result=bool(city_routes_exist),
        id="maps_mark_stores",
        desc="[Action Node] maps.google.com:F1:A2 - Search and mark record store locations within each city",
        parent=maps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A18 - Add intermediate waypoints (o10)
    has_visiting_order = (
        (trip_info and trip_info.la_city_route and trip_info.la_city_route.visiting_order_description) or
        (trip_info and trip_info.sf_city_route and trip_info.sf_city_route.visiting_order_description) or
        (trip_info and trip_info.portland_city_route and trip_info.portland_city_route.visiting_order_description)
    )
    evaluator.add_custom_node(
        result=bool(has_visiting_order),
        id="maps_add_waypoints",
        desc="[Action Node] maps.google.com:F2:A18 - Add multiple record stores as waypoints to optimize within-city routes",
        parent=maps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F9:A15 - Save routes (o9, o11)
    has_intercity_links = (
        trip_info and
        has_valid_link(trip_info.la_to_sf_route.route_link if trip_info.la_to_sf_route else None, 'google') and
        has_valid_link(trip_info.sf_to_portland_route.route_link if trip_info.sf_to_portland_route else None, 'google')
    )
    has_city_links = (
        trip_info and (
            has_valid_link(trip_info.la_city_route.route_link if trip_info.la_city_route else None, 'google') or
            has_valid_link(trip_info.sf_city_route.route_link if trip_info.sf_city_route else None, 'google') or
            has_valid_link(trip_info.portland_city_route.route_link if trip_info.portland_city_route else None, 'google')
        )
    )
    evaluator.add_custom_node(
        result=bool(has_intercity_links and has_city_links),
        id="maps_save_routes",
        desc="[Action Node] maps.google.com:F9:A15 - Save and generate shareable links for all planned routes",
        parent=maps_section,
        critical=False
    )

    # 3.3 Airbnb accommodation section
    airbnb_section = evaluator.add_sequential(
        id="airbnb_accommodation",
        desc="Airbnb accommodation search in San Francisco and Portland",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date selection
    has_sf_airbnb = trip_info and trip_info.sf_airbnb and trip_info.sf_airbnb.name
    has_portland_airbnb = trip_info and trip_info.portland_airbnb and trip_info.portland_airbnb.name

    evaluator.add_custom_node(
        result=bool(has_sf_airbnb and has_portland_airbnb),
        id="airbnb_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Select travel dates to check availability (o12: one listing in each city)",
        parent=airbnb_section,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Location search near record stores (o14)
    near_stores = has_any_ci(answer, ['near', 'close', 'concentration', 'cluster', 'record store'])
    evaluator.add_custom_node(
        result=bool(near_stores),
        id="airbnb_location_search",
        desc="[Action Node] airbnb.com:F1:A10 - Search for accommodations near record store concentration areas",
        parent=airbnb_section,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A13 - Price range filter (o13)
    sf_price = extract_price_value(trip_info.sf_airbnb.price if trip_info and trip_info.sf_airbnb else None)
    portland_price = extract_price_value(trip_info.portland_airbnb.price if trip_info and trip_info.portland_airbnb else None)

    price_ok = (sf_price and sf_price <= 150) and (portland_price and portland_price <= 150)
    evaluator.add_custom_node(
        result=bool(price_ok),
        id="airbnb_price_filter",
        desc="[Action Node] airbnb.com:F2:A13 - Set price range filter to max $150 per night",
        parent=airbnb_section,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A34 - Open filter drawer for rating (o15)
    sf_rating = trip_info.sf_airbnb.rating if trip_info and trip_info.sf_airbnb else None
    portland_rating = trip_info.portland_airbnb.rating if trip_info and trip_info.portland_airbnb else None

    rating_ok = (sf_rating and sf_rating >= 4.2) and (portland_rating and portland_rating >= 4.2)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="airbnb_rating_filter",
        desc="[Action Node] airbnb.com:F2:A34 - Open filters and set minimum rating to 4.2",
        parent=airbnb_section,
        critical=False
    )

    # [Action Node] airbnb.com:F3:A19 - Click listing for details (o12, o13, o14, o15)
    has_full_info = (
        trip_info and
        trip_info.sf_airbnb and trip_info.sf_airbnb.name and trip_info.sf_airbnb.price and
        trip_info.sf_airbnb.address and trip_info.sf_airbnb.rating is not None and
        trip_info.portland_airbnb and trip_info.portland_airbnb.name and trip_info.portland_airbnb.price and
        trip_info.portland_airbnb.address and trip_info.portland_airbnb.rating is not None
    )
    evaluator.add_custom_node(
        result=bool(has_full_info),
        id="airbnb_click_details",
        desc="[Action Node] airbnb.com:F3:A19 - Click listings to view complete details (name, price, address, rating)",
        parent=airbnb_section,
        critical=False
    )

    # [Perception Node] airbnb.com:F3:P10 - List content understanding (o14)
    evaluator.add_custom_node(
        result=bool(near_stores and has_full_info),
        id="airbnb_list_perception",
        desc="[Perception Node] airbnb.com:F3:P10 - Understand listing locations relative to record store areas",
        parent=airbnb_section,
        critical=False
    )

    # Airbnb links present
    has_airbnb_links = (
        trip_info and
        has_valid_link(trip_info.sf_airbnb.listing_link if trip_info.sf_airbnb else None, 'airbnb') and
        has_valid_link(trip_info.portland_airbnb.listing_link if trip_info.portland_airbnb else None, 'airbnb')
    )
    evaluator.add_custom_node(
        result=bool(has_airbnb_links),
        id="airbnb_links_present",
        desc="Provide Airbnb listing links for both accommodations",
        parent=airbnb_section,
        critical=False
    )

    # 3.4 Discogs seller page section
    discogs_section = evaluator.add_sequential(
        id="discogs_seller_pages",
        desc="Discogs seller page search for record stores",
        parent=root,
        critical=False
    )

    # [Action Node] discogs.com:F1:A1 - Search for store names (o17)
    has_discogs_stores = trip_info and trip_info.discogs_stores and len(trip_info.discogs_stores) > 0
    evaluator.add_custom_node(
        result=bool(has_discogs_stores),
        id="discogs_search",
        desc="[Action Node] discogs.com:F1:A1 - Search for record store names on Discogs",
        parent=discogs_section,
        critical=False
    )

    # [Action Node] discogs.com:F1:A7 - Click results to check for seller pages (o18)
    has_discogs_links = (
        trip_info and trip_info.discogs_stores and
        sum(1 for ds in trip_info.discogs_stores if ds and has_valid_link(ds.discogs_link, 'discogs')) > 0
    )
    evaluator.add_custom_node(
        result=bool(has_discogs_links),
        id="discogs_check_seller",
        desc="[Action Node] discogs.com:F1:A7 - Click search results to verify which stores have Seller pages",
        parent=discogs_section,
        critical=False
    )

    # List of stores with Discogs links provided
    evaluator.add_custom_node(
        result=bool(has_discogs_stores),
        id="discogs_list_output",
        desc="Provide list of record stores with Discogs Seller pages and their links",
        parent=discogs_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
