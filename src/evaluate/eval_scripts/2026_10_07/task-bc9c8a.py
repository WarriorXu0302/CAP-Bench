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
TASK_ID = "task-bc9c8a"
TASK_DESCRIPTION = 'I am planning a cross-country road trip from San Francisco to New York, and my vehicle has a full tank range of approximately 450 miles. Please help me plan a route on Google Maps and identify a series of suitable gas stations along the way.\n\nRequirements:\n*   The driving distance between any two consecutive gas stations should be between 300 and 450 miles.\n*   The first gas station should be 300-450 miles from San Francisco.\n*   All gas stations should be located close to major highways (e.g., I-80).\n\nAfter the route is planned, for each gas station, find restaurants within a 5-mile radius with a rating of 4 stars or higher (using Yelp or TripAdvisor).\n\nOutput:\n*   For each gas station: name, address, Google Maps link, and driving distance from the previous stop.\n*   For each recommended restaurant near a gas station: name, rating, distance from the gas station, and restaurant link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class GasStation(BaseModel):
    """Single gas station information"""
    name: Optional[str] = None
    address: Optional[str] = None
    maps_link: Optional[str] = None
    distance_from_previous: Optional[str] = None


class Restaurant(BaseModel):
    """Single restaurant information"""
    name: Optional[str] = None
    rating: Optional[str] = None
    distance_from_station: Optional[str] = None
    link: Optional[str] = None


class RouteInfo(BaseModel):
    """Route and gas stations extracted from answer"""
    gas_stations: Optional[List[GasStation]] = Field(default_factory=list)
    restaurants: Optional[List[Restaurant]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_route_info() -> str:
    return """
Extract all gas stations and restaurants mentioned in the answer for the San Francisco to New York road trip.

For each gas station, extract:
- name: the gas station name
- address: the gas station address
- maps_link: Google Maps link if provided
- distance_from_previous: driving distance from the previous stop (in miles)

For each restaurant, extract:
- name: the restaurant name
- rating: the rating (stars)
- distance_from_station: distance from the gas station
- link: restaurant link (Yelp or TripAdvisor)

If any field is missing, set it to null. Return empty lists if no gas stations or restaurants are found.
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


def is_distance_in_range(distance_text: Optional[str], min_miles: float, max_miles: float) -> bool:
    """Check if distance is within the specified range"""
    if not distance_text:
        return False
    num = extract_float(distance_text)
    if num is None:
        return False
    return min_miles <= num <= max_miles


def is_rating_at_least_4(rating_text: Optional[str]) -> bool:
    """Check if rating is at least 4 stars"""
    if not rating_text:
        return False
    num = extract_float(rating_text)
    if num is None:
        return False
    return num >= 4.0


def is_distance_within_5_miles(distance_text: Optional[str]) -> bool:
    """Check if distance is within 5 miles"""
    if not distance_text:
        return False
    num = extract_float(distance_text)
    if num is None:
        return False
    return num <= 5.0


def has_valid_link(link: Optional[str]) -> bool:
    """Check if link looks valid"""
    if not link:
        return False
    return link.startswith('http://') or link.startswith('https://')


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
    route_info = await evaluator.extract(
        prompt=prompt_extract_route_info(),
        template_class=RouteInfo,
        extraction_name="route_and_stations"
    )

    gas_stations = route_info.gas_stations or []
    restaurants = route_info.restaurants or []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Google Maps route planning section
    maps_route_node = evaluator.add_sequential(
        id="maps_route_planning",
        desc="Google Maps route planning from San Francisco to New York",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route input with start and end points
    route_input_ok = (has_any_ci(answer, ['san francisco']) and
                      has_any_ci(answer, ['new york']) and
                      (has_any_ci(answer, ['route', 'directions', 'google maps'])))
    evaluator.add_custom_node(
        result=bool(route_input_ok),
        id="maps_route_input",
        desc="[Action Node] maps.google.com:F2:A5 - Input start point (San Francisco) and end point (New York) for route planning",
        parent=maps_route_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode
    driving_mode_ok = has_any_ci(answer, ['driv', 'car', 'vehicle'])
    evaluator.add_custom_node(
        result=bool(driving_mode_ok),
        id="maps_driving_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode for route calculation",
        parent=maps_route_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A20 - Search for gas stations along route
    gas_search_ok = (len(gas_stations) > 0 and
                     has_any_ci(answer, ['gas station', 'fuel', 'refuel']))
    evaluator.add_custom_node(
        result=bool(gas_search_ok),
        id="maps_gas_station_search",
        desc="[Action Node] maps.google.com:F2:A20 - Search for gas stations along the route",
        parent=maps_route_node,
        critical=False
    )

    # 3.2 Google Maps map interaction section
    maps_interaction_node = evaluator.add_parallel(
        id="maps_map_interaction",
        desc="Google Maps zoom, pan, and marker identification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F3:A9 - Map zoom and pan to mark stations
    zoom_pan_ok = (len(gas_stations) >= 2)
    evaluator.add_custom_node(
        result=bool(zoom_pan_ok),
        id="maps_zoom_pan",
        desc="[Action Node] maps.google.com:F3:A9 - Zoom and pan the map to identify and mark gas stations at 300-450 mile intervals",
        parent=maps_interaction_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P12 - Identify gas station markers
    marker_identification_ok = (len(gas_stations) > 0 and
                                all(s.name for s in gas_stations if s))
    evaluator.add_custom_node(
        result=bool(marker_identification_ok),
        id="maps_marker_identification",
        desc="[Perception Node] maps.google.com:F3:P12 - Identify gas station markers on the map",
        parent=maps_interaction_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P13 - Perceive map scale and distance
    scale_perception_ok = (len(gas_stations) > 0 and
                          any(s.distance_from_previous for s in gas_stations if s))
    evaluator.add_custom_node(
        result=bool(scale_perception_ok),
        id="maps_scale_perception",
        desc="[Perception Node] maps.google.com:F3:P13 - Perceive map scale to judge distances for 300-450 mile intervals",
        parent=maps_interaction_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P20 - Perceive zoom level for accurate distance
    zoom_level_perception_ok = (len(gas_stations) > 0 and
                                any(s.distance_from_previous for s in gas_stations if s))
    evaluator.add_custom_node(
        result=bool(zoom_level_perception_ok),
        id="maps_zoom_level_perception",
        desc="[Perception Node] maps.google.com:F3:P20 - Perceive map zoom level to ensure accurate distance measurement",
        parent=maps_interaction_node,
        critical=False
    )

    # 3.3 Gas station details section
    gas_station_details_node = evaluator.add_parallel(
        id="gas_station_details",
        desc="Gas station information completeness and compliance",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Search for gas stations at target points
    point_search_ok = len(gas_stations) > 0
    evaluator.add_custom_node(
        result=bool(point_search_ok),
        id="maps_point_gas_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for gas stations near target points along the route",
        parent=gas_station_details_node,
        critical=False
    )

    # [Action Node] maps.google.com:F11:A20 - Along-route search for gas stations
    along_route_search_ok = (len(gas_stations) > 0 and
                             has_any_ci(answer, ['i-80', 'highway', 'interstate']))
    evaluator.add_custom_node(
        result=bool(along_route_search_ok),
        id="maps_along_route_search",
        desc="[Action Node] maps.google.com:F11:A20 - Use along-route search to locate gas stations on major highways",
        parent=gas_station_details_node,
        critical=False
    )

    # Check gas station output completeness
    has_names = len(gas_stations) > 0 and all(s.name for s in gas_stations if s)
    has_addresses = len(gas_stations) > 0 and all(s.address for s in gas_stations if s)
    has_links = len(gas_stations) > 0 and any(s.maps_link for s in gas_stations if s)

    evaluator.add_custom_node(
        result=bool(has_names and has_addresses),
        id="gas_station_basic_info",
        desc="Gas stations include name and address (o1, o2)",
        parent=gas_station_details_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_links),
        id="gas_station_maps_links",
        desc="Gas stations include Google Maps links (o3)",
        parent=gas_station_details_node,
        critical=False
    )

    # Check distance constraints (300-450 miles)
    distances_valid = []
    for station in gas_stations:
        if station and station.distance_from_previous:
            distances_valid.append(is_distance_in_range(station.distance_from_previous, 300, 450))

    distance_constraint_ok = len(distances_valid) > 0 and all(distances_valid)
    evaluator.add_custom_node(
        result=bool(distance_constraint_ok),
        id="distance_constraint_check",
        desc="All gas stations meet the 300-450 mile distance constraint (o4)",
        parent=gas_station_details_node,
        critical=False
    )

    # 3.4 Restaurant search section (Yelp)
    yelp_node = evaluator.add_parallel(
        id="yelp_restaurant_search",
        desc="Yelp restaurant search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Search for restaurants
    yelp_search_ok = (len(restaurants) > 0 and
                      has_any_ci(answer, ['yelp', 'restaurant']))
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_restaurant_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for restaurants near gas stations on Yelp",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-criteria filtering (distance and rating)
    yelp_filter_ok = (len(restaurants) > 0 and
                      has_any_ci(answer, ['5 mile', 'rating', '4 star']))
    evaluator.add_custom_node(
        result=bool(yelp_filter_ok),
        id="yelp_filter_criteria",
        desc="[Action Node] yelp.com:F1:A2 - Apply filters for 5-mile radius and 4+ star rating",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Extract restaurant basic info
    has_restaurant_names = len(restaurants) > 0 and all(r.name for r in restaurants if r)
    has_ratings = len(restaurants) > 0 and all(r.rating for r in restaurants if r)
    has_distances = len(restaurants) > 0 and all(r.distance_from_station for r in restaurants if r)

    yelp_perception_ok = has_restaurant_names and has_ratings and has_distances
    evaluator.add_custom_node(
        result=bool(yelp_perception_ok),
        id="yelp_info_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Extract restaurant name, rating, and distance from cards (o5, o6, o7)",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F9:A19 - Map drag and zoom
    yelp_map_interaction_ok = len(restaurants) > 0
    evaluator.add_custom_node(
        result=bool(yelp_map_interaction_ok),
        id="yelp_map_interaction",
        desc="[Action Node] yelp.com:F9:A19 - Drag and zoom Yelp map to locate area around gas stations",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F9:P10 - Perceive restaurant location relative to gas station
    yelp_location_perception_ok = (len(restaurants) > 0 and
                                   all(r.distance_from_station for r in restaurants if r))
    evaluator.add_custom_node(
        result=bool(yelp_location_perception_ok),
        id="yelp_location_perception",
        desc="[Perception Node] yelp.com:F9:P10 - Perceive restaurant locations relative to gas stations on the map",
        parent=yelp_node,
        critical=False
    )

    # 3.5 Restaurant search section (TripAdvisor)
    tripadvisor_node = evaluator.add_parallel(
        id="tripadvisor_restaurant_search",
        desc="TripAdvisor restaurant search and filtering",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F6:A1 - Search for restaurants
    tripadvisor_search_ok = has_any_ci(answer, ['tripadvisor'])
    evaluator.add_custom_node(
        result=bool(tripadvisor_search_ok),
        id="tripadvisor_restaurant_search",
        desc="[Action Node] tripadvisor.com:F6:A1 - Search for restaurants near gas stations on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F6:A6 - Multi-criteria filtering for rating
    tripadvisor_filter_ok = (tripadvisor_search_ok and
                             has_any_ci(answer, ['rating', '4 star']))
    evaluator.add_custom_node(
        result=bool(tripadvisor_filter_ok),
        id="tripadvisor_filter_rating",
        desc="[Action Node] tripadvisor.com:F6:A6 - Apply rating filter for 4+ stars on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.6 Restaurant output validation
    restaurant_output_node = evaluator.add_parallel(
        id="restaurant_output_validation",
        desc="Restaurant information validation",
        parent=root,
        critical=False
    )

    # Check rating constraint (4+ stars)
    ratings_valid = []
    for restaurant in restaurants:
        if restaurant and restaurant.rating:
            ratings_valid.append(is_rating_at_least_4(restaurant.rating))

    rating_constraint_ok = len(ratings_valid) > 0 and all(ratings_valid)
    evaluator.add_custom_node(
        result=bool(rating_constraint_ok),
        id="restaurant_rating_constraint",
        desc="All restaurants have ratings of 4 stars or higher (o6)",
        parent=restaurant_output_node,
        critical=False
    )

    # Check distance constraint (within 5 miles)
    restaurant_distances_valid = []
    for restaurant in restaurants:
        if restaurant and restaurant.distance_from_station:
            restaurant_distances_valid.append(is_distance_within_5_miles(restaurant.distance_from_station))

    restaurant_distance_ok = len(restaurant_distances_valid) > 0 and all(restaurant_distances_valid)
    evaluator.add_custom_node(
        result=bool(restaurant_distance_ok),
        id="restaurant_distance_constraint",
        desc="All restaurants are within 5 miles of their respective gas stations (o7)",
        parent=restaurant_output_node,
        critical=False
    )

    # Check restaurant links
    has_restaurant_links = len(restaurants) > 0 and any(r.link for r in restaurants if r)
    evaluator.add_custom_node(
        result=bool(has_restaurant_links),
        id="restaurant_links",
        desc="Restaurants include links to Yelp or TripAdvisor (o8)",
        parent=restaurant_output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
