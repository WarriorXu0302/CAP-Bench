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
TASK_ID = "task-2f4ebf"
TASK_DESCRIPTION = 'I’m planning to drive my 2024 Tesla Model Y Long Range on a weekend next month from Griffith Observatory in Los Angeles to the Bellagio Hotel in Las Vegas. I don’t trust the official EPA range, so first go to Car and Driver and find the real-world “75-mph highway range” test result for this model. If the exact model’s 75-mph test data is unavailable, then find the tested 75-mph range for a same-year or nearby-year Model Y Long Range on Car and Driver, and clearly note any model/year differences. Use 80% of that tested range as my safe maximum distance for the first driving leg.\n\nNext, plan the route in Google Maps and use the “Search along route” feature to identify 3 Supercharger stations located between 150 miles from the origin and that safe maximum distance (if fewer than 3 are available, record all that can be found and note the count).\n\nFinally, for each of those 3 charging stations, go to Yelp and find the highest-rated restaurant within 0.2 miles of the station’s address (rating must be 4.0+ and price level must be $$ or lower). If no qualifying option exists within 0.2 miles, expand the radius to 0.5 miles and explicitly note that this relaxation was applied.\n\nOutput: the Car and Driver tested 75-mph range, the calculated safe maximum distance, and the C&D article link; the names of 3 candidate charging stations and their distances (in miles) from the origin; and for each station, the recommended restaurant name, Yelp rating, price tier, and Yelp detail page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CarAndDriverData(BaseModel):
    """Car and Driver 75-mph range test data extracted from the answer"""
    tested_75mph_range_text: Optional[str] = None
    safe_maximum_distance_text: Optional[str] = None
    article_link: Optional[str] = None
    model_year_note: Optional[str] = None


class SuperchargerStation(BaseModel):
    """Individual Supercharger station information"""
    station_name: Optional[str] = None
    distance_from_origin_text: Optional[str] = None
    restaurant_name: Optional[str] = None
    restaurant_rating_text: Optional[str] = None
    restaurant_price_tier: Optional[str] = None
    restaurant_yelp_link: Optional[str] = None
    radius_relaxation_note: Optional[str] = None


class SuperchargerStations(BaseModel):
    """All Supercharger stations extracted from the answer"""
    stations: List[SuperchargerStation] = Field(default_factory=list)
    station_count_note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_caranddriver_data() -> str:
    return """
Extract the Car and Driver 75-mph highway range test data from the answer for the 2024 Tesla Model Y Long Range (or similar model/year if noted):

Return:
- tested_75mph_range_text: the 75-mph highway range exactly as stated (include units if present)
- safe_maximum_distance_text: the calculated safe maximum distance (80% of tested range) exactly as stated
- article_link: the Car and Driver article URL or link
- model_year_note: any note about model or year differences mentioned

If any field is missing, set it to null.
"""


def prompt_extract_supercharger_stations() -> str:
    return """
From the answer, extract information about the 3 Supercharger stations (or fewer if noted) identified along the route from Griffith Observatory to Bellagio Hotel:

For each station, extract:
- station_name: the name of the Supercharger station
- distance_from_origin_text: the distance from Griffith Observatory in miles
- restaurant_name: the recommended restaurant name
- restaurant_rating_text: the Yelp rating of the restaurant
- restaurant_price_tier: the price tier ($ or $$)
- restaurant_yelp_link: the Yelp detail page URL
- radius_relaxation_note: any note about expanding search radius from 0.2 to 0.5 miles

Also extract:
- station_count_note: any note about having fewer than 3 stations available

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


def looks_like_miles(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['mile', 'mi']) or re.search(r'\d+\s*$', text.strip()))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['http://', 'https://']) or has_any_ci(text, ['.com', '.org', '.net'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_price_tier(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    return t in ['$', '$$', '$$$', '$$$$'] or re.search(r'\$+', t)


def is_valid_price_tier(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip()
    return t in ['$', '$$']


def rating_meets_threshold(text: Optional[str], threshold: float = 4.0) -> bool:
    num = extract_float(text)
    if num is None:
        return False
    return num >= threshold


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
    caranddriver_data = await evaluator.extract(
        prompt=prompt_extract_caranddriver_data(),
        template_class=CarAndDriverData,
        extraction_name="caranddriver_data"
    )

    supercharger_data = await evaluator.extract(
        prompt=prompt_extract_supercharger_stations(),
        template_class=SuperchargerStations,
        extraction_name="supercharger_stations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver 75-mph highway range test for 2024 Tesla Model Y Long Range",
        parent=root,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Extract 75-mph range test result
    range_text_ok = bool(caranddriver_data.tested_75mph_range_text and
                         contains_digits(caranddriver_data.tested_75mph_range_text))
    range_units_ok = looks_like_miles(caranddriver_data.tested_75mph_range_text) if caranddriver_data.tested_75mph_range_text else False
    range_keyword_ok = has_any_ci(answer, ['75-mph', '75 mph', '75mph'])

    evaluator.add_custom_node(
        result=bool(range_text_ok and (range_units_ok or range_keyword_ok)),
        id="caranddriver_perception_75mph_range",
        desc="[Perception Node] caranddriver.com:F1:P2 - Extract the 75-mph highway range test result",
        parent=caranddriver_node,
        critical=False
    )

    # Safe maximum distance calculation (80% of tested range)
    safe_distance_ok = bool(caranddriver_data.safe_maximum_distance_text and
                           contains_digits(caranddriver_data.safe_maximum_distance_text))
    percent_mention = has_any_ci(answer, ['80%', '80 percent', 'eighty percent'])

    evaluator.add_custom_node(
        result=bool(safe_distance_ok and (percent_mention or has_any_ci(answer, ['safe', 'maximum']))),
        id="caranddriver_safe_distance_calculation",
        desc="Calculate safe maximum distance as 80% of tested 75-mph range",
        parent=caranddriver_node,
        critical=False
    )

    # Article link provided
    link_ok = looks_like_url(caranddriver_data.article_link)
    caranddriver_mention = has_any_ci(answer, ['car and driver', 'caranddriver'])

    evaluator.add_custom_node(
        result=bool(link_ok and caranddriver_mention),
        id="caranddriver_article_link",
        desc="Provide Car and Driver article link",
        parent=caranddriver_node,
        critical=False
    )

    # 3.2 Google Maps section
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps route planning and Supercharger station identification",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Plan route from Griffith Observatory to Bellagio Hotel
    route_mention_ok = (has_any_ci(answer, ['griffith observatory']) and
                        has_any_ci(answer, ['bellagio']))
    maps_mention = has_any_ci(answer, ['google maps', 'maps'])

    evaluator.add_custom_node(
        result=bool(route_mention_ok and maps_mention),
        id="googlemaps_action_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Plan route from Griffith Observatory to Bellagio Hotel",
        parent=googlemaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A20 - Search along route for Supercharger stations
    supercharger_mention = has_any_ci(answer, ['supercharger'])
    along_route_mention = has_any_ci(answer, ['along route', 'along the route', 'search along'])
    stations_found = len(supercharger_data.stations) if supercharger_data.stations else 0

    evaluator.add_custom_node(
        result=bool(supercharger_mention and (along_route_mention or stations_found > 0)),
        id="googlemaps_action_search_along_route",
        desc="[Action Node] maps.google.com:F2:A20 - Use 'Search along route' to find Supercharger stations",
        parent=googlemaps_node,
        critical=False
    )

    # Distance range constraint (150 miles to safe maximum distance)
    distance_constraint_ok = has_any_ci(answer, ['150 mile']) or has_any_ci(answer, ['150mi'])

    evaluator.add_custom_node(
        result=bool(distance_constraint_ok and stations_found > 0),
        id="googlemaps_distance_constraint",
        desc="Identify stations between 150 miles and safe maximum distance from origin",
        parent=googlemaps_node,
        critical=False
    )

    # Station count (target 3, accept fewer if noted)
    station_count_ok = stations_found >= 3 or (stations_found > 0 and supercharger_data.station_count_note)

    evaluator.add_custom_node(
        result=bool(station_count_ok),
        id="googlemaps_station_count",
        desc="Provide 3 Supercharger stations (or note if fewer are available)",
        parent=googlemaps_node,
        critical=False
    )

    # 3.3 Yelp section (for each station's restaurant)
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp restaurant search near each Supercharger station",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Search for restaurants near charging stations
    yelp_mention = has_any_ci(answer, ['yelp'])
    restaurants_found = sum(1 for s in supercharger_data.stations if s.restaurant_name) if supercharger_data.stations else 0

    evaluator.add_custom_node(
        result=bool(yelp_mention and restaurants_found > 0),
        id="yelp_action_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for restaurants near charging station addresses",
        parent=yelp_node,
        critical=False
    )

    # Distance constraint (0.2 miles, expandable to 0.5 miles)
    distance_mention = has_any_ci(answer, ['0.2 mile', '0.5 mile', '.2 mile', '.5 mile'])

    evaluator.add_custom_node(
        result=bool(distance_mention and restaurants_found > 0),
        id="yelp_distance_constraint",
        desc="Search within 0.2 miles (or expand to 0.5 miles if needed)",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Rating must be 4.0+
    ratings_ok = 0
    for station in supercharger_data.stations:
        if station.restaurant_rating_text and rating_meets_threshold(station.restaurant_rating_text, 4.0):
            ratings_ok += 1

    evaluator.add_custom_node(
        result=bool(ratings_ok > 0),
        id="yelp_perception_rating",
        desc="[Perception Node] yelp.com:F1:P1 - Restaurant rating must be 4.0 or higher",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Filter by price tier ($$ or lower)
    price_tiers_ok = 0
    for station in supercharger_data.stations:
        if station.restaurant_price_tier and is_valid_price_tier(station.restaurant_price_tier):
            price_tiers_ok += 1

    evaluator.add_custom_node(
        result=bool(price_tiers_ok > 0),
        id="yelp_action_price_filter",
        desc="[Action Node] yelp.com:F1:A2 - Filter restaurants by price level ($$ or lower)",
        parent=yelp_node,
        critical=False
    )

    # Yelp detail page links provided
    yelp_links_ok = sum(1 for s in supercharger_data.stations if looks_like_url(s.restaurant_yelp_link)) if supercharger_data.stations else 0

    evaluator.add_custom_node(
        result=bool(yelp_links_ok > 0),
        id="yelp_detail_links",
        desc="Provide Yelp detail page links for recommended restaurants",
        parent=yelp_node,
        critical=False
    )

    # 3.4 Output completeness checks (non-prefixed nodes)
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required output elements are provided",
        parent=root,
        critical=False
    )

    # Station names and distances from origin
    station_names_ok = sum(1 for s in supercharger_data.stations if s.station_name) if supercharger_data.stations else 0
    station_distances_ok = sum(1 for s in supercharger_data.stations if looks_like_miles(s.distance_from_origin_text)) if supercharger_data.stations else 0

    evaluator.add_custom_node(
        result=bool(station_names_ok > 0 and station_distances_ok > 0),
        id="output_station_info",
        desc="Provide station names and distances from origin for each charging station",
        parent=output_node,
        critical=False
    )

    # Restaurant information completeness
    restaurant_names_ok = sum(1 for s in supercharger_data.stations if s.restaurant_name) if supercharger_data.stations else 0
    restaurant_ratings_ok = sum(1 for s in supercharger_data.stations if looks_like_rating(s.restaurant_rating_text)) if supercharger_data.stations else 0
    restaurant_prices_ok = sum(1 for s in supercharger_data.stations if looks_like_price_tier(s.restaurant_price_tier)) if supercharger_data.stations else 0

    evaluator.add_custom_node(
        result=bool(restaurant_names_ok > 0 and restaurant_ratings_ok > 0 and restaurant_prices_ok > 0),
        id="output_restaurant_info",
        desc="Provide complete restaurant information (name, rating, price tier) for each station",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
