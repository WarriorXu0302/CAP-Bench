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
TASK_ID = "task-c6f472"
TASK_DESCRIPTION = 'I plan to invest in a rental property in Austin, Texas, targeting families with children.\n\nFirst, search for active listings in Austin on Redfin. Filter for single-family homes or townhouses with 2-3 bedrooms, priced between $250,000 and $400,000. Identify 5 properties that meet these criteria.\n\nFor each of these properties, visit GreatSchools (school district information is available on the Redfin detail page) to check the elementary school rating for its corresponding school district. Retain only properties with a school rating of 8 or higher.\n\nNext, use Google Maps to calculate the driving commute time from these properties to downtown Austin (e.g., Texas State Capitol). Filter out properties with a commute time exceeding 30 minutes.\n\nFinally, on Yelp, find the number of restaurants, supermarkets, and parks within 1 mile of each property. Count the number of facilities with a rating of 4 stars or higher.\n\nFor each final retained property, output the following information: Redfin listing link, address, asking price, number of bedrooms/bathrooms, property type, corresponding elementary school name, elementary school rating, GreatSchools link (if available on Redfin page), driving time to downtown, Google Maps route link, number of 4-star-and-above restaurants within 1 mile, number of supermarkets within 1 mile, number of parks within 1 mile, and Yelp search results page link.\n\nIf the total number of surrounding amenities (restaurants, supermarkets, parks) for a property is less than 10, state the reason and proceed to process other properties.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyInfo(BaseModel):
    """Information for a single property extracted from the answer"""
    redfin_link: Optional[str] = None
    address: Optional[str] = None
    price: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    property_type: Optional[str] = None
    elementary_school_name: Optional[str] = None
    school_rating: Optional[str] = None
    greatschools_link: Optional[str] = None
    commute_time: Optional[str] = None
    maps_route_link: Optional[str] = None
    restaurants_count: Optional[str] = None
    supermarkets_count: Optional[str] = None
    parks_count: Optional[str] = None
    yelp_link: Optional[str] = None


class PropertiesList(BaseModel):
    """List of all properties extracted from the answer"""
    properties: List[PropertyInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
Extract all property information from the answer. For each property that was retained after all filtering steps, extract:

- redfin_link: the Redfin listing URL
- address: the full property address
- price: the asking price
- bedrooms: number of bedrooms
- bathrooms: number of bathrooms
- property_type: type of property (House, Townhouse, etc.)
- elementary_school_name: name of the elementary school
- school_rating: the school rating (should be 8 or higher)
- greatschools_link: GreatSchools URL if mentioned
- commute_time: driving time to downtown Austin
- maps_route_link: Google Maps route URL
- restaurants_count: number of 4-star+ restaurants within 1 mile
- supermarkets_count: number of supermarkets within 1 mile
- parks_count: number of parks within 1 mile
- yelp_link: Yelp search results page URL

If any field is missing, set it to null. Return all properties found in the answer.
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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def extract_integer(text: Optional[str]) -> Optional[int]:
    num = extract_number(text)
    if num is None:
        return None
    return int(num)


def is_austin_address(address: Optional[str]) -> bool:
    if not address:
        return False
    return has_any_ci(address, ['austin', 'tx', 'texas'])


def is_valid_price_range(price: Optional[str]) -> bool:
    num = extract_number(price)
    if num is None:
        return False
    # Handle both raw numbers and formatted prices
    if num < 1000:  # Likely in thousands
        num = num * 1000
    return 250000 <= num <= 400000


def is_valid_bedroom_count(bedrooms: Optional[str]) -> bool:
    num = extract_integer(bedrooms)
    if num is None:
        return False
    return 2 <= num <= 3


def is_house_or_townhouse(property_type: Optional[str]) -> bool:
    if not property_type:
        return False
    return has_any_ci(property_type, ['house', 'townhouse', 'single-family', 'single family'])


def is_school_rating_8_plus(rating: Optional[str]) -> bool:
    num = extract_number(rating)
    if num is None:
        return False
    return num >= 8


def is_commute_30_min_or_less(commute: Optional[str]) -> bool:
    num = extract_number(commute)
    if num is None:
        return False
    return num <= 30


def count_properties(properties: List[PropertyInfo]) -> int:
    return len([p for p in properties if p.address])


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
    properties_data = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesList,
        extraction_name="properties_list"
    )

    properties = properties_data.properties if properties_data else []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Redfin search and filtering section
    redfin_node = evaluator.add_sequential(
        id="redfin_section",
        desc="Redfin property search and filtering in Austin",
        parent=root,
        critical=False
    )

    # [Action Node] redfin.com:F1:A18 - City search with cascade selection
    city_search_ok = has_any_ci(answer, ['redfin', 'austin'])
    addresses_have_austin = any(is_austin_address(p.address) for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(city_search_ok and addresses_have_austin),
        id="redfin_city_search",
        desc="[Action Node] redfin.com:F1:A18 - Search for Austin listings with cascade city selection",
        parent=redfin_node,
        critical=False
    )

    # [Action Node] redfin.com:F2:A1 - Price range slider filter
    price_filter_ok = has_any_ci(answer, ['250', '400', 'price'])
    prices_in_range = all(is_valid_price_range(p.price) for p in properties if p.price) if properties else False
    evaluator.add_custom_node(
        result=bool(price_filter_ok and (prices_in_range or not properties)),
        id="redfin_price_filter",
        desc="[Action Node] redfin.com:F2:A1 - Apply price range filter ($250,000-$400,000)",
        parent=redfin_node,
        critical=False
    )

    # [Action Node] redfin.com:F2:A2 - Property type multi-select checkbox
    type_filter_ok = has_any_ci(answer, ['house', 'townhouse', 'single-family'])
    types_match = all(is_house_or_townhouse(p.property_type) for p in properties if p.property_type) if properties else False
    evaluator.add_custom_node(
        result=bool(type_filter_ok and (types_match or not properties)),
        id="redfin_property_type_filter",
        desc="[Action Node] redfin.com:F2:A2 - Filter for House and Townhouse property types",
        parent=redfin_node,
        critical=False
    )

    # [Action Node] redfin.com:F2:A10 - Bedroom count filter
    bedroom_filter_ok = has_any_ci(answer, ['bedroom', '2', '3', 'bed'])
    bedrooms_match = all(is_valid_bedroom_count(p.bedrooms) for p in properties if p.bedrooms) if properties else False
    evaluator.add_custom_node(
        result=bool(bedroom_filter_ok and (bedrooms_match or not properties)),
        id="redfin_bedroom_filter",
        desc="[Action Node] redfin.com:F2:A10 - Filter for 2-3 bedrooms",
        parent=redfin_node,
        critical=False
    )

    # [Action Node] redfin.com:F2:A17 - Pagination to find 5 properties
    property_count = count_properties(properties)
    pagination_ok = property_count >= 3
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="redfin_pagination",
        desc="[Action Node] redfin.com:F2:A17 - Browse multiple pages to identify sufficient properties",
        parent=redfin_node,
        critical=False
    )

    # [Perception Node] redfin.com:F1:P1 - Identify listing status markers
    status_markers_ok = has_any_ci(answer, ['active', 'listing', 'for sale'])
    evaluator.add_custom_node(
        result=bool(status_markers_ok),
        id="redfin_status_markers",
        desc="[Perception Node] redfin.com:F1:P1 - Identify active listing status markers",
        parent=redfin_node,
        critical=False
    )

    # 3.2 School rating section
    school_node = evaluator.add_sequential(
        id="school_section",
        desc="School district rating verification via GreatSchools",
        parent=root,
        critical=False
    )

    # [Action Node] redfin.com:F4:A9 - Navigate to Schools tab
    schools_tab_ok = has_any_ci(answer, ['school', 'greatschools'])
    schools_have_names = any(p.elementary_school_name for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(schools_tab_ok and schools_have_names),
        id="redfin_schools_tab",
        desc="[Action Node] redfin.com:F4:A9 - Navigate to Schools tab in property details",
        parent=school_node,
        critical=False
    )

    # [Perception Node] redfin.com:F12:P14 - Compare and filter school ratings
    ratings_8_plus = all(is_school_rating_8_plus(p.school_rating) for p in properties if p.school_rating) if properties else False
    has_ratings = any(p.school_rating for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(has_ratings and ratings_8_plus),
        id="school_rating_filter",
        desc="[Perception Node] redfin.com:F12:P14 - Verify school ratings are 8 or higher",
        parent=school_node,
        critical=False
    )

    # 3.3 Google Maps commute section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps commute time calculation",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input origin and destination
    maps_route_ok = has_any_ci(answer, ['google maps', 'commute', 'downtown', 'capitol'])
    has_commute_times = any(p.commute_time for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(maps_route_ok and has_commute_times),
        id="maps_route_input",
        desc="[Action Node] maps.google.com:F2:A5 - Input property address and Texas State Capitol for route planning",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode
    driving_mode_ok = has_any_ci(answer, ['driving', 'drive', 'car'])
    commutes_30_or_less = all(is_commute_30_min_or_less(p.commute_time) for p in properties if p.commute_time) if properties else False
    evaluator.add_custom_node(
        result=bool(driving_mode_ok and (commutes_30_or_less or not has_commute_times)),
        id="maps_driving_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode and verify 30-minute threshold",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P12 - Identify map markers
    has_route_links = any(p.maps_route_link for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(has_route_links),
        id="maps_markers",
        desc="[Perception Node] maps.google.com:F3:P12 - Identify location markers on map",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] redfin.com:F11:P13 - Multi-criteria comprehensive judgment
    multi_criteria_ok = all(
        p.price and p.bedrooms and p.school_rating and p.commute_time
        for p in properties if p.address
    ) if properties else False
    evaluator.add_custom_node(
        result=bool(multi_criteria_ok and property_count > 0),
        id="multi_criteria_filter",
        desc="[Perception Node] redfin.com:F11:P13 - Verify all properties meet combined criteria (price, bedrooms, school, commute)",
        parent=root,
        critical=False
    )

    # 3.4 Yelp amenities section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp amenities search and counting",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Keyword search for amenities
    yelp_search_ok = has_any_ci(answer, ['yelp', 'restaurant', 'supermarket', 'park'])
    has_amenity_counts = any(
        p.restaurants_count or p.supermarkets_count or p.parks_count
        for p in properties
    ) if properties else False
    evaluator.add_custom_node(
        result=bool(yelp_search_ok and has_amenity_counts),
        id="yelp_keyword_search",
        desc="[Action Node] yelp.com:F1:A1 - Search for restaurants, supermarkets, and parks near each property",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Multi-condition filter panel
    yelp_filter_ok = has_any_ci(answer, ['4 star', '4-star', 'rating', '1 mile'])
    evaluator.add_custom_node(
        result=bool(yelp_filter_ok),
        id="yelp_filter_panel",
        desc="[Action Node] yelp.com:F1:A2 - Apply rating (4+ stars) and distance (1 mile) filters",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Identify business rating and distance
    amenity_counts_valid = all(
        (extract_integer(p.restaurants_count) is not None if p.restaurants_count else True) and
        (extract_integer(p.supermarkets_count) is not None if p.supermarkets_count else True) and
        (extract_integer(p.parks_count) is not None if p.parks_count else True)
        for p in properties if p.address
    ) if properties else False
    evaluator.add_custom_node(
        result=bool(amenity_counts_valid and has_amenity_counts),
        id="yelp_business_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Extract rating and distance from business cards",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F9:A19 - Map drag and zoom
    yelp_map_ok = has_any_ci(answer, ['yelp', 'map', 'mile'])
    has_yelp_links = any(p.yelp_link for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(yelp_map_ok or has_yelp_links),
        id="yelp_map_interaction",
        desc="[Action Node] yelp.com:F9:A19 - Adjust map view to see 1-mile radius",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F9:P20 - Identify business distribution
    distribution_ok = has_amenity_counts and property_count > 0
    evaluator.add_custom_node(
        result=bool(distribution_ok),
        id="yelp_distribution",
        desc="[Perception Node] yelp.com:F9:P20 - Identify geographic distribution of amenities",
        parent=yelp_node,
        critical=False
    )

    # 3.5 Output completeness checks
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify completeness of final property information",
        parent=root,
        critical=False
    )

    # Check for required output fields
    has_redfin_links = any(p.redfin_link for p in properties) if properties else False
    has_addresses = any(p.address for p in properties) if properties else False
    has_prices = any(p.price for p in properties) if properties else False

    evaluator.add_custom_node(
        result=bool(has_redfin_links and has_addresses and has_prices),
        id="basic_property_info",
        desc="Output includes Redfin links, addresses, and prices",
        parent=output_node,
        critical=False
    )

    has_school_info = any(p.elementary_school_name and p.school_rating for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(has_school_info),
        id="school_info_output",
        desc="Output includes school names and ratings",
        parent=output_node,
        critical=False
    )

    has_commute_info = any(p.commute_time for p in properties) if properties else False
    evaluator.add_custom_node(
        result=bool(has_commute_info),
        id="commute_info_output",
        desc="Output includes commute times",
        parent=output_node,
        critical=False
    )

    has_amenities_info = any(
        p.restaurants_count and p.supermarkets_count and p.parks_count
        for p in properties
    ) if properties else False
    evaluator.add_custom_node(
        result=bool(has_amenities_info),
        id="amenities_info_output",
        desc="Output includes amenity counts (restaurants, supermarkets, parks)",
        parent=output_node,
        critical=False
    )

    # Check if answer addresses the 10-amenity threshold
    threshold_mentioned = has_any_ci(answer, ['less than 10', 'fewer than 10', 'below 10', 'under 10'])
    evaluator.add_custom_node(
        result=bool(threshold_mentioned or property_count > 0),
        id="amenity_threshold_check",
        desc="Addresses the 10-amenity threshold requirement",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
