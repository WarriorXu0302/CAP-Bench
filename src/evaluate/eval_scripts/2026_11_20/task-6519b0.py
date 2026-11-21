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
TASK_ID = "task-6519b0"
TASK_DESCRIPTION = "I am evaluating the investment potential for vacation rentals in Kissimmee, Florida (34747), with a baseline for analysis set for December 2025. First, on Zillow, find three single-family homes for sale. Requirements: price between $450,000 and $600,000, at least 4 bedrooms, and explicitly listed with a 'Private Pool' in the 'Facts and features' section.\n\nFor each of these three Zillow properties, find comparable Airbnb listings to estimate potential rental income. Using the map search function, within the same neighborhood or a 0.5-mile radius of each Zillow property, find a short-term rental property that also has 4 bedrooms and is tagged as 'Guest favorite'. Check the nightly rate and cleaning fee for this comparable listing for a 5-night period, specifically from March 15th to 20th of the next calendar year (e.g., if the current date is in 2024, use March 15-20, 2025).\n\nFinally, using Google Maps, calculate the driving time from each Zillow property to 'Magic Kingdom Park' and verify if it is within a 20-minute drive.\n\n**Output: Zillow property address, sale price, Zillow link, Airbnb comparable listing link, Airbnb nightly rate (for the specified March 15-20 period), cleaning fee, whether it is a 'Guest Favorite', and driving time to Magic Kingdom Park.**"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ZillowProperty(BaseModel):
    """Single Zillow property extracted from the answer"""
    address: Optional[str] = None
    sale_price: Optional[str] = None
    zillow_link: Optional[str] = None


class AirbnbComparable(BaseModel):
    """Airbnb comparable listing extracted from the answer"""
    airbnb_link: Optional[str] = None
    nightly_rate: Optional[str] = None
    cleaning_fee: Optional[str] = None
    guest_favorite: Optional[bool] = None


class DrivingTime(BaseModel):
    """Driving time extracted from the answer"""
    driving_time_text: Optional[str] = None


class AllProperties(BaseModel):
    """All three properties with their comparables"""
    property1_zillow: Optional[ZillowProperty] = None
    property1_airbnb: Optional[AirbnbComparable] = None
    property1_driving_time: Optional[DrivingTime] = None

    property2_zillow: Optional[ZillowProperty] = None
    property2_airbnb: Optional[AirbnbComparable] = None
    property2_driving_time: Optional[DrivingTime] = None

    property3_zillow: Optional[ZillowProperty] = None
    property3_airbnb: Optional[AirbnbComparable] = None
    property3_driving_time: Optional[DrivingTime] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_properties() -> str:
    return """
Extract all three Zillow properties and their corresponding Airbnb comparables and driving times from the answer.

For each of the three properties, return:
- Zillow property: address, sale_price (as text with units), zillow_link (URL)
- Airbnb comparable: airbnb_link (URL), nightly_rate (as text with units), cleaning_fee (as text with units), guest_favorite (boolean)
- Driving time: driving_time_text (as stated, e.g., "15 minutes", "18 min")

If any field is missing for any property, set it to null. If a property is not mentioned at all, set all its fields to null.
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
    # Remove common currency symbols and commas
    cleaned = re.sub(r'[$,]', '', text)
    m = re.findall(r'(\d+(?:\.\d+)?)', cleaned)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Check if it's in the expected range (450k-600k)
    return 400000 <= num <= 650000


def looks_like_kissimmee(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['kissimmee', '34747'])


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_zillow_url(text: Optional[str]) -> bool:
    return looks_like_url(text) and ci_contains(text, 'zillow')


def looks_like_airbnb_url(text: Optional[str]) -> bool:
    return looks_like_url(text) and ci_contains(text, 'airbnb')


def looks_like_nightly_rate(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'dollar', 'night'])


def looks_like_fee(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['$', 'dollar', 'fee'])


def looks_like_driving_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour'])


def is_within_20_minutes(text: Optional[str]) -> bool:
    if not text:
        return False
    minutes = extract_float(text)
    if minutes is None:
        return False
    return minutes <= 20


def count_properties(all_props: AllProperties) -> int:
    count = 0
    if all_props.property1_zillow and all_props.property1_zillow.address:
        count += 1
    if all_props.property2_zillow and all_props.property2_zillow.address:
        count += 1
    if all_props.property3_zillow and all_props.property3_zillow.address:
        count += 1
    return count


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
    all_properties = await evaluator.extract(
        prompt=prompt_extract_all_properties(),
        template_class=AllProperties,
        extraction_name="all_properties"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Zillow section - finding properties
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow property search in Kissimmee, FL (34747)",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Location search for Kissimmee/34747
    location_mentioned = has_any_ci(answer, ['kissimmee', '34747'])
    prop_count = count_properties(all_properties)

    evaluator.add_custom_node(
        result=bool(location_mentioned and prop_count > 0),
        id="zillow_action_location_search",
        desc="[Action Node] zillow.com:F1:A35 - Search for properties in Kissimmee, FL (34747)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:A26 - Scroll to find multiple properties matching criteria
    found_three = prop_count >= 3
    evaluator.add_custom_node(
        result=bool(found_three),
        id="zillow_action_scroll_browse",
        desc="[Action Node] zillow.com:F1:A26 - Browse and scroll through listings to find three properties meeting all criteria",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A23 - Expand Facts and Features to verify Private Pool
    pool_mentioned = has_any_ci(answer, ['pool', 'private pool'])
    facts_mentioned = has_any_ci(answer, ['facts', 'features'])

    evaluator.add_custom_node(
        result=bool(pool_mentioned and (facts_mentioned or prop_count >= 3)),
        id="zillow_action_expand_facts",
        desc="[Action Node] zillow.com:F4:A23 - Expand 'Facts and features' section to verify Private Pool listing",
        parent=zillow_node,
        critical=False
    )

    # [Perception Node] zillow.com:F4:P18 - Understand and verify Private Pool in features
    evaluator.add_custom_node(
        result=bool(pool_mentioned and prop_count >= 3),
        id="zillow_perception_pool",
        desc="[Perception Node] zillow.com:F4:P18 - Understand and verify that properties have 'Private Pool' explicitly listed",
        parent=zillow_node,
        critical=False
    )

    # Check properties meet price and bedroom criteria
    props_with_valid_data = 0
    for i in [1, 2, 3]:
        zillow_prop = getattr(all_properties, f'property{i}_zillow', None)
        if zillow_prop and zillow_prop.address:
            price_ok = looks_like_price(zillow_prop.sale_price)
            address_ok = looks_like_kissimmee(zillow_prop.address) or looks_like_kissimmee(answer)
            link_ok = looks_like_zillow_url(zillow_prop.zillow_link)
            if price_ok and link_ok:
                props_with_valid_data += 1

    evaluator.add_custom_node(
        result=bool(props_with_valid_data >= 3),
        id="zillow_properties_complete",
        desc="All three Zillow properties have valid address, price ($450k-$600k), and Zillow link",
        parent=zillow_node,
        critical=False
    )

    # 3.2 Airbnb section - finding comparables
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Airbnb comparable listings for each Zillow property",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A38 - Map search within 0.5 mile radius
    map_mentioned = has_any_ci(answer, ['map', 'radius', '0.5', 'half mile', 'neighborhood'])
    airbnb_mentioned = has_any_ci(answer, ['airbnb'])

    evaluator.add_custom_node(
        result=bool(map_mentioned and airbnb_mentioned),
        id="airbnb_action_map_search",
        desc="[Action Node] airbnb.com:F1:A38 - Use map search to find listings within 0.5-mile radius of each Zillow property",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A10 - Filter for 4 bedrooms
    bedroom_filter = has_any_ci(answer, ['4 bedroom', 'four bedroom', 'bedrooms'])

    evaluator.add_custom_node(
        result=bool(bedroom_filter and airbnb_mentioned),
        id="airbnb_action_bedroom_filter",
        desc="[Action Node] airbnb.com:F1:A10 - Filter search results for 4-bedroom properties",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A7 - Navigate calendar to March 15-20
    march_mentioned = has_any_ci(answer, ['march', 'march 15', '3/15', 'march 20', '3/20'])
    calendar_interaction = march_mentioned or has_any_ci(answer, ['nightly rate', 'cleaning fee'])

    evaluator.add_custom_node(
        result=bool(calendar_interaction),
        id="airbnb_action_calendar_navigation",
        desc="[Action Node] airbnb.com:F1:A7 - Navigate calendar to select March 15-20 dates for pricing",
        parent=airbnb_node,
        critical=False
    )

    # [Perception Node] airbnb.com:F1:P3 - Identify Guest Favorite tag
    guest_fav_count = 0
    for i in [1, 2, 3]:
        airbnb_prop = getattr(all_properties, f'property{i}_airbnb', None)
        if airbnb_prop and airbnb_prop.guest_favorite:
            guest_fav_count += 1

    evaluator.add_custom_node(
        result=bool(guest_fav_count >= 3),
        id="airbnb_perception_guest_favorite",
        desc="[Perception Node] airbnb.com:F1:P3 - Identify and verify 'Guest favorite' tag on comparable listings",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F3:A28 - Expand pricing details to see cleaning fee
    cleaning_fee_count = 0
    nightly_rate_count = 0
    for i in [1, 2, 3]:
        airbnb_prop = getattr(all_properties, f'property{i}_airbnb', None)
        if airbnb_prop:
            if looks_like_fee(airbnb_prop.cleaning_fee):
                cleaning_fee_count += 1
            if looks_like_nightly_rate(airbnb_prop.nightly_rate):
                nightly_rate_count += 1

    evaluator.add_custom_node(
        result=bool(cleaning_fee_count >= 3 or (cleaning_fee_count >= 2 and has_any_ci(answer, ['cleaning fee']))),
        id="airbnb_action_expand_fees",
        desc="[Action Node] airbnb.com:F3:A28 - Expand pricing details to view cleaning fee",
        parent=airbnb_node,
        critical=False
    )

    # Check Airbnb comparables completeness
    airbnb_complete = 0
    for i in [1, 2, 3]:
        airbnb_prop = getattr(all_properties, f'property{i}_airbnb', None)
        if airbnb_prop:
            link_ok = looks_like_airbnb_url(airbnb_prop.airbnb_link)
            rate_ok = looks_like_nightly_rate(airbnb_prop.nightly_rate)
            fee_ok = looks_like_fee(airbnb_prop.cleaning_fee)
            if link_ok and rate_ok and fee_ok:
                airbnb_complete += 1

    evaluator.add_custom_node(
        result=bool(airbnb_complete >= 3),
        id="airbnb_comparables_complete",
        desc="All three Airbnb comparables have valid link, nightly rate, and cleaning fee",
        parent=airbnb_node,
        critical=False
    )

    # 3.3 Google Maps section - driving time verification
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps driving time calculation to Magic Kingdom Park",
        parent=root,
        critical=False
    )

    # Check if Google Maps and Magic Kingdom mentioned
    maps_mentioned = has_any_ci(answer, ['google maps', 'driving', 'drive time'])
    magic_kingdom = has_any_ci(answer, ['magic kingdom'])

    evaluator.add_custom_node(
        result=bool(maps_mentioned and magic_kingdom),
        id="maps_action_route_calculation",
        desc="Use Google Maps to calculate driving time from each property to Magic Kingdom Park",
        parent=maps_node,
        critical=False
    )

    # Check driving times provided
    driving_time_count = 0
    within_20_count = 0
    for i in [1, 2, 3]:
        driving = getattr(all_properties, f'property{i}_driving_time', None)
        if driving and looks_like_driving_time(driving.driving_time_text):
            driving_time_count += 1
            if is_within_20_minutes(driving.driving_time_text):
                within_20_count += 1

    evaluator.add_custom_node(
        result=bool(driving_time_count >= 3),
        id="maps_driving_times_provided",
        desc="Driving times to Magic Kingdom Park provided for all three properties",
        parent=maps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(within_20_count > 0 or has_any_ci(answer, ['within 20', 'under 20', '20 minute'])),
        id="maps_20min_verification",
        desc="Verification of whether properties are within 20-minute drive mentioned",
        parent=maps_node,
        critical=False
    )

    # 3.4 Overall completeness check
    overall_complete = (props_with_valid_data >= 3 and
                       airbnb_complete >= 3 and
                       driving_time_count >= 3)

    evaluator.add_custom_node(
        result=bool(overall_complete),
        id="complete_analysis",
        desc="Complete investment analysis with all required data for three properties",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
