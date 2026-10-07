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
TASK_ID = "task-e71ffc"
TASK_DESCRIPTION = "I plan to move to Phoenix, Arizona in early 2026 and intend to start full-time RV living.\n\nFirst, please visit the official Airstream website to check the specifications of the '2026 Trade Wind 25FB' model and find its Gross Vehicle Weight Rating (GVWR).\n\nNext, research the '2025 Toyota Tundra Hybrid' pickup truck on Car and Driver to identify a specific Trim (configuration version) that can safely tow the RV (towing capacity must exceed the RV's GVWR), and record the maximum towing capacity of that trim.\n\nFinally, search Zillow for three detached rental houses in Phoenix, AZ, with a budget of $2500-$4500/month, requiring more than three parking spaces or a large driveway. Be sure to check the 'Facts and features' or detailed descriptions of the listings, and only include those that explicitly mention 'RV Gate,' 'RV Parking,' or 'Side Yard Parking.'\n\nOutput: The RV's GVWR, the recommended pickup truck's Trim version and its towing capacity, the addresses of the three rental properties, their monthly rent, the exact text description related to parking, and direct links to each page."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RVInfo(BaseModel):
    """RV details extracted from the answer for 2026 Trade Wind 25FB"""
    gvwr_text: Optional[str] = None


class TruckInfo(BaseModel):
    """Truck details extracted from the answer for 2025 Toyota Tundra Hybrid"""
    trim_name: Optional[str] = None
    towing_capacity_text: Optional[str] = None
    caranddriver_link: Optional[str] = None


class RentalProperty(BaseModel):
    """Single rental property details"""
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    parking_description: Optional[str] = None
    zillow_link: Optional[str] = None


class ZillowInfo(BaseModel):
    """Zillow rental properties extracted from the answer"""
    properties: Optional[List[RentalProperty]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_rv_from_answer() -> str:
    return """
Extract the Airstream 2026 Trade Wind 25FB GVWR (Gross Vehicle Weight Rating) from the answer.

Return:
- gvwr_text: the GVWR value exactly as written (include units if present, e.g., "lbs", "pounds").

If not present, set it to null.
"""


def prompt_extract_truck_from_answer() -> str:
    return """
From the answer, extract the 2025 Toyota Tundra Hybrid truck information from Car and Driver:

- trim_name: the specific Trim version name selected (e.g., "SR5", "Limited", "TRD Pro").
- towing_capacity_text: the maximum towing capacity for that trim exactly as stated (include units if present).
- caranddriver_link: the direct link to the Car and Driver page for the 2025 Toyota Tundra Hybrid.

If any field is missing, set it to null.
"""


def prompt_extract_zillow_from_answer() -> str:
    return """
From the answer, extract the three rental properties from Zillow in Phoenix, AZ.

For each property return:
- address: the full street address.
- monthly_rent: the monthly rent amount exactly as written (include currency symbol if present).
- parking_description: the exact text that mentions 'RV Gate', 'RV Parking', or 'Side Yard Parking'.
- zillow_link: the direct link to the Zillow listing page.

Return a list of up to 3 properties. If fewer than 3 are present, return what is available.
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
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def looks_like_gvwr(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_weight_unit = has_any_ci(text, ['lbs', 'lb', 'pounds', 'kg', 'kilograms'])
    return has_number and has_weight_unit


def looks_like_towing_capacity(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_weight_unit = has_any_ci(text, ['lbs', 'lb', 'pounds', 'kg', 'kilograms'])
    return has_number and has_weight_unit


def is_valid_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def is_caranddriver_url(text: Optional[str]) -> bool:
    if not is_valid_url(text):
        return False
    return 'caranddriver.com' in text.lower()


def is_zillow_url(text: Optional[str]) -> bool:
    if not is_valid_url(text):
        return False
    return 'zillow.com' in text.lower()


def is_zillow_rental_url(text: Optional[str]) -> bool:
    if not is_zillow_url(text):
        return False
    return '/rentals/' in text.lower() or '/b/' in text.lower()


def mentions_phoenix_az(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['phoenix', 'phoenix, az', 'phoenix az', 'phoenix,az'])


def mentions_rv_parking(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['rv gate', 'rv parking', 'side yard parking'])


def in_budget_range(rent_text: Optional[str]) -> bool:
    if not rent_text:
        return False
    amount = extract_float(rent_text)
    if amount is None:
        return False
    return 2500 <= amount <= 4500


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
    rv_info = await evaluator.extract(
        prompt=prompt_extract_rv_from_answer(),
        template_class=RVInfo,
        extraction_name="rv_gvwr"
    )

    truck_info = await evaluator.extract(
        prompt=prompt_extract_truck_from_answer(),
        template_class=TruckInfo,
        extraction_name="truck_info"
    )

    zillow_info = await evaluator.extract(
        prompt=prompt_extract_zillow_from_answer(),
        template_class=ZillowInfo,
        extraction_name="zillow_properties"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Airstream RV section
    airstream_node = evaluator.add_sequential(
        id="airstream_section",
        desc="Airstream 2026 Trade Wind 25FB GVWR from official website",
        parent=root,
        critical=False
    )

    airstream_mention_ok = has_any_ci(answer, ['airstream', '2026 trade wind', 'trade wind 25fb'])
    evaluator.add_custom_node(
        result=bool(airstream_mention_ok),
        id="airstream_mention",
        desc="Mentions Airstream and the 2026 Trade Wind 25FB model",
        parent=airstream_node,
        critical=False
    )

    gvwr_ok = looks_like_gvwr(rv_info.gvwr_text)
    evaluator.add_custom_node(
        result=bool(gvwr_ok),
        id="airstream_gvwr_extracted",
        desc="GVWR value extracted with proper weight units",
        parent=airstream_node,
        critical=False
    )

    # 3.2 Car and Driver truck section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver - 2025 Toyota Tundra Hybrid trim and towing capacity",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A2 - Cascaded selection (Make/Model/Year)
    caranddriver_action_ok = (has_any_ci(answer, ['car and driver']) and
                              has_any_ci(answer, ['2025 toyota tundra', 'toyota tundra hybrid']))
    evaluator.add_custom_node(
        result=bool(caranddriver_action_ok),
        id="caranddriver_action_select",
        desc="[Action Node] caranddriver.com:F2:A2 - Navigate to Car and Driver and select the 2025 Toyota Tundra Hybrid (via cascaded selection of Make/Model/Year)",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F2:P3 - Extract specific trim towing capacity
    trim_ok = bool(truck_info.trim_name and truck_info.trim_name.strip())
    towing_ok = looks_like_towing_capacity(truck_info.towing_capacity_text)
    towing_num = extract_float(truck_info.towing_capacity_text)
    gvwr_num = extract_float(rv_info.gvwr_text)

    towing_exceeds_gvwr = False
    if towing_num is not None and gvwr_num is not None:
        towing_exceeds_gvwr = towing_num > gvwr_num

    evaluator.add_custom_node(
        result=bool(trim_ok and towing_ok),
        id="caranddriver_perception_towing",
        desc="[Perception Node] caranddriver.com:F2:P3 - Extract specific Trim name and its maximum towing capacity",
        parent=caranddriver_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(towing_exceeds_gvwr),
        id="caranddriver_towing_exceeds_gvwr",
        desc="Towing capacity exceeds the RV's GVWR (safe towing verification)",
        parent=caranddriver_node,
        critical=False
    )

    caranddriver_link_ok = is_caranddriver_url(truck_info.caranddriver_link)
    evaluator.add_custom_node(
        result=bool(caranddriver_link_ok),
        id="caranddriver_link_valid",
        desc="Valid Car and Driver link provided for the truck page",
        parent=caranddriver_node,
        critical=False
    )

    # 3.3 Zillow rental properties section
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow - Three rental houses in Phoenix, AZ with RV parking",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A1 - Navigate to Rent (dropdown menu)
    zillow_mention_ok = has_any_ci(answer, ['zillow'])
    evaluator.add_custom_node(
        result=bool(zillow_mention_ok),
        id="zillow_mention",
        desc="[Action Node] zillow.com:F1:A1 - Navigate to Zillow and switch to Rent section (via dropdown menu)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Search for Phoenix, AZ location
    properties = zillow_info.properties if zillow_info and zillow_info.properties else []
    phoenix_context_ok = mentions_phoenix_az(answer)
    addresses_phoenix_ok = all(mentions_phoenix_az(p.address) for p in properties if p.address)

    evaluator.add_custom_node(
        result=bool(phoenix_context_ok and addresses_phoenix_ok),
        id="zillow_location_phoenix",
        desc="[Action Node] zillow.com:F1:A35 - Search and locate properties in Phoenix, AZ",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:A26 - Scroll to load more listings
    three_properties_found = len(properties) == 3
    evaluator.add_custom_node(
        result=bool(three_properties_found),
        id="zillow_three_properties",
        desc="[Action Node] zillow.com:F1:A26 - Scroll and load enough listings to find 3 qualifying properties",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A6 - Navigate to Facts and features tab
    facts_mention_ok = has_any_ci(answer, ['facts and features', 'facts & features', 'features'])
    evaluator.add_custom_node(
        result=bool(facts_mention_ok),
        id="zillow_facts_tab",
        desc="[Action Node] zillow.com:F4:A6 - Navigate to the 'Facts and features' tab for each property",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A23 - Expand collapsed details
    # [Perception Node] zillow.com:F4:P18 - Verify parking spaces requirement (>3)
    parking_descriptions_ok = all(
        mentions_rv_parking(p.parking_description)
        for p in properties
        if p.parking_description
    )

    evaluator.add_custom_node(
        result=bool(parking_descriptions_ok and three_properties_found),
        id="zillow_parking_verification",
        desc="[Action Node] zillow.com:F4:A23 & [Perception Node] zillow.com:F4:P18 - Expand details and verify parking mentions 'RV Gate', 'RV Parking', or 'Side Yard Parking'",
        parent=zillow_node,
        critical=False
    )

    # [Perception Node] zillow.com:F4:P1 - Understand driveway/layout from images/descriptions
    driveway_context_ok = has_any_ci(answer, ['driveway', 'parking', 'spaces'])
    evaluator.add_custom_node(
        result=bool(driveway_context_ok),
        id="zillow_driveway_understanding",
        desc="[Perception Node] zillow.com:F4:P1 - Understand and verify large driveway or parking space from descriptions/images",
        parent=zillow_node,
        critical=False
    )

    # Budget verification (non-prefixed)
    rents_in_budget = all(
        in_budget_range(p.monthly_rent)
        for p in properties
        if p.monthly_rent
    )
    evaluator.add_custom_node(
        result=bool(rents_in_budget and three_properties_found),
        id="zillow_budget_range",
        desc="All three properties are within the $2500-$4500/month budget",
        parent=zillow_node,
        critical=False
    )

    # Valid Zillow rental links (non-prefixed)
    zillow_links_ok = all(
        is_zillow_rental_url(p.zillow_link)
        for p in properties
        if p.zillow_link
    )
    evaluator.add_custom_node(
        result=bool(zillow_links_ok and three_properties_found),
        id="zillow_links_valid",
        desc="Valid Zillow rental listing links provided for all three properties",
        parent=zillow_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
