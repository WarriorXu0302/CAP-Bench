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
TASK_ID = "task-3a846e"
TASK_DESCRIPTION = "I recently found a house in the distant suburbs of Houston that I'm interested in, located at 20306 Telge Rd, Tomball, TX 77377. Although the price is attractive, I'm concerned about high future commuting costs. I currently drive a 2022 Toyota RAV4, and I work in downtown Houston (using JPMorgan Chase Tower as the reference point).\n\nPlease help me calculate the commuting cost:\n1. First, check on Zillow to confirm if this house is still for sale. While you're there, please also find its build year and listing price.\n2. Use Google Maps to calculate how long it takes to drive from this house to JPMorgan Chase Tower at 8 AM on a Monday morning, and what the approximate one-way distance in miles is.\n3. Go to Car and Driver to find the actual highway MPG (miles per gallon) for my car (2022 Toyota RAV4) in their reviews.\n4. Finally, help me calculate the total: Assuming one round trip per day, 22 working days per month, and a gas price of $3 per gallon, how much would I likely spend on commuting fuel costs each month if I don't change my car?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ZillowHouseInfo(BaseModel):
    """Zillow house information extracted from the answer"""
    sale_status: Optional[str] = None
    build_year: Optional[str] = None
    listing_price: Optional[str] = None


class GoogleMapsRouteInfo(BaseModel):
    """Google Maps route information extracted from the answer"""
    travel_time: Optional[str] = None
    distance_miles: Optional[str] = None


class VehicleMPGInfo(BaseModel):
    """Vehicle MPG information extracted from the answer"""
    highway_mpg_text: Optional[str] = None


class CommutingCostInfo(BaseModel):
    """Commuting cost calculation extracted from the answer"""
    monthly_fuel_cost: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_zillow_info() -> str:
    return """
Extract the Zillow house information for 20306 Telge Rd, Tomball, TX 77377 from the answer.

Return:
- sale_status: whether the house is for sale or its current listing status (e.g., "For Sale", "Active", "Pending"). If not present, set null.
- build_year: the year the house was built exactly as stated. If not present, set null.
- listing_price: the listing price exactly as stated (include currency symbols if present). If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_maps_route_info() -> str:
    return """
Extract the Google Maps route information from 20306 Telge Rd to JPMorgan Chase Tower at 8 AM Monday morning from the answer.

Return:
- travel_time: the driving time exactly as stated (include units like "minutes" or "hours" if present).
- distance_miles: the one-way distance in miles exactly as stated (include units if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_vehicle_mpg() -> str:
    return """
Extract the Car and Driver highway MPG for the 2022 Toyota RAV4 from the answer.

Return:
- highway_mpg_text: the highway MPG value exactly as stated (include units if present).

If missing in the answer, set it to null.
"""


def prompt_extract_commuting_cost() -> str:
    return """
Extract the calculated monthly fuel commuting cost from the answer.

Return:
- monthly_fuel_cost: the monthly fuel cost exactly as stated (include currency symbols if present).

If missing in the answer, set it to null.
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


def looks_like_year(text: Optional[str]) -> bool:
    if not text:
        return False
    m = re.search(r'\b(19|20)\d{2}\b', text)
    return m is not None


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '$') or ci_contains(text, 'dollar'))


def looks_like_time_duration(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['min', 'minute', 'hour', 'hr'])


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km'])


def looks_like_mpg(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mpg', 'miles per gallon', 'mi/gal'])


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
    zillow_info = await evaluator.extract(
        prompt=prompt_extract_zillow_info(),
        template_class=ZillowHouseInfo,
        extraction_name="zillow_house_info"
    )

    maps_info = await evaluator.extract(
        prompt=prompt_extract_maps_route_info(),
        template_class=GoogleMapsRouteInfo,
        extraction_name="google_maps_route_info"
    )

    mpg_info = await evaluator.extract(
        prompt=prompt_extract_vehicle_mpg(),
        template_class=VehicleMPGInfo,
        extraction_name="vehicle_mpg_info"
    )

    cost_info = await evaluator.extract(
        prompt=prompt_extract_commuting_cost(),
        template_class=CommutingCostInfo,
        extraction_name="commuting_cost_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Zillow part
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow house information for 20306 Telge Rd, Tomball, TX 77377",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Search and select property by address
    zillow_address_ok = has_any_ci(answer, ['20306 telge', 'tomball', '77377']) and has_any_ci(answer, ['zillow'])
    evaluator.add_custom_node(
        result=bool(zillow_address_ok),
        id="zillow_action_search_address",
        desc="[Action Node] zillow.com:F1:A35 - Search for and select the property at 20306 Telge Rd, Tomball, TX 77377 on Zillow",
        parent=zillow_node,
        critical=False
    )

    # Check sale status
    sale_status_ok = bool(zillow_info.sale_status and zillow_info.sale_status.strip())
    evaluator.add_custom_node(
        result=bool(sale_status_ok),
        id="zillow_sale_status",
        desc="Identifies whether the house is for sale or its current listing status",
        parent=zillow_node,
        critical=False
    )

    # [Perception Node] zillow.com:F4:P18 - Extract build year from Facts and features
    build_year_ok = looks_like_year(zillow_info.build_year)
    evaluator.add_custom_node(
        result=bool(build_year_ok),
        id="zillow_perception_build_year",
        desc="[Perception Node] zillow.com:F4:P18 - Extract the build year from Facts and features section",
        parent=zillow_node,
        critical=False
    )

    # Extract listing price
    price_ok = looks_like_price(zillow_info.listing_price)
    evaluator.add_custom_node(
        result=bool(price_ok),
        id="zillow_listing_price",
        desc="Extract the listing price from the Zillow page",
        parent=zillow_node,
        critical=False
    )

    # 3.2 Google Maps part
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route from house to JPMorgan Chase Tower at 8 AM Monday",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Set origin and destination
    maps_route_ok = (has_any_ci(answer, ['google maps']) and
                     has_any_ci(answer, ['20306 telge', 'tomball']) and
                     has_any_ci(answer, ['jpmorgan', 'chase tower', 'downtown houston']))
    evaluator.add_custom_node(
        result=bool(maps_route_ok),
        id="maps_action_route_input",
        desc="[Action Node] maps.google.com:F2:A5 - Enter origin (20306 Telge Rd) and destination (JPMorgan Chase Tower) for route planning",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select driving mode and set departure time
    maps_settings_ok = (has_any_ci(answer, ['8 am', '8am', '8:00', 'monday']) or
                        has_any_ci(answer, ['depart', 'departure time']))
    evaluator.add_custom_node(
        result=bool(maps_settings_ok),
        id="maps_action_settings",
        desc="[Action Node] maps.google.com:F2:A7 - Select driving mode and set departure time to 8 AM Monday",
        parent=maps_node,
        critical=False
    )

    # Extract travel time
    travel_time_ok = looks_like_time_duration(maps_info.travel_time)
    evaluator.add_custom_node(
        result=bool(travel_time_ok),
        id="maps_travel_time",
        desc="Extract the driving time from the route results",
        parent=maps_node,
        critical=False
    )

    # Extract distance
    distance_ok = looks_like_distance(maps_info.distance_miles)
    evaluator.add_custom_node(
        result=bool(distance_ok),
        id="maps_distance",
        desc="Extract the one-way distance in miles from the route results",
        parent=maps_node,
        critical=False
    )

    # 3.3 Car and Driver part
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver highway MPG for 2022 Toyota RAV4",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search for vehicle
    caranddriver_search_ok = (has_any_ci(answer, ['car and driver']) and
                              has_any_ci(answer, ['2022 toyota rav4', 'rav4']))
    evaluator.add_custom_node(
        result=bool(caranddriver_search_ok),
        id="caranddriver_action_search",
        desc="[Action Node] caranddriver.com:F2:A1 - Search for the 2022 Toyota RAV4 on Car and Driver",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F2:P3 - Extract highway MPG from review
    mpg_ok = looks_like_mpg(mpg_info.highway_mpg_text)
    mpg_highway_keyword = has_any_ci(answer, ['highway', 'hwy'])
    evaluator.add_custom_node(
        result=bool(mpg_ok and mpg_highway_keyword),
        id="caranddriver_perception_highway_mpg",
        desc="[Perception Node] caranddriver.com:F2:P3 - Extract the actual highway MPG from the Car and Driver review",
        parent=caranddriver_node,
        critical=False
    )

    # 3.4 Cost calculation part
    cost_node = evaluator.add_sequential(
        id="cost_calculation_section",
        desc="Monthly commuting fuel cost calculation",
        parent=root,
        critical=False
    )

    # Check calculation parameters are mentioned
    calc_params_ok = (has_any_ci(answer, ['22 working days', '22 days']) and
                      has_any_ci(answer, ['round trip', 'two trips']) and
                      has_any_ci(answer, ['$3', '3 per gallon', '3 dollar']))
    evaluator.add_custom_node(
        result=bool(calc_params_ok),
        id="cost_calculation_parameters",
        desc="Mentions the calculation parameters (22 working days, round trip, $3/gallon)",
        parent=cost_node,
        critical=False
    )

    # Check final monthly cost is provided
    monthly_cost_ok = looks_like_price(cost_info.monthly_fuel_cost)
    evaluator.add_custom_node(
        result=bool(monthly_cost_ok),
        id="cost_final_monthly_amount",
        desc="Provides the calculated monthly fuel commuting cost",
        parent=cost_node,
        critical=False
    )

    # Verify calculation logic is reasonable (if all data present)
    if (extract_float(maps_info.distance_miles) and
        extract_float(mpg_info.highway_mpg_text) and
        extract_float(cost_info.monthly_fuel_cost)):

        distance = extract_float(maps_info.distance_miles)
        mpg = extract_float(mpg_info.highway_mpg_text)
        reported_cost = extract_float(cost_info.monthly_fuel_cost)

        # Expected: (distance * 2 * 22) / mpg * 3
        expected_cost = (distance * 2 * 22) / mpg * 3
        tolerance = 0.15  # 15% tolerance
        calc_reasonable = abs(reported_cost - expected_cost) / expected_cost <= tolerance

        evaluator.add_custom_node(
            result=bool(calc_reasonable),
            id="cost_calculation_accuracy",
            desc="Verifies the monthly cost calculation is mathematically reasonable",
            parent=cost_node,
            critical=False
        )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
