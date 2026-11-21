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
TASK_ID = "task-8b9af7"
TASK_DESCRIPTION = 'I just moved to Texas and live in San Antonio, TX, but my workplace is in Austin, TX, so I need to commute daily. Since I don’t want to stop for gas too often, I want to buy a used compact SUV with strong real-world range.\n\nPlease first go to Edmunds and find 3 listings that meet all of the following criteria: model year 2022 or newer, under 50,000 miles, priced below $28,000, and currently for sale. For these 3 models, go to Car and Driver and find their corresponding professional reviews (if the exact model year is unavailable, use a nearby year from the same generation).\n\nPrioritize extracting the “75-mph Highway Driving” fuel-economy result and fuel-tank capacity from the “C/D TEST RESULTS” table (do not use EPA official figures). If a model truly does not have corresponding tested data, replace it with another vehicle that meets the Edmunds criteria, and continue until you have 3 vehicles with available tested data. If repeated attempts still produce fewer than 3, clearly note the shortfall and perform calculations only for the vehicles with available tested data.\n\nFinally, use Google Maps to plan a driving route from San Antonio to Austin. Based on the round-trip distance, calculate whether each of the 3 vehicles can support my 5-day workweek commute on a full tank under highway-only real-world conditions, without refueling midweek (i.e., fuel tank capacity × tested highway MPG > daily round-trip distance × 5).\n\nOutput: decision recommendation, Edmunds link for each vehicle, Car and Driver review link, tested highway MPG, fuel-tank capacity, theoretical highway range, and whether it meets my one-week commuting requirement.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VehicleListing(BaseModel):
    """Single vehicle listing extracted from answer"""
    vehicle_name: Optional[str] = None
    model_year: Optional[int] = None
    mileage: Optional[int] = None
    price: Optional[float] = None
    edmunds_link: Optional[str] = None
    caranddriver_link: Optional[str] = None
    tested_highway_mpg: Optional[float] = None
    fuel_tank_capacity: Optional[float] = None
    theoretical_range: Optional[float] = None
    meets_requirement: Optional[bool] = None


class AllVehicles(BaseModel):
    """All vehicles extracted from answer"""
    vehicles: List[VehicleListing] = Field(default_factory=list)
    route_distance: Optional[float] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_vehicles_from_answer() -> str:
    return """
Extract all vehicle listings from the answer. For each vehicle, extract:
- vehicle_name: the make and model name
- model_year: the year as an integer
- mileage: the odometer reading in miles as an integer
- price: the listing price in dollars as a float
- edmunds_link: the Edmunds listing URL
- caranddriver_link: the Car and Driver review URL
- tested_highway_mpg: the 75-mph highway driving fuel economy from C/D test results
- fuel_tank_capacity: the fuel tank capacity in gallons
- theoretical_range: the calculated highway range (tank capacity × highway MPG)
- meets_requirement: whether it meets the 5-day commute requirement (true/false)

Also extract:
- route_distance: the one-way distance from San Antonio to Austin in miles

If any field is missing, set it to null. Return all vehicles found in the vehicles list.
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


def looks_like_compact_suv(vehicle_name: Optional[str]) -> bool:
    """Check if vehicle name suggests it's a compact SUV"""
    if not vehicle_name:
        return False
    # Common compact SUV models
    compact_suvs = ['cx-5', 'cr-v', 'rav4', 'forester', 'tucson', 'sportage',
                    'escape', 'equinox', 'compass', 'cherokee', 'rogue', 'seltos',
                    'kicks', 'venue', 'trailblazer', 'ecosport', 'bronco sport']
    return has_any_ci(vehicle_name, compact_suvs) or ci_contains(vehicle_name, 'suv')


def is_valid_year(year: Optional[int]) -> bool:
    """Check if year is 2022 or newer"""
    if year is None:
        return False
    return year >= 2022


def is_valid_mileage(mileage: Optional[int]) -> bool:
    """Check if mileage is under 50,000"""
    if mileage is None:
        return False
    return mileage < 50000


def is_valid_price(price: Optional[float]) -> bool:
    """Check if price is below $28,000"""
    if price is None:
        return False
    return price < 28000


def is_valid_url(url: Optional[str], domain: str) -> bool:
    """Check if URL contains expected domain"""
    if not url:
        return False
    return domain.lower() in url.lower()


def is_reasonable_distance(distance: Optional[float]) -> bool:
    """Check if San Antonio to Austin distance is reasonable (75-85 miles)"""
    if distance is None:
        return False
    return 70 <= distance <= 90


def is_reasonable_mpg(mpg: Optional[float]) -> bool:
    """Check if highway MPG is reasonable for compact SUV (20-40 MPG)"""
    if mpg is None:
        return False
    return 20 <= mpg <= 45


def is_reasonable_tank_capacity(capacity: Optional[float]) -> bool:
    """Check if fuel tank capacity is reasonable (12-18 gallons)"""
    if capacity is None:
        return False
    return 10 <= capacity <= 20


def verify_range_calculation(tank: Optional[float], mpg: Optional[float],
                            calculated_range: Optional[float]) -> bool:
    """Verify that theoretical range = tank × mpg"""
    if tank is None or mpg is None or calculated_range is None:
        return False
    expected = tank * mpg
    # Allow 5% tolerance for rounding
    return abs(calculated_range - expected) / expected < 0.05


def verify_requirement_logic(tank: Optional[float], mpg: Optional[float],
                            distance: Optional[float], meets: Optional[bool]) -> bool:
    """Verify that meets_requirement logic is correct: tank × mpg > distance × 2 × 5"""
    if tank is None or mpg is None or distance is None or meets is None:
        return False
    weekly_distance = distance * 2 * 5  # round trip × 5 days
    can_support = (tank * mpg) > weekly_distance
    return can_support == meets


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
    all_vehicles = await evaluator.extract(
        prompt=prompt_extract_vehicles_from_answer(),
        template_class=AllVehicles,
        extraction_name="all_vehicles"
    )

    vehicles = all_vehicles.vehicles if all_vehicles else []
    route_distance = all_vehicles.route_distance if all_vehicles else None

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Edmunds section
    edmunds_node = evaluator.add_sequential(
        id="edmunds_section",
        desc="Edmunds vehicle listings meeting criteria",
        parent=root,
        critical=False
    )

    # [Action Node] edmunds.com:F1:A1 - Search for compact SUVs
    edmunds_search_ok = has_any_ci(answer, ['edmunds']) and has_any_ci(answer, ['compact suv', 'suv'])
    evaluator.add_custom_node(
        result=bool(edmunds_search_ok),
        id="edmunds_search_compact_suv",
        desc="[Action Node] edmunds.com:F1:A1 - Search for compact SUV listings on Edmunds",
        parent=edmunds_node,
        critical=False
    )

    # [Action Node] edmunds.com:F1:A14 - Navigate to Used Cars section
    used_cars_ok = has_any_ci(answer, ['used', 'for sale'])
    evaluator.add_custom_node(
        result=bool(used_cars_ok),
        id="edmunds_used_cars_section",
        desc="[Action Node] edmunds.com:F1:A14 - Navigate to Used Cars section on Edmunds",
        parent=edmunds_node,
        critical=False
    )

    # [Perception Node] edmunds.com:F1:P1 - Extract listing information
    has_three_vehicles = len(vehicles) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_vehicles),
        id="edmunds_three_listings",
        desc="[Perception Node] edmunds.com:F1:P1 - Extract 3 vehicle listings from search results",
        parent=edmunds_node,
        critical=False
    )

    # Verify vehicle criteria
    criteria_node = evaluator.add_parallel(
        id="vehicle_criteria_checks",
        desc="Verify all vehicles meet Edmunds criteria",
        parent=edmunds_node,
        critical=False
    )

    vehicles_meet_year = all(is_valid_year(v.model_year) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(vehicles_meet_year),
        id="criteria_year_2022_or_newer",
        desc="All vehicles are model year 2022 or newer",
        parent=criteria_node,
        critical=False
    )

    vehicles_meet_mileage = all(is_valid_mileage(v.mileage) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(vehicles_meet_mileage),
        id="criteria_under_50k_miles",
        desc="All vehicles have under 50,000 miles",
        parent=criteria_node,
        critical=False
    )

    vehicles_meet_price = all(is_valid_price(v.price) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(vehicles_meet_price),
        id="criteria_under_28k_price",
        desc="All vehicles are priced below $28,000",
        parent=criteria_node,
        critical=False
    )

    vehicles_are_compact_suv = all(looks_like_compact_suv(v.vehicle_name) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(vehicles_are_compact_suv),
        id="criteria_compact_suv_type",
        desc="All vehicles are compact SUVs",
        parent=criteria_node,
        critical=False
    )

    # Check Edmunds links
    edmunds_links_valid = all(is_valid_url(v.edmunds_link, 'edmunds') for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(edmunds_links_valid),
        id="edmunds_links_present",
        desc="All vehicles have valid Edmunds listing links",
        parent=edmunds_node,
        critical=False
    )

    # 3.2 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver professional reviews and test data",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search for vehicle reviews
    caranddriver_search_ok = has_any_ci(answer, ['car and driver'])
    evaluator.add_custom_node(
        result=bool(caranddriver_search_ok),
        id="caranddriver_search_reviews",
        desc="[Action Node] caranddriver.com:F2:A1 - Search for vehicle reviews on Car and Driver",
        parent=caranddriver_node,
        critical=False
    )

    # [Action Node] caranddriver.com:F1:A11 - Click into review details
    caranddriver_links_valid = all(is_valid_url(v.caranddriver_link, 'caranddriver') for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(caranddriver_links_valid),
        id="caranddriver_click_reviews",
        desc="[Action Node] caranddriver.com:F1:A11 - Click into review articles for each vehicle",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P12 - Identify Instrumented Test badge
    instrumented_test_ok = has_any_ci(answer, ['instrumented test', 'road test', 'c/d test', 'test results'])
    evaluator.add_custom_node(
        result=bool(instrumented_test_ok),
        id="caranddriver_identify_test_badge",
        desc="[Perception Node] caranddriver.com:F1:P12 - Identify Instrumented Test or Road Test articles with detailed data",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Extract 75-mph highway MPG from test results table
    all_have_mpg = all(v.tested_highway_mpg is not None for v in vehicles) if vehicles else False
    mpg_values_reasonable = all(is_reasonable_mpg(v.tested_highway_mpg) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(all_have_mpg and mpg_values_reasonable),
        id="caranddriver_extract_75mph_mpg",
        desc="[Perception Node] caranddriver.com:F1:P2 - Extract 75-mph Highway Driving fuel economy from C/D TEST RESULTS table",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F2:P3 - Extract fuel tank capacity
    all_have_tank = all(v.fuel_tank_capacity is not None for v in vehicles) if vehicles else False
    tank_values_reasonable = all(is_reasonable_tank_capacity(v.fuel_tank_capacity) for v in vehicles) if vehicles else False
    evaluator.add_custom_node(
        result=bool(all_have_tank and tank_values_reasonable),
        id="caranddriver_extract_tank_capacity",
        desc="[Perception Node] caranddriver.com:F2:P3 - Extract fuel tank capacity from test results",
        parent=caranddriver_node,
        critical=False
    )

    # Verify not using EPA figures
    not_using_epa = has_any_ci(answer, ['75-mph', '75 mph', 'c/d test', 'tested', 'instrumented']) and not has_any_ci(answer, ['epa official', 'epa rating', 'epa estimate'])
    evaluator.add_custom_node(
        result=bool(not_using_epa),
        id="caranddriver_not_epa_figures",
        desc="Using C/D tested data, not EPA official figures",
        parent=caranddriver_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Plan route with multiple points
    maps_route_ok = has_any_ci(answer, ['google maps', 'maps']) and has_any_ci(answer, ['san antonio', 'austin'])
    evaluator.add_custom_node(
        result=bool(maps_route_ok),
        id="maps_plan_route",
        desc="[Action Node] maps.google.com:F2:A5 - Plan driving route from San Antonio to Austin",
        parent=maps_node,
        critical=False
    )

    # Verify distance is reasonable
    distance_reasonable = is_reasonable_distance(route_distance)
    evaluator.add_custom_node(
        result=bool(distance_reasonable),
        id="maps_distance_reasonable",
        desc="Route distance is reasonable for San Antonio to Austin (75-85 miles)",
        parent=maps_node,
        critical=False
    )

    # 3.4 Calculation and logic section
    calculation_node = evaluator.add_parallel(
        id="calculation_section",
        desc="Range calculations and requirement verification",
        parent=root,
        critical=False
    )

    # Verify theoretical range calculations
    all_have_range = all(v.theoretical_range is not None for v in vehicles) if vehicles else False
    range_calculations_correct = all(
        verify_range_calculation(v.fuel_tank_capacity, v.tested_highway_mpg, v.theoretical_range)
        for v in vehicles
    ) if vehicles else False
    evaluator.add_custom_node(
        result=bool(all_have_range and range_calculations_correct),
        id="calculate_theoretical_range",
        desc="Calculate theoretical highway range (tank capacity × tested MPG) for each vehicle",
        parent=calculation_node,
        critical=False
    )

    # Verify requirement logic
    all_have_requirement_check = all(v.meets_requirement is not None for v in vehicles) if vehicles else False
    requirement_logic_correct = all(
        verify_requirement_logic(v.fuel_tank_capacity, v.tested_highway_mpg, route_distance, v.meets_requirement)
        for v in vehicles
    ) if vehicles and route_distance else False
    evaluator.add_custom_node(
        result=bool(all_have_requirement_check and requirement_logic_correct),
        id="verify_5day_requirement",
        desc="Verify whether each vehicle can support 5-day workweek commute without refueling (tank × MPG > distance × 2 × 5)",
        parent=calculation_node,
        critical=False
    )

    # Check for decision recommendation
    has_recommendation = has_any_ci(answer, ['recommend', 'suggestion', 'best choice', 'decision'])
    evaluator.add_custom_node(
        result=bool(has_recommendation),
        id="decision_recommendation_present",
        desc="Provides decision recommendation based on analysis",
        parent=calculation_node,
        critical=False
    )

    # 3.5 Output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required output elements are present",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(edmunds_links_valid),
        id="output_edmunds_links",
        desc="Output includes Edmunds links for each vehicle",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(caranddriver_links_valid),
        id="output_caranddriver_links",
        desc="Output includes Car and Driver review links",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(all_have_mpg),
        id="output_tested_mpg",
        desc="Output includes tested highway MPG for each vehicle",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(all_have_tank),
        id="output_tank_capacity",
        desc="Output includes fuel tank capacity for each vehicle",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(all_have_range),
        id="output_theoretical_range",
        desc="Output includes theoretical highway range for each vehicle",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(all_have_requirement_check),
        id="output_requirement_check",
        desc="Output includes whether each vehicle meets 5-day commute requirement",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
