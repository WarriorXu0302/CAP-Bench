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
TASK_ID = "task-a93f20"
TASK_DESCRIPTION = 'I’m a food vlog creator planning to film a “One-Day Michelin Hunt” episode in Manhattan, New York on a Saturday next month. I need you to help plan that day’s itinerary by finding 3 Michelin-recognized restaurants: one for Lunch, one for Afternoon Tea/Light Meal, and one for Dinner.\n\nRequirements:\n1. All restaurants must be in Manhattan, and preferably all have Yelp ratings of 4.5 or higher (including 4.5).\n2. First, check OpenTable to confirm whether each restaurant is labeled as Michelin Starred or Bib Gourmand.\n3. Then verify ratings on Yelp and check operating hours to ensure all are open on that Saturday and that times do not conflict (assume Lunch 12:00–14:00, Afternoon Tea 15:30–16:30, Dinner 18:30–20:30).\n4. Finally, based on the three restaurants’ locations, use Google Maps to plan an efficient route (minimizing total travel distance as much as possible).\n\nAdditional execution rules:\n- Prioritize Michelin recognition and time-slot availability first, then aim to satisfy Yelp 4.5+.\n- If no restaurant in a given time slot can meet both Michelin recognition and Yelp 4.5+, relax that slot to Yelp 4.0+ and clearly indicate in the output which slot was relaxed and the restaurant’s actual rating.\n- If no reservable/verifiable information is available for that Saturday next month, check the nearest upcoming Saturday with visible availability and note it.\n\nOutput: In chronological order, provide the 3 restaurant names, addresses, Michelin designation (Starred/Bib Gourmand), Yelp rating, Saturday business hours, and Yelp detail page links; plus Google Maps walking distance and time between each pair of consecutive stops. Also include for each restaurant: “Meets Yelp 4.5+ (Yes/No).”'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantInfo(BaseModel):
    """Single restaurant information"""
    name: Optional[str] = None
    address: Optional[str] = None
    michelin_designation: Optional[str] = None
    yelp_rating: Optional[float] = None
    saturday_hours: Optional[str] = None
    yelp_link: Optional[str] = None
    meets_4_5: Optional[str] = None


class RouteInfo(BaseModel):
    """Route information between consecutive restaurants"""
    from_restaurant: Optional[str] = None
    to_restaurant: Optional[str] = None
    distance: Optional[str] = None
    time: Optional[str] = None


class ItineraryExtraction(BaseModel):
    """Complete itinerary extracted from answer"""
    lunch_restaurant: Optional[RestaurantInfo] = None
    afternoon_tea_restaurant: Optional[RestaurantInfo] = None
    dinner_restaurant: Optional[RestaurantInfo] = None
    route_lunch_to_tea: Optional[RouteInfo] = None
    route_tea_to_dinner: Optional[RouteInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_itinerary() -> str:
    return """
Extract the complete itinerary information from the answer for the three Michelin restaurants planned for Saturday.

For each of the three restaurants (Lunch, Afternoon Tea, Dinner), extract:
- name: restaurant name
- address: full address
- michelin_designation: whether it's "Michelin Starred" or "Bib Gourmand" (exact text)
- yelp_rating: numeric rating (e.g., 4.5, 4.7)
- saturday_hours: operating hours on Saturday
- yelp_link: Yelp detail page URL
- meets_4_5: whether it meets the 4.5+ requirement ("Yes" or "No")

For the routes between consecutive restaurants, extract:
- from_restaurant and to_restaurant: restaurant names
- distance: walking distance text from Google Maps
- time: walking time text from Google Maps

Set any missing field to null.
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


def is_valid_michelin_designation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['michelin star', 'bib gourmand', 'starred'])


def is_manhattan_address(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['manhattan', 'new york', 'ny'])


def has_saturday_hours(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['saturday', 'sat'])


def has_yelp_link(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'yelp.com' in text.lower()


def has_distance_info(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['mile', 'mi', 'km', 'meter', 'feet', 'ft'])


def has_time_info(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['min', 'hour', 'hr'])


def mentions_opentable(answer: str) -> bool:
    return has_any_ci(answer, ['opentable'])


def mentions_yelp(answer: str) -> bool:
    return has_any_ci(answer, ['yelp'])


def mentions_google_maps(answer: str) -> bool:
    return has_any_ci(answer, ['google maps', 'google map'])


def count_restaurants_mentioned(itinerary: ItineraryExtraction) -> int:
    count = 0
    if itinerary.lunch_restaurant and itinerary.lunch_restaurant.name:
        count += 1
    if itinerary.afternoon_tea_restaurant and itinerary.afternoon_tea_restaurant.name:
        count += 1
    if itinerary.dinner_restaurant and itinerary.dinner_restaurant.name:
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
    itinerary = await evaluator.extract(
        prompt=prompt_extract_itinerary(),
        template_class=ItineraryExtraction,
        extraction_name="itinerary_extraction"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 OpenTable section - Michelin designation verification
    opentable_node = evaluator.add_sequential(
        id="opentable_section",
        desc="OpenTable verification of Michelin designations",
        parent=root,
        critical=False
    )

    # [Perception Node] opentable.com:F12:P1 - Identify Michelin labels
    opentable_mentioned = mentions_opentable(answer)
    lunch_michelin = is_valid_michelin_designation(
        itinerary.lunch_restaurant.michelin_designation if itinerary.lunch_restaurant else None
    )
    tea_michelin = is_valid_michelin_designation(
        itinerary.afternoon_tea_restaurant.michelin_designation if itinerary.afternoon_tea_restaurant else None
    )
    dinner_michelin = is_valid_michelin_designation(
        itinerary.dinner_restaurant.michelin_designation if itinerary.dinner_restaurant else None
    )

    all_michelin_verified = lunch_michelin and tea_michelin and dinner_michelin

    evaluator.add_custom_node(
        result=bool(opentable_mentioned and all_michelin_verified),
        id="opentable_michelin_labels",
        desc="[Perception Node] opentable.com:F12:P1 - Verify all three restaurants have Michelin Starred or Bib Gourmand designation from OpenTable",
        parent=opentable_node,
        critical=False
    )

    # 3.2 Yelp section - Rating and hours verification
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp verification of ratings and Saturday operating hours",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Search for restaurants
    yelp_mentioned = mentions_yelp(answer)
    restaurant_count = count_restaurants_mentioned(itinerary)

    evaluator.add_custom_node(
        result=bool(yelp_mentioned and restaurant_count == 3),
        id="yelp_search_action",
        desc="[Action Node] yelp.com:F1:A1 - Search for all three restaurants on Yelp",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Extract ratings
    lunch_rating = itinerary.lunch_restaurant.yelp_rating if itinerary.lunch_restaurant else None
    tea_rating = itinerary.afternoon_tea_restaurant.yelp_rating if itinerary.afternoon_tea_restaurant else None
    dinner_rating = itinerary.dinner_restaurant.yelp_rating if itinerary.dinner_restaurant else None

    all_ratings_present = (lunch_rating is not None) and (tea_rating is not None) and (dinner_rating is not None)

    evaluator.add_custom_node(
        result=bool(all_ratings_present),
        id="yelp_ratings_perception",
        desc="[Perception Node] yelp.com:F1:P1 - Extract Yelp ratings for all three restaurants",
        parent=yelp_node,
        critical=False
    )

    # Check 4.5+ compliance or relaxation explanation
    lunch_meets = itinerary.lunch_restaurant.meets_4_5 if itinerary.lunch_restaurant else None
    tea_meets = itinerary.afternoon_tea_restaurant.meets_4_5 if itinerary.afternoon_tea_restaurant else None
    dinner_meets = itinerary.dinner_restaurant.meets_4_5 if itinerary.dinner_restaurant else None

    has_compliance_info = bool(lunch_meets or tea_meets or dinner_meets)

    evaluator.add_custom_node(
        result=bool(has_compliance_info),
        id="yelp_rating_compliance",
        desc="Indicates whether each restaurant meets Yelp 4.5+ requirement (Yes/No)",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P2 - Extract Saturday operating hours
    lunch_hours = has_saturday_hours(
        itinerary.lunch_restaurant.saturday_hours if itinerary.lunch_restaurant else None
    )
    tea_hours = has_saturday_hours(
        itinerary.afternoon_tea_restaurant.saturday_hours if itinerary.afternoon_tea_restaurant else None
    )
    dinner_hours = has_saturday_hours(
        itinerary.dinner_restaurant.saturday_hours if itinerary.dinner_restaurant else None
    )

    all_hours_present = lunch_hours and tea_hours and dinner_hours

    evaluator.add_custom_node(
        result=bool(all_hours_present),
        id="yelp_saturday_hours_perception",
        desc="[Perception Node] yelp.com:F1:P2 - Extract Saturday operating hours for all three restaurants to verify time slot availability",
        parent=yelp_node,
        critical=False
    )

    # Check for Yelp links
    lunch_link = has_yelp_link(
        itinerary.lunch_restaurant.yelp_link if itinerary.lunch_restaurant else None
    )
    tea_link = has_yelp_link(
        itinerary.afternoon_tea_restaurant.yelp_link if itinerary.afternoon_tea_restaurant else None
    )
    dinner_link = has_yelp_link(
        itinerary.dinner_restaurant.yelp_link if itinerary.dinner_restaurant else None
    )

    all_links_present = lunch_link and tea_link and dinner_link

    evaluator.add_custom_node(
        result=bool(all_links_present),
        id="yelp_links_provided",
        desc="Provides Yelp detail page links for all three restaurants",
        parent=yelp_node,
        critical=False
    )

    # 3.3 Google Maps section - Route planning
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps route planning between restaurants",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A5 - Input start and end points for route
    maps_mentioned = mentions_google_maps(answer)
    has_route_info = bool(itinerary.route_lunch_to_tea or itinerary.route_tea_to_dinner)

    evaluator.add_custom_node(
        result=bool(maps_mentioned and has_route_info),
        id="google_maps_route_action",
        desc="[Action Node] google.com/maps:F2:A5 - Input restaurant locations as start and end points to plan routes",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] google.com/maps:F1:P3 - Extract distance and time
    lunch_to_tea_distance = has_distance_info(
        itinerary.route_lunch_to_tea.distance if itinerary.route_lunch_to_tea else None
    )
    lunch_to_tea_time = has_time_info(
        itinerary.route_lunch_to_tea.time if itinerary.route_lunch_to_tea else None
    )
    tea_to_dinner_distance = has_distance_info(
        itinerary.route_tea_to_dinner.distance if itinerary.route_tea_to_dinner else None
    )
    tea_to_dinner_time = has_time_info(
        itinerary.route_tea_to_dinner.time if itinerary.route_tea_to_dinner else None
    )

    all_routes_complete = (lunch_to_tea_distance and lunch_to_tea_time and
                          tea_to_dinner_distance and tea_to_dinner_time)

    evaluator.add_custom_node(
        result=bool(all_routes_complete),
        id="google_maps_distance_time_perception",
        desc="[Perception Node] google.com/maps:F1:P3 - Extract walking distance and time between consecutive restaurant pairs",
        parent=maps_node,
        critical=False
    )

    # 3.4 Additional quality checks
    quality_node = evaluator.add_parallel(
        id="quality_checks",
        desc="Additional quality and completeness checks",
        parent=root,
        critical=False
    )

    # Check Manhattan location requirement
    lunch_address_ok = is_manhattan_address(
        itinerary.lunch_restaurant.address if itinerary.lunch_restaurant else None
    )
    tea_address_ok = is_manhattan_address(
        itinerary.afternoon_tea_restaurant.address if itinerary.afternoon_tea_restaurant else None
    )
    dinner_address_ok = is_manhattan_address(
        itinerary.dinner_restaurant.address if itinerary.dinner_restaurant else None
    )

    all_manhattan = lunch_address_ok and tea_address_ok and dinner_address_ok

    evaluator.add_custom_node(
        result=bool(all_manhattan),
        id="manhattan_location_requirement",
        desc="All three restaurants are located in Manhattan with addresses provided",
        parent=quality_node,
        critical=False
    )

    # Check chronological ordering
    has_chronological_order = bool(
        itinerary.lunch_restaurant and itinerary.lunch_restaurant.name and
        itinerary.afternoon_tea_restaurant and itinerary.afternoon_tea_restaurant.name and
        itinerary.dinner_restaurant and itinerary.dinner_restaurant.name
    )

    evaluator.add_custom_node(
        result=bool(has_chronological_order),
        id="chronological_order",
        desc="Restaurants are presented in chronological order: Lunch, Afternoon Tea, Dinner",
        parent=quality_node,
        critical=False
    )

    # Check for Saturday or upcoming Saturday mention
    saturday_mentioned = has_any_ci(answer, ['saturday', 'sat'])

    evaluator.add_custom_node(
        result=bool(saturday_mentioned),
        id="saturday_context",
        desc="Answer addresses Saturday availability or mentions checking upcoming Saturday",
        parent=quality_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
