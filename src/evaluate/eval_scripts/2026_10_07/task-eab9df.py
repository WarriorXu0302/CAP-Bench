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
TASK_ID = "task-eab9df"
TASK_DESCRIPTION = 'I’m planning a *La La Land*-themed anniversary activity in Los Angeles. First, go to IMDb and check the film details. In the **Filming Locations** or **Trivia** section, confirm the real names of these three key locations: the jazz club where Seb plays piano (**The Lighthouse Cafe**), the pier where the leads walk (**Hermosa Beach Pier**), and **Griffith Observatory**.  \n\nNext, go to Google Maps and check whether all three places are open this Saturday; if the current page does not yet show business hours for this Saturday, check the status for the nearest visible Saturday instead. Then plan a driving route in this order: **club -> pier -> observatory**, and confirm whether the total driving distance is within **40 miles**.  \n\nFinally, using the observatory as the center point, find a restaurant on TripAdvisor within **2 miles**, with a **“Romantic”** style. Prefer one with a rating of **4.5+** and with reviews that explicitly mention **“View.”** If too few options meet all criteria, keep the **2-mile** and **“Romantic”** requirements unchanged, but relax the rating threshold to **4.0** and note the reason for this relaxation.  \n\nOutput the names of the three filming locations and their Saturday open status, the total mileage calculated by Google Maps, and the final selected restaurant’s name, rating, and TripAdvisor link, and indicate whether its reviews include the keyword **“View.”**'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FilmingLocations(BaseModel):
    """Three filming locations extracted from the answer"""
    lighthouse_cafe: Optional[str] = None
    hermosa_beach_pier: Optional[str] = None
    griffith_observatory: Optional[str] = None


class SaturdayOpenStatus(BaseModel):
    """Saturday open status for the three locations"""
    lighthouse_cafe_saturday: Optional[str] = None
    hermosa_beach_pier_saturday: Optional[str] = None
    griffith_observatory_saturday: Optional[str] = None


class RouteInfo(BaseModel):
    """Route information from Google Maps"""
    total_distance_text: Optional[str] = None
    route_order_mentioned: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Restaurant information from TripAdvisor"""
    restaurant_name: Optional[str] = None
    rating_text: Optional[str] = None
    tripadvisor_link: Optional[str] = None
    has_view_keyword: Optional[str] = None
    is_romantic: Optional[str] = None
    distance_from_observatory: Optional[str] = None
    rating_relaxation_note: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_filming_locations() -> str:
    return """
Extract the three filming locations from the answer as confirmed from IMDb:
- lighthouse_cafe: the name of the jazz club where Seb plays piano
- hermosa_beach_pier: the name of the pier where the leads walk
- griffith_observatory: the name of the observatory

Return the exact names as stated in the answer. If any is missing, set it to null.
"""


def prompt_extract_saturday_status() -> str:
    return """
Extract the Saturday open status for each of the three locations:
- lighthouse_cafe_saturday: open status on Saturday
- hermosa_beach_pier_saturday: open status on Saturday
- griffith_observatory_saturday: open status on Saturday

Return the status text exactly as stated. If any is missing, set it to null.
"""


def prompt_extract_route_info() -> str:
    return """
Extract the Google Maps route information:
- total_distance_text: the total driving distance for the route (include units)
- route_order_mentioned: any mention of the route order (club -> pier -> observatory)

If any field is missing, set it to null.
"""


def prompt_extract_restaurant_info() -> str:
    return """
Extract the TripAdvisor restaurant information:
- restaurant_name: the name of the selected restaurant
- rating_text: the rating of the restaurant
- tripadvisor_link: the TripAdvisor link to the restaurant
- has_view_keyword: whether the answer mentions "View" in reviews (yes/no or similar)
- is_romantic: whether the answer mentions "Romantic" style
- distance_from_observatory: the distance from Griffith Observatory
- rating_relaxation_note: any note about relaxing the rating threshold from 4.5 to 4.0

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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def looks_like_location_name(text: Optional[str], expected_keywords: List[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, expected_keywords)


def looks_like_open_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['open', 'closed', 'hours', 'yes', 'no'])


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km'])


def extract_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m.group(1))
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
    filming_locations = await evaluator.extract(
        prompt=prompt_extract_filming_locations(),
        template_class=FilmingLocations,
        extraction_name="filming_locations"
    )

    saturday_status = await evaluator.extract(
        prompt=prompt_extract_saturday_status(),
        template_class=SaturdayOpenStatus,
        extraction_name="saturday_status"
    )

    route_info = await evaluator.extract(
        prompt=prompt_extract_route_info(),
        template_class=RouteInfo,
        extraction_name="route_info"
    )

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant_info(),
        template_class=RestaurantInfo,
        extraction_name="restaurant_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IMDb section
    imdb_node = evaluator.add_sequential(
        id="imdb_section",
        desc="IMDb filming locations verification for La La Land",
        parent=root,
        critical=False
    )

    # [Action Node] imdb.com:F3:A34 - Expand collapsed panels (Filming Locations/Trivia)
    imdb_action_ok = (has_any_ci(answer, ['imdb']) and
                      (has_any_ci(answer, ['filming locations']) or has_any_ci(answer, ['trivia'])))
    evaluator.add_custom_node(
        result=bool(imdb_action_ok),
        id="imdb_action_expand_panels",
        desc="[Action Node] imdb.com:F3:A34 - Navigate to IMDb and access Filming Locations or Trivia section (typically collapsed panels)",
        parent=imdb_node,
        critical=False
    )

    # Check individual location names
    lighthouse_ok = looks_like_location_name(filming_locations.lighthouse_cafe, ['lighthouse', 'cafe'])
    hermosa_ok = looks_like_location_name(filming_locations.hermosa_beach_pier, ['hermosa', 'beach', 'pier'])
    griffith_ok = looks_like_location_name(filming_locations.griffith_observatory, ['griffith', 'observatory'])

    evaluator.add_custom_node(
        result=bool(lighthouse_ok and hermosa_ok and griffith_ok),
        id="imdb_location_names_verified",
        desc="All three filming locations (The Lighthouse Cafe, Hermosa Beach Pier, Griffith Observatory) correctly identified",
        parent=imdb_node,
        critical=False
    )

    # 3.2 Google Maps section
    gmaps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps business hours and route planning",
        parent=root,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P15 - Saturday business hours status
    lighthouse_status_ok = looks_like_open_status(saturday_status.lighthouse_cafe_saturday)
    hermosa_status_ok = looks_like_open_status(saturday_status.hermosa_beach_pier_saturday)
    griffith_status_ok = looks_like_open_status(saturday_status.griffith_observatory_saturday)

    evaluator.add_custom_node(
        result=bool(lighthouse_status_ok and hermosa_status_ok and griffith_status_ok),
        id="gmaps_perception_saturday_hours",
        desc="[Perception Node] maps.google.com:F5:P15 - Extract Saturday open status for all three locations",
        parent=gmaps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Input origin and destination
    # [Action Node] maps.google.com:F2:A18 - Add waypoints for multi-stop route
    route_order_ok = has_any_ci(answer, ['club', 'lighthouse']) and has_any_ci(answer, ['pier', 'hermosa']) and has_any_ci(answer, ['observatory', 'griffith'])
    evaluator.add_custom_node(
        result=bool(route_order_ok),
        id="gmaps_action_route_order",
        desc="[Action Node] maps.google.com:F2:A5 + maps.google.com:F2:A18 - Plan driving route in correct order: club -> pier -> observatory",
        parent=gmaps_node,
        critical=False
    )

    # Verify total distance
    distance_num = extract_float(route_info.total_distance_text)
    distance_ok = looks_like_distance(route_info.total_distance_text)
    within_40_miles = distance_num is not None and distance_num <= 40.0 if distance_num else False

    evaluator.add_custom_node(
        result=bool(distance_ok and distance_num is not None),
        id="gmaps_total_distance_extracted",
        desc="Total driving distance extracted from Google Maps route",
        parent=gmaps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(within_40_miles),
        id="gmaps_distance_within_40_miles",
        desc="Total driving distance is within 40 miles",
        parent=gmaps_node,
        critical=False
    )

    # 3.3 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor restaurant search with filters",
        parent=root,
        critical=False
    )

    # Check restaurant was found
    restaurant_name_ok = bool(restaurant_info.restaurant_name and restaurant_info.restaurant_name.strip())
    evaluator.add_custom_node(
        result=bool(restaurant_name_ok),
        id="tripadvisor_restaurant_found",
        desc="Restaurant identified on TripAdvisor",
        parent=tripadvisor_node,
        critical=False
    )

    # Check distance from observatory
    restaurant_distance_ok = has_any_ci(restaurant_info.distance_from_observatory, ['mile', 'mi']) or has_any_ci(answer, ['2 mile', 'within 2', '2-mile'])
    evaluator.add_custom_node(
        result=bool(restaurant_distance_ok),
        id="tripadvisor_distance_filter",
        desc="Restaurant is within 2 miles of Griffith Observatory",
        parent=tripadvisor_node,
        critical=False
    )

    # Check romantic style
    romantic_ok = has_any_ci(restaurant_info.is_romantic, ['romantic', 'yes']) or has_any_ci(answer, ['romantic'])
    evaluator.add_custom_node(
        result=bool(romantic_ok),
        id="tripadvisor_romantic_style",
        desc="Restaurant has 'Romantic' style designation",
        parent=tripadvisor_node,
        critical=False
    )

    # [Perception Node] tripadvisor.com:F3:P1 - Rating identification
    rating_value = extract_rating(restaurant_info.rating_text)
    rating_ok = rating_value is not None and rating_value >= 4.0
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="tripadvisor_perception_rating",
        desc="[Perception Node] tripadvisor.com:F3:P1 - Restaurant rating is 4.0+ (with 4.5+ preferred)",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F3:A9 - Review filtering/search for "View" keyword
    view_keyword_ok = has_any_ci(restaurant_info.has_view_keyword, ['view', 'yes']) or has_any_ci(answer, ['view'])
    evaluator.add_custom_node(
        result=bool(view_keyword_ok),
        id="tripadvisor_action_review_filter",
        desc="[Action Node] tripadvisor.com:F3:A9 - Reviews explicitly mention 'View' keyword",
        parent=tripadvisor_node,
        critical=False
    )

    # Check if rating relaxation was noted (if applicable)
    relaxation_noted = bool(restaurant_info.rating_relaxation_note and restaurant_info.rating_relaxation_note.strip())
    if rating_value and rating_value < 4.5:
        evaluator.add_custom_node(
            result=bool(relaxation_noted),
            id="tripadvisor_relaxation_documented",
            desc="Rating threshold relaxation (from 4.5 to 4.0) is documented with reason",
            parent=tripadvisor_node,
            critical=False
        )

    # Check TripAdvisor link provided
    link_ok = bool(restaurant_info.tripadvisor_link and restaurant_info.tripadvisor_link.strip())
    evaluator.add_custom_node(
        result=bool(link_ok),
        id="tripadvisor_link_provided",
        desc="TripAdvisor link to the restaurant is provided",
        parent=tripadvisor_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
