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
TASK_ID = "task-cc33fd"
TASK_DESCRIPTION = 'I’m planning a family trip to Seattle, WA, but my child has severe pollen allergies, so we need to be extremely cautious. Please help me design a safe lodging plan.\n\nFirst, go to U.S. News Health and find all hospitals in the Seattle, WA area with strong ratings in the “Pediatrics” specialty. From those, select 2 hospitals that are either ranked in the top 50 nationally for Pediatrics (Ranked) or rated “High Performing.”\n\nThen, mark these 2 hospitals on Google Maps and measure each hospital’s driving distance to the Space Needle. Choose the hospital closer to the Space Needle as the safety anchor.\n\nUsing that anchor hospital as the center point, find a hotel on Google Maps that is within a 10-minute drive from the hospital and has a rating above 4.0 stars.\n\nFinally, go to TripAdvisor to confirm whether the hotel lists “Air conditioning” as an available amenity (important for pollen control), and identify 2 nearby “Indoor” attractions (Museums or Galleries) suitable for avoiding pollen exposure.\n\nIf a TripAdvisor login/sign-up wall blocks access to full amenities or attraction details, document that access limitation and continue completing the remaining steps using visible information (prioritizing TripAdvisor’s publicly visible hotel amenity summary and attraction category pages).\n\nOutput:\n- Names of the two selected pediatric hospitals and their Pediatrics rating status  \n- Distance from the anchor hospital to the Space Needle  \n- Selected hotel name, Google Maps rating, and driving time to the anchor hospital  \n- Whether TripAdvisor shows air conditioning at the hotel, plus names and categories of two recommended indoor attractions  \n- Detail-page links for all relevant locations'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HospitalInfo(BaseModel):
    """Two pediatric hospitals extracted from the answer"""
    hospital1_name: Optional[str] = None
    hospital1_status: Optional[str] = None
    hospital2_name: Optional[str] = None
    hospital2_status: Optional[str] = None


class AnchorHospitalInfo(BaseModel):
    """Anchor hospital and its distance to Space Needle"""
    anchor_hospital_name: Optional[str] = None
    distance_to_space_needle: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel extracted from the answer"""
    hotel_name: Optional[str] = None
    google_maps_rating: Optional[str] = None
    driving_time_to_hospital: Optional[str] = None


class TripAdvisorInfo(BaseModel):
    """TripAdvisor amenity and attractions extracted from the answer"""
    has_air_conditioning: Optional[str] = None
    attraction1_name: Optional[str] = None
    attraction1_category: Optional[str] = None
    attraction2_name: Optional[str] = None
    attraction2_category: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hospitals_from_answer() -> str:
    return """
Extract the two pediatric hospitals the user selected from U.S. News Health in Seattle, WA from the answer.

Return:
- hospital1_name: name of the first hospital exactly as stated
- hospital1_status: the pediatrics rating status (e.g., "Ranked #XX" or "High Performing")
- hospital2_name: name of the second hospital exactly as stated
- hospital2_status: the pediatrics rating status (e.g., "Ranked #XX" or "High Performing")

If any field is missing, set it to null.
"""


def prompt_extract_anchor_hospital_from_answer() -> str:
    return """
Extract the anchor hospital (the one closer to Space Needle) and its distance to Space Needle from the answer.

Return:
- anchor_hospital_name: name of the anchor hospital
- distance_to_space_needle: driving distance to Space Needle exactly as stated (include units)

If any field is missing, set it to null.
"""


def prompt_extract_hotel_from_answer() -> str:
    return """
Extract the hotel details from the answer that was found on Google Maps near the anchor hospital.

Return:
- hotel_name: name of the hotel
- google_maps_rating: the Google Maps rating (e.g., "4.5", "4.2 stars")
- driving_time_to_hospital: driving time from hotel to anchor hospital exactly as stated

If any field is missing, set it to null.
"""


def prompt_extract_tripadvisor_from_answer() -> str:
    return """
Extract the TripAdvisor amenity confirmation and indoor attractions from the answer.

Return:
- has_air_conditioning: whether air conditioning is available (e.g., "Yes", "No", "Listed", "Not visible")
- attraction1_name: name of the first indoor attraction
- attraction1_category: category of the first attraction (e.g., "Museum", "Gallery")
- attraction2_name: name of the second indoor attraction
- attraction2_category: category of the second attraction (e.g., "Museum", "Gallery")

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


def looks_like_pediatrics_status(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['ranked', 'high performing', '#'])


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km', 'minute', 'min'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (has_any_ci(text, ['star', 'rating']) or re.search(r'\d\.\d', text))


def looks_like_driving_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['minute', 'min', 'drive'])


def looks_like_indoor_category(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['museum', 'gallery', 'indoor'])


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
    hospitals_info = await evaluator.extract(
        prompt=prompt_extract_hospitals_from_answer(),
        template_class=HospitalInfo,
        extraction_name="hospitals_info"
    )

    anchor_info = await evaluator.extract(
        prompt=prompt_extract_anchor_hospital_from_answer(),
        template_class=AnchorHospitalInfo,
        extraction_name="anchor_hospital_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    tripadvisor_info = await evaluator.extract(
        prompt=prompt_extract_tripadvisor_from_answer(),
        template_class=TripAdvisorInfo,
        extraction_name="tripadvisor_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 U.S. News Health section
    usnews_node = evaluator.add_sequential(
        id="usnews_section",
        desc="U.S. News Health - Find pediatric hospitals in Seattle, WA",
        parent=root,
        critical=False
    )

    # [Action Node] health.usnews.com:F1:A1 - Advanced search input
    usnews_search_ok = has_any_ci(answer, ['u.s. news', 'us news', 'usnews']) and has_any_ci(answer, ['seattle', 'pediatric'])
    evaluator.add_custom_node(
        result=bool(usnews_search_ok),
        id="usnews_action_search",
        desc="[Action Node] health.usnews.com:F1:A1 - Search for Seattle area hospitals with Pediatrics specialty",
        parent=usnews_node,
        critical=False
    )

    # [Action Node] health.usnews.com:F1:A2 - Multi-condition filtering
    hospital1_ok = bool(hospitals_info and hospitals_info.hospital1_name and hospitals_info.hospital1_name.strip())
    hospital2_ok = bool(hospitals_info and hospitals_info.hospital2_name and hospitals_info.hospital2_name.strip())
    filtering_ok = hospital1_ok and hospital2_ok and has_any_ci(answer, ['seattle'])
    evaluator.add_custom_node(
        result=bool(filtering_ok),
        id="usnews_action_filtering",
        desc="[Action Node] health.usnews.com:F1:A2 - Filter and select 2 hospitals meeting Pediatrics criteria (Ranked top 50 or High Performing)",
        parent=usnews_node,
        critical=False
    )

    # [Perception Node] health.usnews.com:F1:P1 - Status/badge identification
    status1_ok = looks_like_pediatrics_status(hospitals_info.hospital1_status)
    status2_ok = looks_like_pediatrics_status(hospitals_info.hospital2_status)
    evaluator.add_custom_node(
        result=bool(status1_ok and status2_ok),
        id="usnews_perception_status",
        desc="[Perception Node] health.usnews.com:F1:P1 - Identify Pediatrics rating status (Ranked #XX or High Performing) for both hospitals",
        parent=usnews_node,
        critical=False
    )

    # 3.2 Google Maps distance measurement section
    maps_distance_node = evaluator.add_sequential(
        id="maps_distance_section",
        desc="Google Maps - Measure driving distances from hospitals to Space Needle",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection
    distance_action_ok = has_any_ci(answer, ['google maps', 'space needle']) and has_any_ci(answer, ['distance', 'driving'])
    evaluator.add_custom_node(
        result=bool(distance_action_ok),
        id="maps_action_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 + maps.google.com:F2:A7 - Measure driving distance from each hospital to Space Needle",
        parent=maps_distance_node,
        critical=False
    )

    # Check if anchor hospital and distance are provided
    anchor_distance_ok = bool(anchor_info and anchor_info.anchor_hospital_name and looks_like_distance(anchor_info.distance_to_space_needle))
    evaluator.add_custom_node(
        result=bool(anchor_distance_ok),
        id="maps_anchor_selection",
        desc="Select the closer hospital as anchor and report its distance to Space Needle",
        parent=maps_distance_node,
        critical=False
    )

    # 3.3 Google Maps hotel search section
    maps_hotel_node = evaluator.add_sequential(
        id="maps_hotel_section",
        desc="Google Maps - Find hotel near anchor hospital within 10-minute drive",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F11:A20 - Nearby/proximity search
    hotel_search_ok = has_any_ci(answer, ['google maps', 'hotel']) and bool(hotel_info and hotel_info.hotel_name)
    evaluator.add_custom_node(
        result=bool(hotel_search_ok),
        id="maps_action_hotel_search",
        desc="[Action Node] maps.google.com:F11:A20 - Search for hotel within 10-minute drive from anchor hospital",
        parent=maps_hotel_node,
        critical=False
    )

    # Verify hotel rating and driving time
    rating_ok = looks_like_rating(hotel_info.google_maps_rating) if hotel_info else False
    rating_value = extract_float(hotel_info.google_maps_rating) if hotel_info else None
    rating_above_4 = rating_value and rating_value > 4.0

    driving_time_ok = looks_like_driving_time(hotel_info.driving_time_to_hospital) if hotel_info else False

    evaluator.add_custom_node(
        result=bool(rating_ok and rating_above_4),
        id="maps_hotel_rating",
        desc="Hotel has Google Maps rating above 4.0 stars",
        parent=maps_hotel_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(driving_time_ok),
        id="maps_hotel_driving_time",
        desc="Driving time from hotel to anchor hospital is reported (should be within 10 minutes)",
        parent=maps_hotel_node,
        critical=False
    )

    # 3.4 TripAdvisor section
    tripadvisor_node = evaluator.add_sequential(
        id="tripadvisor_section",
        desc="TripAdvisor - Verify hotel amenities and find indoor attractions",
        parent=root,
        critical=False
    )

    # [Action Node] tripadvisor.com:F1:A4 - Amenity filter/view
    # [Perception Node] tripadvisor.com:F1:P2 - Badge/label identification
    ac_info_ok = bool(tripadvisor_info and tripadvisor_info.has_air_conditioning and tripadvisor_info.has_air_conditioning.strip())
    ac_check_ok = has_any_ci(answer, ['tripadvisor', 'air conditioning', 'amenity', 'amenities'])
    evaluator.add_custom_node(
        result=bool(ac_info_ok and ac_check_ok),
        id="tripadvisor_action_amenity",
        desc="[Action Node] tripadvisor.com:F1:A4 + [Perception Node] tripadvisor.com:F1:P2 - Check and identify air conditioning amenity",
        parent=tripadvisor_node,
        critical=False
    )

    # [Action Node] tripadvisor.com:F4:A7 - Activity category filtering
    attraction1_ok = bool(tripadvisor_info and tripadvisor_info.attraction1_name and tripadvisor_info.attraction1_name.strip())
    attraction2_ok = bool(tripadvisor_info and tripadvisor_info.attraction2_name and tripadvisor_info.attraction2_name.strip())
    category1_ok = looks_like_indoor_category(tripadvisor_info.attraction1_category) if tripadvisor_info else False
    category2_ok = looks_like_indoor_category(tripadvisor_info.attraction2_category) if tripadvisor_info else False

    attractions_ok = attraction1_ok and attraction2_ok and (category1_ok or category2_ok)
    evaluator.add_custom_node(
        result=bool(attractions_ok),
        id="tripadvisor_action_attractions",
        desc="[Action Node] tripadvisor.com:F4:A7 - Find 2 indoor attractions (Museums or Galleries) near the hotel",
        parent=tripadvisor_node,
        critical=False
    )

    # Optional: Check for login wall documentation
    login_wall_mentioned = has_any_ci(answer, ['login', 'sign up', 'sign-up', 'wall', 'blocked', 'access limitation'])
    evaluator.add_custom_node(
        result=bool(login_wall_mentioned or ac_info_ok),
        id="tripadvisor_access_handling",
        desc="Handle TripAdvisor access (either full access or documented limitation)",
        parent=tripadvisor_node,
        critical=False
    )

    # 3.5 Output completeness section
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required output elements are present",
        parent=root,
        critical=False
    )

    # Check for detail-page links
    has_links = has_any_ci(answer, ['http://', 'https://', 'www.', '.com'])
    evaluator.add_custom_node(
        result=bool(has_links),
        id="output_has_links",
        desc="Answer includes detail-page links for relevant locations",
        parent=output_node,
        critical=False
    )

    # Check overall structure
    has_hospitals_section = hospital1_ok and hospital2_ok
    has_distance_section = anchor_distance_ok
    has_hotel_section = bool(hotel_info and hotel_info.hotel_name)
    has_tripadvisor_section = ac_info_ok or attraction1_ok

    evaluator.add_custom_node(
        result=bool(has_hospitals_section and has_distance_section and has_hotel_section and has_tripadvisor_section),
        id="output_structure",
        desc="Answer covers all major sections (hospitals, distances, hotel, TripAdvisor)",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
