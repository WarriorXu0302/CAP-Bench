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
TASK_ID = "task-2a47cb"
TASK_DESCRIPTION = 'I’m arranging an interstate consultation trip with gastroenterology specialists for my father, and I need to screen doctors and plan the itinerary. First, go to health.usnews.com and find one **Gastroenterologist** at **Mayo Clinic in Rochester, Minnesota** and one at **Cleveland Clinic in Ohio**. Both doctors must meet two strict criteria: **more than 20 years of experience** and a **perfect 5-star patient rating** (if no doctor at that location has a full 5 stars, relax the rating requirement to **4.5 stars or higher** and clearly note the reason for the relaxation).  \n\nAfter identifying the doctors, go to maps.google.com and calculate the estimated **air travel time** from **Mayo Clinic (Rochester)** to **Cleveland Clinic**.  \n\nFinally, return to the map view for **Mayo Clinic (Rochester)** and find a hotel. Prioritize hotels that meet both conditions: **within 1 mile walking distance** and **rated above 4.5**. If no hotel satisfies both at the same time, keep the rating threshold at **≥4.5** and relax walking distance to **within 1.5 miles**, and explicitly state this relaxation in the output.  \n\nOutput the following: both doctors’ names, years of experience, ratings, flight travel time between the two locations, and the selected hotel’s name, walking distance, rating, and all detail-page links.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DoctorInfo(BaseModel):
    """Doctor information extracted from the answer"""
    name: Optional[str] = None
    years_of_experience: Optional[str] = None
    rating: Optional[str] = None
    hospital: Optional[str] = None
    specialty: Optional[str] = None
    profile_link: Optional[str] = None


class TravelInfo(BaseModel):
    """Travel time information extracted from the answer"""
    travel_time: Optional[str] = None
    travel_mode: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel information extracted from the answer"""
    name: Optional[str] = None
    walking_distance: Optional[str] = None
    rating: Optional[str] = None
    location_context: Optional[str] = None
    detail_link: Optional[str] = None


class RelaxationNotes(BaseModel):
    """Relaxation notes extracted from the answer"""
    rating_relaxation_mentioned: Optional[bool] = False
    distance_relaxation_mentioned: Optional[bool] = False


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_mayo_doctor() -> str:
    return """
Extract the Mayo Clinic (Rochester, Minnesota) gastroenterologist information from the answer.

Return:
- name: the doctor's full name
- years_of_experience: the years of experience as stated (include units if present)
- rating: the patient rating as stated (e.g., "5.0", "4.5")
- hospital: the hospital name mentioned
- specialty: the specialty mentioned (should be gastroenterology related)
- profile_link: any URL or link to the doctor's profile page

If any field is missing, set it to null.
"""


def prompt_extract_cleveland_doctor() -> str:
    return """
Extract the Cleveland Clinic (Ohio) gastroenterologist information from the answer.

Return:
- name: the doctor's full name
- years_of_experience: the years of experience as stated (include units if present)
- rating: the patient rating as stated (e.g., "5.0", "4.5")
- hospital: the hospital name mentioned
- specialty: the specialty mentioned (should be gastroenterology related)
- profile_link: any URL or link to the doctor's profile page

If any field is missing, set it to null.
"""


def prompt_extract_travel_time() -> str:
    return """
Extract the air travel time between Mayo Clinic (Rochester) and Cleveland Clinic from the answer.

Return:
- travel_time: the estimated travel time as stated (include units like hours, minutes)
- travel_mode: the mode of transportation mentioned (should be air/flight related)

If any field is missing, set it to null.
"""


def prompt_extract_hotel() -> str:
    return """
Extract the hotel information near Mayo Clinic (Rochester) from the answer.

Return:
- name: the hotel name
- walking_distance: the walking distance as stated (include units like miles)
- rating: the hotel rating as stated
- location_context: any mention of proximity to Mayo Clinic
- detail_link: any URL or link to the hotel detail page

If any field is missing, set it to null.
"""


def prompt_extract_relaxation_notes() -> str:
    return """
Check if the answer mentions any relaxation of criteria.

Return:
- rating_relaxation_mentioned: true if the answer mentions relaxing the 5-star rating requirement to 4.5 stars, false otherwise
- distance_relaxation_mentioned: true if the answer mentions relaxing the 1 mile distance to 1.5 miles, false otherwise
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


def looks_like_years_experience(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num > 0 and has_any_ci(text, ['year', 'yr', 'experience'])


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return num > 0 and has_any_ci(text, ['mile', 'mi', 'km', 'meter', 'feet', 'ft'])


def looks_like_travel_time(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['hour', 'hr', 'minute', 'min', 'h', 'm'])


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://') or text.startswith('www.')


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
    mayo_doctor = await evaluator.extract(
        prompt=prompt_extract_mayo_doctor(),
        template_class=DoctorInfo,
        extraction_name="mayo_doctor"
    )

    cleveland_doctor = await evaluator.extract(
        prompt=prompt_extract_cleveland_doctor(),
        template_class=DoctorInfo,
        extraction_name="cleveland_doctor"
    )

    travel_info = await evaluator.extract(
        prompt=prompt_extract_travel_time(),
        template_class=TravelInfo,
        extraction_name="travel_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotel(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    relaxation_notes = await evaluator.extract(
        prompt=prompt_extract_relaxation_notes(),
        template_class=RelaxationNotes,
        extraction_name="relaxation_notes"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 US News Health - Mayo Clinic doctor
    mayo_node = evaluator.add_sequential(
        id="mayo_doctor_section",
        desc="US News Health - Find gastroenterologist at Mayo Clinic (Rochester, MN)",
        parent=root,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A5 - Multi-field search (specialty + hospital)
    mayo_search_ok = (
        has_any_ci(answer, ['mayo clinic', 'mayo']) and
        has_any_ci(answer, ['gastro', 'gi']) and
        has_any_ci(answer, ['rochester', 'minnesota', 'mn'])
    )
    evaluator.add_custom_node(
        result=bool(mayo_search_ok),
        id="mayo_search_action",
        desc="[Action Node] health.usnews.com:F2:A5 - Perform multi-field search combining gastroenterology specialty and Mayo Clinic hospital",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A6 - Multi-condition filtering
    mayo_name_ok = bool(mayo_doctor and mayo_doctor.name and mayo_doctor.name.strip())
    mayo_exp_num = extract_float(mayo_doctor.years_of_experience) if mayo_doctor else None
    mayo_exp_ok = mayo_exp_num is not None and mayo_exp_num > 20
    mayo_rating_num = extract_float(mayo_doctor.rating) if mayo_doctor else None
    mayo_rating_ok = mayo_rating_num is not None and mayo_rating_num >= 4.5

    mayo_filter_ok = mayo_name_ok and mayo_exp_ok and mayo_rating_ok

    evaluator.add_custom_node(
        result=bool(mayo_filter_ok),
        id="mayo_filter_action",
        desc="[Action Node] health.usnews.com:F2:A6 - Apply multi-condition filtering (>20 years experience and rating ≥4.5)",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] health.usnews.com:F2:P7 - Rating understanding
    mayo_rating_perception_ok = (
        mayo_rating_num is not None and
        looks_like_rating(mayo_doctor.rating if mayo_doctor else None)
    )
    evaluator.add_custom_node(
        result=bool(mayo_rating_perception_ok),
        id="mayo_rating_perception",
        desc="[Perception Node] health.usnews.com:F2:P7 - Extract and understand patient rating value",
        parent=mayo_node,
        critical=False
    )

    # Extra checks for Mayo doctor
    mayo_profile_link_ok = bool(mayo_doctor and looks_like_url(mayo_doctor.profile_link))
    evaluator.add_custom_node(
        result=bool(mayo_profile_link_ok),
        id="mayo_profile_link",
        desc="Provides profile link/URL for Mayo Clinic doctor",
        parent=mayo_node,
        critical=False
    )

    # 3.2 US News Health - Cleveland Clinic doctor
    cleveland_node = evaluator.add_sequential(
        id="cleveland_doctor_section",
        desc="US News Health - Find gastroenterologist at Cleveland Clinic (Ohio)",
        parent=root,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A5 - Multi-field search (specialty + hospital)
    cleveland_search_ok = (
        has_any_ci(answer, ['cleveland clinic', 'cleveland']) and
        has_any_ci(answer, ['gastro', 'gi']) and
        has_any_ci(answer, ['ohio', 'oh'])
    )
    evaluator.add_custom_node(
        result=bool(cleveland_search_ok),
        id="cleveland_search_action",
        desc="[Action Node] health.usnews.com:F2:A5 - Perform multi-field search combining gastroenterology specialty and Cleveland Clinic hospital",
        parent=cleveland_node,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A6 - Multi-condition filtering
    cleveland_name_ok = bool(cleveland_doctor and cleveland_doctor.name and cleveland_doctor.name.strip())
    cleveland_exp_num = extract_float(cleveland_doctor.years_of_experience) if cleveland_doctor else None
    cleveland_exp_ok = cleveland_exp_num is not None and cleveland_exp_num > 20
    cleveland_rating_num = extract_float(cleveland_doctor.rating) if cleveland_doctor else None
    cleveland_rating_ok = cleveland_rating_num is not None and cleveland_rating_num >= 4.5

    cleveland_filter_ok = cleveland_name_ok and cleveland_exp_ok and cleveland_rating_ok

    evaluator.add_custom_node(
        result=bool(cleveland_filter_ok),
        id="cleveland_filter_action",
        desc="[Action Node] health.usnews.com:F2:A6 - Apply multi-condition filtering (>20 years experience and rating ≥4.5)",
        parent=cleveland_node,
        critical=False
    )

    # [Perception Node] health.usnews.com:F2:P7 - Rating understanding
    cleveland_rating_perception_ok = (
        cleveland_rating_num is not None and
        looks_like_rating(cleveland_doctor.rating if cleveland_doctor else None)
    )
    evaluator.add_custom_node(
        result=bool(cleveland_rating_perception_ok),
        id="cleveland_rating_perception",
        desc="[Perception Node] health.usnews.com:F2:P7 - Extract and understand patient rating value",
        parent=cleveland_node,
        critical=False
    )

    # Extra checks for Cleveland doctor
    cleveland_profile_link_ok = bool(cleveland_doctor and looks_like_url(cleveland_doctor.profile_link))
    evaluator.add_custom_node(
        result=bool(cleveland_profile_link_ok),
        id="cleveland_profile_link",
        desc="Provides profile link/URL for Cleveland Clinic doctor",
        parent=cleveland_node,
        critical=False
    )

    # 3.3 Google Maps - Travel time calculation
    travel_node = evaluator.add_sequential(
        id="travel_section",
        desc="Google Maps - Calculate air travel time between two clinics",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    route_input_ok = (
        has_any_ci(answer, ['maps', 'google maps']) and
        has_any_ci(answer, ['mayo', 'rochester']) and
        has_any_ci(answer, ['cleveland'])
    )
    evaluator.add_custom_node(
        result=bool(route_input_ok),
        id="route_input_action",
        desc="[Action Node] maps.google.com:F2:A5 - Input route planning from Mayo Clinic (Rochester) to Cleveland Clinic",
        parent=travel_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Transportation mode selection
    flight_mode_ok = (
        travel_info and
        has_any_ci(travel_info.travel_mode, ['air', 'flight', 'fly', 'plane']) or
        has_any_ci(answer, ['air', 'flight', 'fly', 'plane', 'flying'])
    )
    evaluator.add_custom_node(
        result=bool(flight_mode_ok),
        id="flight_mode_action",
        desc="[Action Node] maps.google.com:F2:A7 - Select air/flight transportation mode",
        parent=travel_node,
        critical=False
    )

    # Extra check for travel time extraction
    travel_time_ok = bool(travel_info and looks_like_travel_time(travel_info.travel_time))
    evaluator.add_custom_node(
        result=bool(travel_time_ok),
        id="travel_time_extraction",
        desc="Extracts air travel time with proper time units",
        parent=travel_node,
        critical=False
    )

    # 3.4 Google Maps - Hotel search near Mayo Clinic
    hotel_node = evaluator.add_sequential(
        id="hotel_section",
        desc="Google Maps - Find hotel near Mayo Clinic (Rochester)",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F6:A6 - Tab switching/filtering to hotels
    hotel_search_ok = (
        has_any_ci(answer, ['hotel']) and
        has_any_ci(answer, ['mayo', 'rochester'])
    )
    evaluator.add_custom_node(
        result=bool(hotel_search_ok),
        id="hotel_search_action",
        desc="[Action Node] maps.google.com:F6:A6 - Switch to hotels tab/filter near Mayo Clinic",
        parent=hotel_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F6:P9 - Distance and rating extraction
    hotel_name_ok = bool(hotel_info and hotel_info.name and hotel_info.name.strip())
    hotel_distance_num = extract_float(hotel_info.walking_distance) if hotel_info else None
    hotel_distance_ok = hotel_distance_num is not None and hotel_distance_num <= 1.5
    hotel_rating_num = extract_float(hotel_info.rating) if hotel_info else None
    hotel_rating_ok = hotel_rating_num is not None and hotel_rating_num >= 4.5

    hotel_perception_ok = hotel_name_ok and hotel_distance_ok and hotel_rating_ok

    evaluator.add_custom_node(
        result=bool(hotel_perception_ok),
        id="hotel_perception",
        desc="[Perception Node] maps.google.com:F6:P9 - Extract hotel walking distance (≤1.5 miles) and rating (≥4.5)",
        parent=hotel_node,
        critical=False
    )

    # Extra checks for hotel
    hotel_link_ok = bool(hotel_info and looks_like_url(hotel_info.detail_link))
    evaluator.add_custom_node(
        result=bool(hotel_link_ok),
        id="hotel_link",
        desc="Provides detail link/URL for selected hotel",
        parent=hotel_node,
        critical=False
    )

    hotel_location_context_ok = bool(hotel_info and hotel_info.location_context and has_any_ci(hotel_info.location_context, ['mayo']))
    evaluator.add_custom_node(
        result=bool(hotel_location_context_ok),
        id="hotel_location_context",
        desc="Mentions hotel proximity to Mayo Clinic",
        parent=hotel_node,
        critical=False
    )

    # 3.5 Relaxation criteria handling
    relaxation_node = evaluator.add_parallel(
        id="relaxation_section",
        desc="Proper handling of relaxation criteria when strict requirements cannot be met",
        parent=root,
        critical=False
    )

    # Check if rating relaxation is appropriately noted
    rating_relaxation_appropriate = (
        (mayo_rating_num is not None and mayo_rating_num < 5.0 and relaxation_notes.rating_relaxation_mentioned) or
        (cleveland_rating_num is not None and cleveland_rating_num < 5.0 and relaxation_notes.rating_relaxation_mentioned) or
        (mayo_rating_num == 5.0 and cleveland_rating_num == 5.0)
    )
    evaluator.add_custom_node(
        result=bool(rating_relaxation_appropriate),
        id="rating_relaxation_noted",
        desc="Properly notes rating relaxation from 5.0 to ≥4.5 when no 5-star doctor is found",
        parent=relaxation_node,
        critical=False
    )

    # Check if distance relaxation is appropriately noted
    distance_relaxation_appropriate = (
        (hotel_distance_num is not None and hotel_distance_num > 1.0 and relaxation_notes.distance_relaxation_mentioned) or
        (hotel_distance_num is not None and hotel_distance_num <= 1.0)
    )
    evaluator.add_custom_node(
        result=bool(distance_relaxation_appropriate),
        id="distance_relaxation_noted",
        desc="Properly notes distance relaxation from 1 mile to 1.5 miles when no closer hotel meets rating requirement",
        parent=relaxation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
