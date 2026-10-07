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
TASK_ID = "task-9a23a2"
TASK_DESCRIPTION = 'My father has been diagnosed with early-onset Alzheimer’s disease, and we need to visit a neurology specialist at Mayo Clinic (Rochester campus). First, please find a neurologist on the Mayo Clinic website who specializes in “Memory Disorders” or “Alzheimer’s.” After identifying a doctor, check this doctor’s rating on US News Health to confirm they have a strong reputation.  \n\nFinally, we need to stay in Rochester for about one month (assume check-in is on the 1st of next month). Please find 3 Airbnb listings. Because my father has limited mobility and we need to cook for ourselves, be sure to filter for properties with both “Elevator” and “Kitchen.” If fewer than 3 listings are available after applying both filters, keep the filters and report the actual number found.  \n\nPlease provide the doctor’s name, US News rating, and the names and total prices of the 3 listings (or the actual number found). Note: We will ultimately need to call the doctor’s office to schedule an appointment, so prioritize doctors whose Mayo profile shows a phone number or an “Appointments/Request appointment” entry point. If the selected doctor’s page does not display a phone number, choose another doctor in the same campus and specialty direction.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DoctorInfo(BaseModel):
    """Doctor information extracted from the answer"""
    doctor_name: Optional[str] = None
    specialty_area: Optional[str] = None
    mayo_campus: Optional[str] = None
    has_phone_or_appointment_info: Optional[bool] = None


class USNewsRating(BaseModel):
    """US News Health rating extracted from the answer"""
    rating_text: Optional[str] = None
    rating_value: Optional[float] = None


class AirbnbListings(BaseModel):
    """Airbnb listings extracted from the answer"""
    listing_count: Optional[int] = None
    listing_names: Optional[List[str]] = Field(default_factory=list)
    listing_prices: Optional[List[str]] = Field(default_factory=list)
    elevator_filter_applied: Optional[bool] = None
    kitchen_filter_applied: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_doctor_from_answer() -> str:
    return """
Extract the neurologist information from Mayo Clinic that the user found in the answer.

Return:
- doctor_name: the full name of the doctor exactly as stated. If not present, set null.
- specialty_area: any mention of specialization (e.g., "Memory Disorders", "Alzheimer's", "Dementia"). If not present, set null.
- mayo_campus: the campus mentioned (should be Rochester). If not present, set null.
- has_phone_or_appointment_info: true if the answer mentions the doctor's profile shows a phone number or "Appointments"/"Request appointment" option; false otherwise; null if unclear.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_usnews_rating_from_answer() -> str:
    return """
From the answer, extract the US News Health rating for the doctor that was checked.

Return:
- rating_text: the rating exactly as stated (e.g., "4.5 out of 5", "Highly Rated", etc.). If not present, set null.
- rating_value: if a numeric rating is present, extract it as a float. If not present, set null.

If any field is missing, set it to null.
"""


def prompt_extract_airbnb_listings_from_answer() -> str:
    return """
From the answer, extract the Airbnb listing information for Rochester.

Return:
- listing_count: the number of listings reported (should be 3 or fewer if fewer were available). If not present, set null.
- listing_names: a list of listing names/titles exactly as stated. If not present, return empty list.
- listing_prices: a list of total prices for each listing exactly as stated (include currency symbols if present). If not present, return empty list.
- elevator_filter_applied: true if the answer indicates the "Elevator" filter was applied; false otherwise; null if unclear.
- kitchen_filter_applied: true if the answer indicates the "Kitchen" filter was applied; false otherwise; null if unclear.

If any field is missing, set it to null or empty list as appropriate.
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


def looks_like_doctor_name(name: Optional[str]) -> bool:
    if not name:
        return False
    # Check if it contains at least two words (first and last name)
    words = name.strip().split()
    return len(words) >= 2


def mentions_mayo_clinic(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['mayo clinic', 'mayo'])


def mentions_rochester(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'rochester')


def mentions_neurology_specialty(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['neurology', 'neurologist', 'memory disorders', 'memory disorder', 'alzheimer', 'dementia'])


def mentions_us_news(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['us news', 'u.s. news', 'usnews'])


def has_rating_value(rating_text: Optional[str], rating_value: Optional[float]) -> bool:
    if rating_value is not None and rating_value > 0:
        return True
    if rating_text and contains_digits(rating_text):
        return True
    return False


def mentions_airbnb(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'airbnb')


def mentions_next_month_checkin(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['next month', '1st of next month', 'first of next month', 'check-in', 'check in'])


def has_valid_listing_count(count: Optional[int]) -> bool:
    return count is not None and 1 <= count <= 3


def has_listing_details(names: Optional[List[str]], prices: Optional[List[str]]) -> bool:
    if not names or not prices:
        return False
    return len(names) > 0 and len(prices) > 0


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
    Restrict evaluator.verify to at most one usage.
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
    doctor_info = await evaluator.extract(
        prompt=prompt_extract_doctor_from_answer(),
        template_class=DoctorInfo,
        extraction_name="doctor_info"
    )

    usnews_rating = await evaluator.extract(
        prompt=prompt_extract_usnews_rating_from_answer(),
        template_class=USNewsRating,
        extraction_name="usnews_rating"
    )

    airbnb_info = await evaluator.extract(
        prompt=prompt_extract_airbnb_listings_from_answer(),
        template_class=AirbnbListings,
        extraction_name="airbnb_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Mayo Clinic section
    mayo_node = evaluator.add_sequential(
        id="mayo_clinic_section",
        desc="Find neurologist at Mayo Clinic Rochester specializing in Memory Disorders/Alzheimer's",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F4 - Filter by department and disease/specialty
    mayo_filter_ok = (mentions_mayo_clinic(answer) and
                      mentions_rochester(answer) and
                      mentions_neurology_specialty(answer))
    evaluator.add_custom_node(
        result=bool(mayo_filter_ok),
        id="mayo_filter_specialty",
        desc="[Action Node] mayoclinic.org:F4 - Navigate to Mayo Clinic and filter for neurology specialists with Memory Disorders/Alzheimer's focus at Rochester campus",
        parent=mayo_node,
        critical=False
    )

    # Doctor identification
    doctor_name_ok = looks_like_doctor_name(doctor_info.doctor_name)
    evaluator.add_custom_node(
        result=bool(doctor_name_ok),
        id="mayo_doctor_identified",
        desc="Successfully identified a neurologist with a valid name",
        parent=mayo_node,
        critical=False
    )

    # Phone/appointment info check
    phone_ok = bool(doctor_info.has_phone_or_appointment_info) if doctor_info.has_phone_or_appointment_info is not None else False
    evaluator.add_custom_node(
        result=bool(phone_ok),
        id="mayo_phone_appointment_info",
        desc="Doctor's profile shows phone number or appointment request option",
        parent=mayo_node,
        critical=False
    )

    # 3.2 US News Health section
    usnews_node = evaluator.add_sequential(
        id="usnews_health_section",
        desc="Check doctor's rating on US News Health",
        parent=root,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A5 - Search for doctor by name
    usnews_search_ok = (mentions_us_news(answer) and
                        doctor_name_ok and
                        has_any_ci(answer, [doctor_info.doctor_name] if doctor_info.doctor_name else []))
    evaluator.add_custom_node(
        result=bool(usnews_search_ok),
        id="usnews_search_doctor",
        desc="[Action Node] health.usnews.com:F2:A5 - Search for the identified doctor on US News Health using their name",
        parent=usnews_node,
        critical=False
    )

    # Rating extraction
    rating_ok = has_rating_value(usnews_rating.rating_text, usnews_rating.rating_value)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="usnews_rating_extracted",
        desc="Successfully extracted US News Health rating for the doctor",
        parent=usnews_node,
        critical=False
    )

    # 3.3 Airbnb section
    airbnb_node = evaluator.add_sequential(
        id="airbnb_section",
        desc="Find Airbnb listings in Rochester with Elevator and Kitchen for one month stay",
        parent=root,
        critical=False
    )

    # [Action Node] airbnb.com:F4:A38 - Search for Rochester location
    airbnb_location_ok = (mentions_airbnb(answer) and mentions_rochester(answer))
    evaluator.add_custom_node(
        result=bool(airbnb_location_ok),
        id="airbnb_search_rochester",
        desc="[Action Node] airbnb.com:F4:A38 - Search for listings in Rochester location",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F1:A6 - Date range selection
    date_selection_ok = mentions_next_month_checkin(answer)
    evaluator.add_custom_node(
        result=bool(date_selection_ok),
        id="airbnb_date_selection",
        desc="[Action Node] airbnb.com:F1:A6 - Select check-in date (1st of next month) and approximately one month stay",
        parent=airbnb_node,
        critical=False
    )

    # [Action Node] airbnb.com:F2:A11 - Apply multiple amenity filters (Elevator and Kitchen)
    elevator_filter_ok = bool(airbnb_info.elevator_filter_applied) if airbnb_info.elevator_filter_applied is not None else False
    kitchen_filter_ok = bool(airbnb_info.kitchen_filter_applied) if airbnb_info.kitchen_filter_applied is not None else False
    both_filters_ok = elevator_filter_ok and kitchen_filter_ok
    evaluator.add_custom_node(
        result=bool(both_filters_ok),
        id="airbnb_amenity_filters",
        desc="[Action Node] airbnb.com:F2:A11 - Apply both Elevator and Kitchen amenity filters",
        parent=airbnb_node,
        critical=False
    )

    # Listing count validation
    listing_count_ok = has_valid_listing_count(airbnb_info.listing_count)
    evaluator.add_custom_node(
        result=bool(listing_count_ok),
        id="airbnb_listing_count",
        desc="Found appropriate number of listings (1-3) with both filters applied",
        parent=airbnb_node,
        critical=False
    )

    # Listing details provided
    listing_details_ok = has_listing_details(airbnb_info.listing_names, airbnb_info.listing_prices)
    evaluator.add_custom_node(
        result=bool(listing_details_ok),
        id="airbnb_listing_details",
        desc="Provided listing names and total prices for found properties",
        parent=airbnb_node,
        critical=False
    )

    # 3.4 Final output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="All required information provided in final answer",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(doctor_name_ok),
        id="output_doctor_name",
        desc="Doctor's name provided",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="output_rating",
        desc="US News rating provided",
        parent=output_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(listing_details_ok),
        id="output_listings",
        desc="Listing names and prices provided",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
