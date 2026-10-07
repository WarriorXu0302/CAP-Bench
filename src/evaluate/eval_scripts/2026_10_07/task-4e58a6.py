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
TASK_ID = "task-4e58a6"
TASK_DESCRIPTION = "I'm planning to visit the Getty Center in Los Angeles to check out their current special exhibitions.\n\nFirst, please go to their official website and find a special exhibition with 'Future' or 'Science' related concepts in its name. If no such exhibition is available, find the most popular one. Confirm its specific start and end dates.\n\nAfter obtaining these dates, please search for hotels near the Getty Center on Booking.com. The booking should be for an upcoming arbitrary weekend during this exhibition period (two nights, check-in Friday, check-out Sunday). I'm looking for three hotels with a rating of 8.5 or higher and marked with 'free cancellation'.\n\nFinally, use Google Maps to determine the driving time from each of these three hotels to the Getty Center. Select one with a driving time of 15 minutes or less, and provide me with the hotel name, price, and driving time."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ExhibitionInfo(BaseModel):
    """Exhibition details extracted from the answer"""
    exhibition_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class HotelInfo(BaseModel):
    """Hotel details extracted from the answer"""
    hotel_names: Optional[List[str]] = Field(default_factory=list)
    ratings: Optional[List[str]] = Field(default_factory=list)
    free_cancellation_mentions: Optional[str] = None


class SelectedHotelInfo(BaseModel):
    """Selected hotel details extracted from the answer"""
    selected_hotel_name: Optional[str] = None
    hotel_price: Optional[str] = None
    driving_time: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_exhibition_from_answer() -> str:
    return """
Extract the Getty Center exhibition details from the answer.

Return:
- exhibition_name: the name of the special exhibition mentioned (should contain 'Future' or 'Science' concepts if available).
- start_date: the specific start date of the exhibition exactly as stated.
- end_date: the specific end date of the exhibition exactly as stated.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_hotels_from_answer() -> str:
    return """
From the answer, extract the hotel information found on Booking.com near the Getty Center.

Return:
- hotel_names: a list of hotel names mentioned (should be three hotels if the task was completed).
- ratings: a list of ratings mentioned for these hotels (should be 8.5 or higher).
- free_cancellation_mentions: any text mentioning 'free cancellation' for the hotels.

If any field is missing, set it to null or an empty list.
"""


def prompt_extract_selected_hotel_from_answer() -> str:
    return """
From the answer, extract the final selected hotel details.

Return:
- selected_hotel_name: the name of the hotel selected (with driving time of 15 minutes or less).
- hotel_price: the price of the selected hotel exactly as stated.
- driving_time: the driving time from the hotel to Getty Center exactly as stated.

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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats: "January 15, 2024", "2024-01-15", "15/01/2024", etc.
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
    ]
    return any(re.search(p, text.lower()) for p in date_patterns)


def mentions_future_or_science(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['future', 'science'])


def mentions_rating_85_or_higher(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for ratings >= 8.5
    ratings = re.findall(r'(\d+\.\d+|\d+)\s*(rating|score|stars)?', text.lower())
    for rating_match in ratings:
        try:
            rating_val = float(rating_match[0])
            if rating_val >= 8.5:
                return True
        except Exception:
            continue
    return False


def mentions_free_cancellation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['free cancellation', 'free cancel'])


def mentions_weekend_friday_sunday(text: Optional[str]) -> bool:
    if not text:
        return False
    friday = has_any_ci(text, ['friday', 'fri'])
    sunday = has_any_ci(text, ['sunday', 'sun'])
    two_nights = has_any_ci(text, ['two nights', '2 nights'])
    weekend = has_any_ci(text, ['weekend'])
    return (friday and sunday) or two_nights or weekend


def extract_driving_minutes(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Look for patterns like "10 minutes", "15 min", "12 mins"
    minutes_match = re.search(r'(\d+)\s*(minute|min|mins)', text.lower())
    if minutes_match:
        try:
            return float(minutes_match.group(1))
        except Exception:
            return None
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
    exhibition_info = await evaluator.extract(
        prompt=prompt_extract_exhibition_from_answer(),
        template_class=ExhibitionInfo,
        extraction_name="exhibition_info"
    )

    hotel_info = await evaluator.extract(
        prompt=prompt_extract_hotels_from_answer(),
        template_class=HotelInfo,
        extraction_name="hotel_info"
    )

    selected_hotel_info = await evaluator.extract(
        prompt=prompt_extract_selected_hotel_from_answer(),
        template_class=SelectedHotelInfo,
        extraction_name="selected_hotel_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Getty Center exhibition part
    getty_node = evaluator.add_sequential(
        id="getty_section",
        desc="Getty Center special exhibition identification with dates",
        parent=root,
        critical=False
    )

    # [Action Node] getty.edu:F2:A1 - Navigate to Getty website and switch tabs to find exhibitions
    getty_action_ok = has_any_ci(answer, ['getty']) and (
        has_any_ci(answer, ['exhibition', 'special exhibition']) or
        has_any_ci(answer, ['current', 'future'])
    )
    evaluator.add_custom_node(
        result=bool(getty_action_ok),
        id="getty_action_navigate_tabs",
        desc="[Action Node] getty.edu:F2:A1 - Navigate to Getty Center website and switch to Current/Future exhibitions tab",
        parent=getty_node,
        critical=False
    )

    # Check exhibition name contains Future or Science concepts
    exhibition_name_ok = False
    if exhibition_info and exhibition_info.exhibition_name:
        exhibition_name_ok = mentions_future_or_science(exhibition_info.exhibition_name)

    evaluator.add_custom_node(
        result=bool(exhibition_name_ok or has_any_ci(answer, ['future', 'science'])),
        id="getty_exhibition_name_concept",
        desc="Exhibition name contains 'Future' or 'Science' related concepts",
        parent=getty_node,
        critical=False
    )

    # Check start and end dates are present
    start_date_ok = looks_like_date(exhibition_info.start_date if exhibition_info else None)
    end_date_ok = looks_like_date(exhibition_info.end_date if exhibition_info else None)

    evaluator.add_custom_node(
        result=bool(start_date_ok and end_date_ok),
        id="getty_exhibition_dates",
        desc="Exhibition start and end dates are confirmed",
        parent=getty_node,
        critical=False
    )

    # 3.2 Booking.com hotel search part
    booking_node = evaluator.add_sequential(
        id="booking_section",
        desc="Booking.com hotel search near Getty Center with filters",
        parent=root,
        critical=False
    )

    # Check Booking.com was used
    booking_action_ok = has_any_ci(answer, ['booking.com', 'booking'])
    evaluator.add_custom_node(
        result=bool(booking_action_ok),
        id="booking_action_search",
        desc="Navigate to Booking.com and search for hotels near Getty Center",
        parent=booking_node,
        critical=False
    )

    # Check weekend dates (Friday-Sunday, two nights)
    weekend_ok = mentions_weekend_friday_sunday(answer)
    evaluator.add_custom_node(
        result=bool(weekend_ok),
        id="booking_weekend_dates",
        desc="Booking for weekend (Friday check-in, Sunday check-out, two nights)",
        parent=booking_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A11 - Apply rating filter (8.5 or higher)
    rating_filter_ok = mentions_rating_85_or_higher(answer)
    evaluator.add_custom_node(
        result=bool(rating_filter_ok),
        id="booking_action_rating_filter",
        desc="[Action Node] Booking.com:F2:A11 - Apply rating filter for 8.5 or higher",
        parent=booking_node,
        critical=False
    )

    # [Action Node] Booking.com:F2:A7 - Apply free cancellation filter
    free_cancel_ok = mentions_free_cancellation(answer)
    evaluator.add_custom_node(
        result=bool(free_cancel_ok),
        id="booking_action_free_cancellation",
        desc="[Action Node] Booking.com:F2:A7 - Apply 'free cancellation' filter checkbox",
        parent=booking_node,
        critical=False
    )

    # Check three hotels were found
    three_hotels_ok = False
    if hotel_info and hotel_info.hotel_names:
        three_hotels_ok = len(hotel_info.hotel_names) >= 3

    evaluator.add_custom_node(
        result=bool(three_hotels_ok or has_any_ci(answer, ['three hotels', '3 hotels'])),
        id="booking_three_hotels",
        desc="Three hotels identified meeting the criteria",
        parent=booking_node,
        critical=False
    )

    # 3.3 Google Maps driving time verification part
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps driving time verification for hotels to Getty Center",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Use Google Maps to check driving time
    maps_action_ok = has_any_ci(answer, ['google maps', 'maps']) and has_any_ci(answer, ['driving', 'drive'])
    evaluator.add_custom_node(
        result=bool(maps_action_ok),
        id="maps_action_driving_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Use Google Maps to check driving time (driving mode selected)",
        parent=maps_node,
        critical=False
    )

    # Check driving times were obtained for hotels
    driving_times_ok = has_any_ci(answer, ['driving time', 'drive time', 'minutes'])
    evaluator.add_custom_node(
        result=bool(driving_times_ok),
        id="maps_driving_times_obtained",
        desc="Driving times from hotels to Getty Center were determined",
        parent=maps_node,
        critical=False
    )

    # 3.4 Final selection part
    selection_node = evaluator.add_sequential(
        id="selection_section",
        desc="Final hotel selection with 15 minutes or less driving time",
        parent=root,
        critical=False
    )

    # Check a hotel was selected
    hotel_selected_ok = False
    if selected_hotel_info and selected_hotel_info.selected_hotel_name:
        hotel_selected_ok = bool(selected_hotel_info.selected_hotel_name.strip())

    evaluator.add_custom_node(
        result=bool(hotel_selected_ok or has_any_ci(answer, ['selected', 'recommend', 'choose'])),
        id="selection_hotel_selected",
        desc="One hotel selected from the three candidates",
        parent=selection_node,
        critical=False
    )

    # Check driving time is 15 minutes or less
    driving_time_15min_ok = False
    if selected_hotel_info and selected_hotel_info.driving_time:
        minutes = extract_driving_minutes(selected_hotel_info.driving_time)
        if minutes is not None and minutes <= 15:
            driving_time_15min_ok = True

    evaluator.add_custom_node(
        result=bool(driving_time_15min_ok or has_any_ci(answer, ['15 minutes or less', 'under 15', 'within 15'])),
        id="selection_driving_time_criteria",
        desc="Selected hotel has driving time of 15 minutes or less to Getty Center",
        parent=selection_node,
        critical=False
    )

    # Check hotel name is provided
    hotel_name_provided = False
    if selected_hotel_info and selected_hotel_info.selected_hotel_name:
        hotel_name_provided = bool(selected_hotel_info.selected_hotel_name.strip())

    evaluator.add_custom_node(
        result=bool(hotel_name_provided),
        id="selection_hotel_name_provided",
        desc="Hotel name is provided in the final answer",
        parent=selection_node,
        critical=False
    )

    # Check price is provided
    price_provided = False
    if selected_hotel_info and selected_hotel_info.hotel_price:
        price_provided = bool(selected_hotel_info.hotel_price.strip())

    evaluator.add_custom_node(
        result=bool(price_provided or has_any_ci(answer, ['price', '$', 'cost'])),
        id="selection_price_provided",
        desc="Hotel price is provided in the final answer",
        parent=selection_node,
        critical=False
    )

    # Check driving time is provided
    driving_time_provided = False
    if selected_hotel_info and selected_hotel_info.driving_time:
        driving_time_provided = bool(selected_hotel_info.driving_time.strip())

    evaluator.add_custom_node(
        result=bool(driving_time_provided),
        id="selection_driving_time_provided",
        desc="Driving time is provided in the final answer",
        parent=selection_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
