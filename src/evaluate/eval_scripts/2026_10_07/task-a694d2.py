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
TASK_ID = "task-a694d2"
TASK_DESCRIPTION = "As a book club organizer, I'm planning a book club meeting next week in downtown Chicago with a 'Lighthearted' theme. First, go to TheStoryGraph and help me select a highly-rated book (4.0 or above) that has both 'Lighthearted' and 'Funny' mood tags and is no more than 350 pages. Please check the tag statistics on the detail page to confirm that at least 70% of readers agree with these two moods.\n\nOnce you have the book title, use Google Maps to find 'The Book Cellar' bookstore near Chicago's River North area (assuming this is where we'll buy the books). Using it as a central point, find a restaurant within walking distance that has an OpenTable rating of 4.5 or higher. It must have 'Outdoor Seating', as the weather is expected to be good. Finally, confirm if this restaurant is '$$' (mid-range) priced, and if a table for 4 can be reserved for next Saturday at 6 PM."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BookInfo(BaseModel):
    """Book details extracted from the answer for TheStoryGraph selection"""
    book_title: Optional[str] = None
    rating_text: Optional[str] = None
    pages_text: Optional[str] = None
    lighthearted_percentage_text: Optional[str] = None
    funny_percentage_text: Optional[str] = None


class RestaurantInfo(BaseModel):
    """Restaurant details extracted from the answer for OpenTable selection"""
    restaurant_name: Optional[str] = None
    opentable_rating_text: Optional[str] = None
    price_range_text: Optional[str] = None
    outdoor_seating_mentioned: Optional[str] = None
    reservation_availability_text: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_book_from_answer() -> str:
    return """
Extract the book selection details from TheStoryGraph reported in the answer.

Return:
- book_title: the title of the selected book exactly as stated. If not present, set null.
- rating_text: the book's rating exactly as written (include any rating format present). If not present, set null.
- pages_text: the page count exactly as written (include units if present). If not present, set null.
- lighthearted_percentage_text: the percentage of readers who agree with the 'Lighthearted' mood tag exactly as stated. If not present, set null.
- funny_percentage_text: the percentage of readers who agree with the 'Funny' mood tag exactly as stated. If not present, set null.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_restaurant_from_answer() -> str:
    return """
From the answer, extract the restaurant selection details from OpenTable:

- restaurant_name: the name of the selected restaurant exactly as stated. If not present, set null.
- opentable_rating_text: the OpenTable rating exactly as written (include any rating format present). If not present, set null.
- price_range_text: the price range indicator exactly as stated (e.g., "$$", "$$$"). If not present, set null.
- outdoor_seating_mentioned: any text that confirms outdoor seating availability. If not present, set null.
- reservation_availability_text: any text about reservation availability for Saturday at 6 PM for 4 people. If not present, set null.

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


def extract_percentage(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    num = extract_float(text)
    if num is not None and 0 <= num <= 100:
        return num
    return None


def looks_like_rating(text: Optional[str], min_rating: float = 0.0) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    if rating is None:
        return False
    return rating >= min_rating


def looks_like_page_count(text: Optional[str], max_pages: int = 999999) -> bool:
    if not text:
        return False
    pages = extract_float(text)
    if pages is None:
        return False
    return pages <= max_pages


def mentions_moods(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['lighthearted', 'funny'])


def mentions_bookstore(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['book cellar', 'bookstore'])


def mentions_outdoor_seating(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['outdoor seating', 'outdoor', 'patio', 'terrace'])


def looks_like_price_range_mid(text: Optional[str]) -> bool:
    if not text:
        return False
    return '$$' in text and '$$$' not in text


def mentions_reservation_context(text: Optional[str]) -> bool:
    if not text:
        return False
    has_saturday = has_any_ci(text, ['saturday', 'sat'])
    has_time = has_any_ci(text, ['6 pm', '6pm', '6:00', '18:00'])
    has_people = has_any_ci(text, ['4 people', 'table for 4', 'party of 4', 'four'])
    return has_saturday or has_time or has_people


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
    book_info = await evaluator.extract(
        prompt=prompt_extract_book_from_answer(),
        template_class=BookInfo,
        extraction_name="book_selection"
    )

    restaurant_info = await evaluator.extract(
        prompt=prompt_extract_restaurant_from_answer(),
        template_class=RestaurantInfo,
        extraction_name="restaurant_selection"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 TheStoryGraph part
    thestorygraph_node = evaluator.add_sequential(
        id="thestorygraph_section",
        desc="TheStoryGraph book selection with mood and page filters",
        parent=root,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A1 - Multi-select mood tags (Lighthearted AND Funny)
    thestorygraph_multiselect_ok = (has_any_ci(answer, ['thestorygraph', 'storygraph']) and
                                     has_any_ci(answer, ['lighthearted']) and
                                     has_any_ci(answer, ['funny']))
    evaluator.add_custom_node(
        result=bool(thestorygraph_multiselect_ok),
        id="thestorygraph_action_multiselect_moods",
        desc="[Action Node] thestorygraph.com:F2:A1 - Apply multi-select filter for both 'Lighthearted' and 'Funny' mood tags",
        parent=thestorygraph_node,
        critical=False
    )

    # [Action Node] thestorygraph.com:F2:A4 - Adjust page count slider (no more than 350)
    pages_num = extract_float(book_info.pages_text)
    pages_filter_ok = (pages_num is not None and pages_num <= 350) or has_any_ci(answer, ['350 pages', '350', 'page'])
    evaluator.add_custom_node(
        result=bool(pages_filter_ok),
        id="thestorygraph_action_page_slider",
        desc="[Action Node] thestorygraph.com:F2:A4 - Adjust page count slider/filter to limit results to 350 pages or fewer",
        parent=thestorygraph_node,
        critical=False
    )

    # [Perception Node] thestorygraph.com:F2:P1 - Verify mood tag percentages (at least 70% for both)
    lighthearted_pct = extract_percentage(book_info.lighthearted_percentage_text)
    funny_pct = extract_percentage(book_info.funny_percentage_text)

    lighthearted_ok = lighthearted_pct is not None and lighthearted_pct >= 70
    funny_ok = funny_pct is not None and funny_pct >= 70

    percentages_ok = lighthearted_ok and funny_ok

    evaluator.add_custom_node(
        result=bool(percentages_ok),
        id="thestorygraph_perception_mood_percentages",
        desc="[Perception Node] thestorygraph.com:F2:P1 - Verify that at least 70% of readers agree with both 'Lighthearted' and 'Funny' mood tags on the book detail page",
        parent=thestorygraph_node,
        critical=False
    )

    # Additional checks for TheStoryGraph section
    rating_ok = looks_like_rating(book_info.rating_text, min_rating=4.0)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="thestorygraph_rating_check",
        desc="Book has a rating of 4.0 or above",
        parent=thestorygraph_node,
        critical=False
    )

    book_title_ok = bool(book_info and book_info.book_title and book_info.book_title.strip())
    evaluator.add_custom_node(
        result=bool(book_title_ok),
        id="thestorygraph_book_title",
        desc="A specific book title is provided",
        parent=thestorygraph_node,
        critical=False
    )

    # 3.2 Google Maps part
    googlemaps_node = evaluator.add_sequential(
        id="googlemaps_section",
        desc="Google Maps location of The Book Cellar bookstore",
        parent=root,
        critical=False
    )

    bookstore_mentioned = has_any_ci(answer, ['book cellar', 'bookstore'])
    location_mentioned = has_any_ci(answer, ['river north', 'chicago', 'downtown'])

    evaluator.add_custom_node(
        result=bool(bookstore_mentioned and location_mentioned),
        id="googlemaps_bookstore_location",
        desc="Locate 'The Book Cellar' bookstore near Chicago's River North area on Google Maps",
        parent=googlemaps_node,
        critical=False
    )

    # 3.3 OpenTable part
    opentable_node = evaluator.add_sequential(
        id="opentable_section",
        desc="OpenTable restaurant search and reservation validation",
        parent=root,
        critical=False
    )

    # [Action Node] opentable.com:F1:A1 - Multi-criteria filter (Outdoor Seating and $$ price range)
    opentable_mentioned = has_any_ci(answer, ['opentable'])
    outdoor_filter_ok = mentions_outdoor_seating(restaurant_info.outdoor_seating_mentioned) or mentions_outdoor_seating(answer)
    price_filter_ok = looks_like_price_range_mid(restaurant_info.price_range_text) or has_any_ci(answer, ['$$', 'mid-range', 'moderate'])

    evaluator.add_custom_node(
        result=bool(opentable_mentioned and outdoor_filter_ok and price_filter_ok),
        id="opentable_action_multicriteria_filter",
        desc="[Action Node] opentable.com:F1:A1 - Apply filters for 'Outdoor Seating' and '$$' price range",
        parent=opentable_node,
        critical=False
    )

    # [Perception Node] opentable.com:F1:P6 - Verify outdoor seating facility
    outdoor_seating_confirmed = mentions_outdoor_seating(restaurant_info.outdoor_seating_mentioned) or mentions_outdoor_seating(answer)
    evaluator.add_custom_node(
        result=bool(outdoor_seating_confirmed),
        id="opentable_perception_outdoor_seating",
        desc="[Perception Node] opentable.com:F1:P6 - Confirm the restaurant has 'Outdoor Seating' facility",
        parent=opentable_node,
        critical=False
    )

    # Additional checks for OpenTable section
    restaurant_name_ok = bool(restaurant_info and restaurant_info.restaurant_name and restaurant_info.restaurant_name.strip())
    evaluator.add_custom_node(
        result=bool(restaurant_name_ok),
        id="opentable_restaurant_name",
        desc="A specific restaurant name is provided",
        parent=opentable_node,
        critical=False
    )

    rating_ok = looks_like_rating(restaurant_info.opentable_rating_text, min_rating=4.5)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="opentable_rating_check",
        desc="Restaurant has an OpenTable rating of 4.5 or higher",
        parent=opentable_node,
        critical=False
    )

    walking_distance_ok = has_any_ci(answer, ['walking distance', 'walk', 'nearby', 'close'])
    evaluator.add_custom_node(
        result=bool(walking_distance_ok),
        id="opentable_walking_distance",
        desc="Restaurant is within walking distance of the bookstore",
        parent=opentable_node,
        critical=False
    )

    price_confirmed = looks_like_price_range_mid(restaurant_info.price_range_text)
    evaluator.add_custom_node(
        result=bool(price_confirmed),
        id="opentable_price_confirmed",
        desc="Price range is confirmed as '$$' (mid-range)",
        parent=opentable_node,
        critical=False
    )

    reservation_checked = (restaurant_info.reservation_availability_text is not None and
                          restaurant_info.reservation_availability_text.strip() != '') or \
                         mentions_reservation_context(answer)
    evaluator.add_custom_node(
        result=bool(reservation_checked),
        id="opentable_reservation_check",
        desc="Reservation availability for Saturday at 6 PM for 4 people is checked",
        parent=opentable_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
