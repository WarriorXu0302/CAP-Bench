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
TASK_ID = "task-76041b"
TASK_DESCRIPTION = "I am planning to move to the South Lake Union area in Seattle, as I've heard it's suitable for young professionals. First, please help me search for apartments in this area on Apartments.com. My budget is within $3500 per month. I need a 2-bedroom unit with an in-unit washer & dryer. Please filter out the top 3 apartments that meet these criteria and note down their specific addresses.\n\nNext, I'd like to check the convenience of living. For each of these three addresses, please go to Yelp and see if there are any coffee shops with a rating of 4.0 or higher within walking distance (search within 0.5 miles).\n\nFinally, please provide a comparative summary: List the names of these three apartments, their monthly rent, and for each, the name and Yelp rating of the highest-rated coffee shop nearby. If there are no eligible coffee shops around a particular apartment, please state that as well."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApartmentInfo(BaseModel):
    """Information about a single apartment"""
    name: Optional[str] = None
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    coffee_shop_name: Optional[str] = None
    coffee_shop_rating: Optional[str] = None


class ApartmentsExtraction(BaseModel):
    """All three apartments extracted from the answer"""
    apartment_1: Optional[ApartmentInfo] = None
    apartment_2: Optional[ApartmentInfo] = None
    apartment_3: Optional[ApartmentInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartments_from_answer() -> str:
    return """
Extract the three apartments from the answer that the user found on Apartments.com in South Lake Union, Seattle.

For each apartment, extract:
- name: the apartment complex name
- address: the specific address
- monthly_rent: the monthly rent amount (include $ and units if present)
- coffee_shop_name: the name of the highest-rated nearby coffee shop from Yelp (if mentioned)
- coffee_shop_rating: the Yelp rating of that coffee shop (if mentioned)

Return apartment_1, apartment_2, and apartment_3. If any field is missing, set it to null.
If fewer than 3 apartments are mentioned, leave the extra apartment objects as null.
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


def looks_like_rent(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_dollar = '$' in text or ci_contains(text, 'dollar')
    return has_number and (has_dollar or True)


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def count_valid_apartments(apt_data: ApartmentsExtraction) -> int:
    count = 0
    for apt in [apt_data.apartment_1, apt_data.apartment_2, apt_data.apartment_3]:
        if apt and apt.name:
            count += 1
    return count


def count_apartments_with_addresses(apt_data: ApartmentsExtraction) -> int:
    count = 0
    for apt in [apt_data.apartment_1, apt_data.apartment_2, apt_data.apartment_3]:
        if apt and apt.address and apt.address.strip():
            count += 1
    return count


def count_apartments_with_rent(apt_data: ApartmentsExtraction) -> int:
    count = 0
    for apt in [apt_data.apartment_1, apt_data.apartment_2, apt_data.apartment_3]:
        if apt and apt.monthly_rent and looks_like_rent(apt.monthly_rent):
            count += 1
    return count


def count_apartments_with_coffee_info(apt_data: ApartmentsExtraction) -> int:
    count = 0
    for apt in [apt_data.apartment_1, apt_data.apartment_2, apt_data.apartment_3]:
        if apt and apt.name:
            if apt.coffee_shop_name or ci_contains(str(apt.coffee_shop_name), 'no') or ci_contains(str(apt.coffee_shop_name), 'none'):
                count += 1
    return count


def all_coffee_ratings_valid(apt_data: ApartmentsExtraction) -> bool:
    for apt in [apt_data.apartment_1, apt_data.apartment_2, apt_data.apartment_3]:
        if apt and apt.name and apt.coffee_shop_rating:
            rating_num = extract_float(apt.coffee_shop_rating)
            if rating_num is not None and rating_num < 4.0:
                return False
    return True


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
    apt_data = await evaluator.extract(
        prompt=prompt_extract_apartments_from_answer(),
        template_class=ApartmentsExtraction,
        extraction_name="apartments_extraction"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Apartments.com section
    apartments_node = evaluator.add_sequential(
        id="apartments_com_section",
        desc="Search and filter apartments on Apartments.com in South Lake Union",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A30 - Location search with autocomplete
    location_action_ok = (has_any_ci(answer, ['apartments.com', 'apartments com']) and
                         has_any_ci(answer, ['south lake union', 'south lake', 'slu']) and
                         has_any_ci(answer, ['seattle']))
    evaluator.add_custom_node(
        result=bool(location_action_ok),
        id="apartments_location_search",
        desc="[Action Node] apartments.com:F1:A30 - Search for apartments in South Lake Union area in Seattle using location search",
        parent=apartments_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A5 - Multi-criteria filtering
    budget_mention = has_any_ci(answer, ['3500', '$3500', '$3,500', '3,500'])
    bedroom_mention = has_any_ci(answer, ['2 bed', '2-bed', 'two bed', '2 br', '2br'])
    washer_dryer_mention = has_any_ci(answer, ['washer', 'dryer', 'w/d', 'laundry'])
    inunit_mention = has_any_ci(answer, ['in-unit', 'in unit', 'inside'])

    filter_action_ok = budget_mention and bedroom_mention and (washer_dryer_mention or inunit_mention)
    evaluator.add_custom_node(
        result=bool(filter_action_ok),
        id="apartments_multi_filter",
        desc="[Action Node] apartments.com:F1:A5 - Apply multiple filters: budget ($3500), 2-bedroom, in-unit washer & dryer",
        parent=apartments_node,
        critical=False
    )

    # Check that 3 apartments were found
    num_apartments = count_valid_apartments(apt_data)
    evaluator.add_custom_node(
        result=bool(num_apartments == 3),
        id="apartments_found_three",
        desc="Found exactly 3 apartments meeting the criteria",
        parent=apartments_node,
        critical=False
    )

    # Check that addresses were noted
    num_with_address = count_apartments_with_addresses(apt_data)
    evaluator.add_custom_node(
        result=bool(num_with_address >= 2),
        id="apartments_addresses_noted",
        desc="Specific addresses noted for the apartments (at least 2 of 3)",
        parent=apartments_node,
        critical=False
    )

    # 3.2 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Search for coffee shops on Yelp near each apartment address",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Complex filtering (distance and rating)
    yelp_mention = has_any_ci(answer, ['yelp'])
    coffee_mention = has_any_ci(answer, ['coffee', 'cafe', 'café'])
    distance_mention = has_any_ci(answer, ['0.5 mile', '0.5 mi', 'half mile', 'walking distance'])
    rating_mention = has_any_ci(answer, ['4.0', '4 star', 'rating'])

    yelp_filter_ok = yelp_mention and coffee_mention and (distance_mention or rating_mention)
    evaluator.add_custom_node(
        result=bool(yelp_filter_ok),
        id="yelp_distance_rating_filter",
        desc="[Action Node] yelp.com:F1:A2 - Search with distance (0.5 miles) and rating (4.0+) filters for coffee shops",
        parent=yelp_node,
        critical=False
    )

    # Check that searches were done for each apartment
    num_with_coffee_info = count_apartments_with_coffee_info(apt_data)
    evaluator.add_custom_node(
        result=bool(num_with_coffee_info >= 2),
        id="yelp_searches_performed",
        desc="Yelp searches performed for each apartment address (at least 2 of 3 have coffee info or explicit 'none' statement)",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F1:P1 - Extract and compare ratings to find highest
    ratings_valid = all_coffee_ratings_valid(apt_data)
    highest_mention = has_any_ci(answer, ['highest', 'best', 'top'])

    rating_extraction_ok = ratings_valid or highest_mention
    evaluator.add_custom_node(
        result=bool(rating_extraction_ok),
        id="yelp_rating_comparison",
        desc="[Perception Node] yelp.com:F1:P1 - Extract ratings and identify the highest-rated coffee shop for each location",
        parent=yelp_node,
        critical=False
    )

    # 3.3 Summary section
    summary_node = evaluator.add_sequential(
        id="summary_section",
        desc="Provide comparative summary with all required information",
        parent=root,
        critical=False
    )

    # Check apartment names are provided
    evaluator.add_custom_node(
        result=bool(num_apartments >= 2),
        id="summary_apartment_names",
        desc="Summary includes apartment names (at least 2 of 3)",
        parent=summary_node,
        critical=False
    )

    # Check monthly rent is provided
    num_with_rent = count_apartments_with_rent(apt_data)
    evaluator.add_custom_node(
        result=bool(num_with_rent >= 2),
        id="summary_monthly_rent",
        desc="Summary includes monthly rent for apartments (at least 2 of 3)",
        parent=summary_node,
        critical=False
    )

    # Check coffee shop info is provided
    evaluator.add_custom_node(
        result=bool(num_with_coffee_info >= 2),
        id="summary_coffee_info",
        desc="Summary includes coffee shop name and rating for each apartment, or states if none found",
        parent=summary_node,
        critical=False
    )

    # Check format is comparative/tabular
    comparative_format = (has_any_ci(answer, ['summary', 'comparison', 'comparative']) or
                         answer.count('\n') > 10)
    evaluator.add_custom_node(
        result=bool(comparative_format),
        id="summary_format",
        desc="Summary is presented in a clear comparative format",
        parent=summary_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
