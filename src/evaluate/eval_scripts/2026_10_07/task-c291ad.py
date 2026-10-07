import asyncio
import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field

from cap_eval.evaluator import Evaluator, AggregationStrategy
from cap_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-c291ad"
TASK_DESCRIPTION = 'I’m new to camping and plan to go to Austin, TX next March for a music festival. First, go to Eventbrite and find a music-related event taking place in mid-March (between the 10th and 20th), and note the exact date. Then go to AccuWeather, open Austin’s **Monthly** forecast for the corresponding month, and check the forecast **Low** temperature and whether rain is expected (Rain/Showers icon or text) for the dates you selected. If forecast data for that month is not currently available, use the nearest visible month data on that page (e.g., **Hist. Avg**) and explicitly note this in the output.  \n\nFinally, go to Amazon to choose gear: based on the lowest temperature you found, pick a sleeping bag with a **Temperature Rating** at least **10°F lower** than that low (e.g., if the low is 40°F, choose 30°F or lower). If rain is forecast for those days, find a tent with **“Waterproof”** in the title; otherwise, find one with **“Ventilated.”** Both items must have ratings above **4.5 stars**.  \n\nOutput required: event name, event date, AccuWeather low temperature and weather condition, selected sleeping bag name/temperature rating/rating, selected tent name/key feature/rating, and the detail-page links from all three websites.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    """Event details extracted from Eventbrite"""
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    event_link: Optional[str] = None


class WeatherInfo(BaseModel):
    """Weather details extracted from AccuWeather Monthly forecast"""
    low_temperature: Optional[str] = None
    weather_condition: Optional[str] = None
    data_source_note: Optional[str] = None


class SleepingBagInfo(BaseModel):
    """Sleeping bag details extracted from Amazon"""
    bag_name: Optional[str] = None
    temperature_rating: Optional[str] = None
    rating: Optional[str] = None
    bag_link: Optional[str] = None


class TentInfo(BaseModel):
    """Tent details extracted from Amazon"""
    tent_name: Optional[str] = None
    key_feature: Optional[str] = None
    rating: Optional[str] = None
    tent_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_from_answer() -> str:
    return """
Extract the music event details from Eventbrite that the user found:

Return:
- event_name: the name of the music event exactly as stated
- event_date: the date of the event exactly as stated
- event_link: the Eventbrite detail page URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_weather_from_answer() -> str:
    return """
Extract the AccuWeather Monthly forecast details for Austin that the user found:

Return:
- low_temperature: the low temperature for the event date exactly as stated (include units)
- weather_condition: the weather condition (rain/showers or not) exactly as stated
- data_source_note: any note about using historical data or forecast data if mentioned

If any field is missing, set it to null.
"""


def prompt_extract_sleeping_bag_from_answer() -> str:
    return """
Extract the sleeping bag details from Amazon that the user selected:

Return:
- bag_name: the product name exactly as stated
- temperature_rating: the temperature rating exactly as stated (include units)
- rating: the star rating exactly as stated
- bag_link: the Amazon detail page URL if provided

If any field is missing, set it to null.
"""


def prompt_extract_tent_from_answer() -> str:
    return """
Extract the tent details from Amazon that the user selected:

Return:
- tent_name: the product name exactly as stated
- key_feature: whether it's waterproof or ventilated as stated
- rating: the star rating exactly as stated
- tent_link: the Amazon detail page URL if provided

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
    m = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def parse_date_march(date_str: Optional[str]) -> Optional[int]:
    """Extract day number from a date string, expecting March dates"""
    if not date_str:
        return None
    match = re.search(r'\b(\d{1,2})\b', date_str)
    if match:
        try:
            day = int(match.group(1))
            return day if 1 <= day <= 31 else None
        except Exception:
            return None
    return None


def is_march_mid_range(date_str: Optional[str]) -> bool:
    """Check if date falls between March 10-20"""
    day = parse_date_march(date_str)
    if day is None:
        return False
    return 10 <= day <= 20


def looks_like_temperature(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['f', '°', 'degree', 'fahrenheit', 'celsius'])


def mentions_rain(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['rain', 'shower', 'precipitation'])


def extract_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:star|★)', text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return extract_float(text)


def is_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower() and ('http://' in text.lower() or 'https://' in text.lower())


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
    event_info = await evaluator.extract(
        prompt=prompt_extract_event_from_answer(),
        template_class=EventInfo,
        extraction_name="event_info"
    )

    weather_info = await evaluator.extract(
        prompt=prompt_extract_weather_from_answer(),
        template_class=WeatherInfo,
        extraction_name="weather_info"
    )

    sleeping_bag_info = await evaluator.extract(
        prompt=prompt_extract_sleeping_bag_from_answer(),
        template_class=SleepingBagInfo,
        extraction_name="sleeping_bag_info"
    )

    tent_info = await evaluator.extract(
        prompt=prompt_extract_tent_from_answer(),
        template_class=TentInfo,
        extraction_name="tent_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Eventbrite section
    eventbrite_node = evaluator.add_sequential(
        id="eventbrite_section",
        desc="Eventbrite music event search for mid-March in Austin",
        parent=root,
        critical=False
    )

    # [Action Node] eventbrite.com:F1:A1 - Date range filtering
    date_in_range = is_march_mid_range(event_info.event_date)
    evaluator.add_custom_node(
        result=bool(date_in_range),
        id="eventbrite_date_filter",
        desc="[Action Node] eventbrite.com:F1:A1 - Filter events to find one between March 10-20",
        parent=eventbrite_node,
        critical=False
    )

    # [Perception Node] eventbrite.com:F1:P3 - List content understanding (music category)
    event_name_ok = bool(event_info.event_name and event_info.event_name.strip())
    mentions_music = has_any_ci(answer, ['music', 'concert', 'festival', 'band', 'performance'])
    evaluator.add_custom_node(
        result=bool(event_name_ok and mentions_music),
        id="eventbrite_music_category",
        desc="[Perception Node] eventbrite.com:F1:P3 - Identify music-related event from list",
        parent=eventbrite_node,
        critical=False
    )

    # Event link validation
    event_link_ok = is_valid_url(event_info.event_link, 'eventbrite')
    evaluator.add_custom_node(
        result=bool(event_link_ok),
        id="eventbrite_link_provided",
        desc="Eventbrite detail page link provided",
        parent=eventbrite_node,
        critical=False
    )

    # 3.2 AccuWeather section
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather Monthly forecast for Austin",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F1:A1 - Location search
    mentions_austin = has_any_ci(answer, ['austin'])
    mentions_accuweather = has_any_ci(answer, ['accuweather'])
    evaluator.add_custom_node(
        result=bool(mentions_austin and mentions_accuweather),
        id="accuweather_location_search",
        desc="[Action Node] accuweather.com:F1:A1 - Search for Austin, TX location",
        parent=accuweather_node,
        critical=False
    )

    # [Action Node] accuweather.com:F2:A2 - Navigate to Monthly tab
    mentions_monthly = has_any_ci(answer, ['monthly', 'month'])
    evaluator.add_custom_node(
        result=bool(mentions_monthly),
        id="accuweather_monthly_tab",
        desc="[Action Node] accuweather.com:F2:A2 - Navigate to Monthly forecast tab",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F2:P3 - Extract calendar/table data
    low_temp_ok = looks_like_temperature(weather_info.low_temperature)
    weather_cond_ok = bool(weather_info.weather_condition and weather_info.weather_condition.strip())
    evaluator.add_custom_node(
        result=bool(low_temp_ok and weather_cond_ok),
        id="accuweather_extract_data",
        desc="[Perception Node] accuweather.com:F2:P3 - Extract low temperature and weather condition for specific date",
        parent=accuweather_node,
        critical=False
    )

    # Weather link validation
    weather_link_ok = is_valid_url(answer, 'accuweather')
    evaluator.add_custom_node(
        result=bool(weather_link_ok),
        id="accuweather_link_provided",
        desc="AccuWeather page link or reference provided",
        parent=accuweather_node,
        critical=False
    )

    # 3.3 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon gear selection (sleeping bag and tent)",
        parent=root,
        critical=False
    )

    # 3.3.1 Sleeping bag subsection
    sleeping_bag_node = evaluator.add_sequential(
        id="sleeping_bag_subsection",
        desc="Sleeping bag selection based on temperature requirement",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A28 - Scroll/paginate to find suitable product
    bag_name_ok = bool(sleeping_bag_info.bag_name and sleeping_bag_info.bag_name.strip())
    evaluator.add_custom_node(
        result=bool(bag_name_ok),
        id="amazon_sleeping_bag_search",
        desc="[Action Node] Amazon:F3:A28 - Search and navigate through listings to find sleeping bag meeting criteria",
        parent=sleeping_bag_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A24 - Expand details to confirm temperature rating
    bag_temp_rating = extract_float(sleeping_bag_info.temperature_rating)
    weather_low = extract_float(weather_info.low_temperature)
    temp_rating_ok = False
    if bag_temp_rating is not None and weather_low is not None:
        temp_rating_ok = bag_temp_rating <= (weather_low - 10)

    evaluator.add_custom_node(
        result=bool(bag_temp_rating is not None),
        id="amazon_sleeping_bag_temp_detail",
        desc="[Action Node] Amazon:F5:A24 - Extract temperature rating from product details",
        parent=sleeping_bag_node,
        critical=False
    )

    # Verify temperature constraint
    evaluator.add_custom_node(
        result=bool(temp_rating_ok),
        id="sleeping_bag_temp_constraint",
        desc="Sleeping bag temperature rating is at least 10°F lower than weather low",
        parent=sleeping_bag_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Multi-select filtering (4.5+ stars)
    bag_rating = extract_rating(sleeping_bag_info.rating)
    bag_rating_ok = bag_rating is not None and bag_rating >= 4.5
    evaluator.add_custom_node(
        result=bool(bag_rating_ok),
        id="amazon_sleeping_bag_rating",
        desc="[Action Node] Amazon:F3:A15 - Filter or verify sleeping bag has rating above 4.5 stars",
        parent=sleeping_bag_node,
        critical=False
    )

    # Sleeping bag link validation
    bag_link_ok = is_valid_url(sleeping_bag_info.bag_link, 'amazon')
    evaluator.add_custom_node(
        result=bool(bag_link_ok),
        id="sleeping_bag_link_provided",
        desc="Amazon sleeping bag detail page link provided",
        parent=sleeping_bag_node,
        critical=False
    )

    # 3.3.2 Tent subsection
    tent_node = evaluator.add_sequential(
        id="tent_subsection",
        desc="Tent selection based on weather condition",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Content filtering/keyword matching
    rain_expected = mentions_rain(weather_info.weather_condition)
    tent_name_ok = bool(tent_info.tent_name and tent_info.tent_name.strip())

    if rain_expected:
        tent_keyword_ok = ci_contains(tent_info.tent_name, 'waterproof') or ci_contains(tent_info.key_feature, 'waterproof')
    else:
        tent_keyword_ok = ci_contains(tent_info.tent_name, 'ventilated') or ci_contains(tent_info.key_feature, 'ventilated')

    evaluator.add_custom_node(
        result=bool(tent_name_ok and tent_keyword_ok),
        id="amazon_tent_keyword_match",
        desc="[Perception Node] Amazon:F3:P1 - Find tent with 'Waterproof' (if rain) or 'Ventilated' (if no rain) in title",
        parent=tent_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Multi-select filtering (4.5+ stars)
    tent_rating = extract_rating(tent_info.rating)
    tent_rating_ok = tent_rating is not None and tent_rating >= 4.5
    evaluator.add_custom_node(
        result=bool(tent_rating_ok),
        id="amazon_tent_rating",
        desc="[Action Node] Amazon:F3:A15 - Filter or verify tent has rating above 4.5 stars",
        parent=tent_node,
        critical=False
    )

    # Tent link validation
    tent_link_ok = is_valid_url(tent_info.tent_link, 'amazon')
    evaluator.add_custom_node(
        result=bool(tent_link_ok),
        id="tent_link_provided",
        desc="Amazon tent detail page link provided",
        parent=tent_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
