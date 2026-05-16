import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-1b8b3c"
TASK_DESCRIPTION = """
I'm planning to go camping at Grand Canyon Village tomorrow with my 2024 Subaru Outback. First, please go to AccuWeather and check the hourly weather forecast for Grand Canyon Village for tomorrow. I'm particularly interested in the temperature and wind speed at 2 PM to see if it's suitable for pitching a tent. Next, to ensure the car can handle the terrain and my loading needs, go to Car and Driver and find a detailed review of the 2024 Subaru Outback. I need you to delve into its 'Specs' page to find out its exact Ground Clearance in inches, and its Max Cargo Volume in cubic feet with the rear seats folded down. Also, please check what the first item listed under 'Highs' (pros) by the editors in the review summary is.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class WeatherAtTwoPM(BaseModel):
    """Weather details extracted from the answer for tomorrow 2 PM at Grand Canyon Village"""
    time_label: Optional[str] = None
    temperature_text: Optional[str] = None
    wind_text: Optional[str] = None


class VehicleSpecs(BaseModel):
    """Vehicle specs extracted from the answer for 2024 Subaru Outback from Car and Driver"""
    ground_clearance_text: Optional[str] = None
    cargo_volume_rear_folded_text: Optional[str] = None


class HighsInfo(BaseModel):
    """Highs first item extracted from the answer"""
    highs_first_item: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_weather_from_answer() -> str:
    return """
Extract the user's reported AccuWeather hourly forecast details for Grand Canyon Village for tomorrow at around 2 PM from the answer.

Return:
- time_label: whatever time expression the answer uses around 2 PM (e.g., "2 PM", "2:00 PM", "14:00"). If not present, set null.
- temperature_text: the temperature at that time exactly as written (include units or symbols if present).
- wind_text: the wind speed at that time exactly as written (include units if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_specs_from_answer() -> str:
    return """
From the answer, extract the Car and Driver 'Specs' details for the 2024 Subaru Outback:

- ground_clearance_text: the ground clearance in inches exactly as stated (include units if present).
- cargo_volume_rear_folded_text: the max cargo volume with the rear seats folded down in cubic feet exactly as stated (include units if present).

If any field is missing, set it to null.
"""


def prompt_extract_highs_first_item_from_answer() -> str:
    return """
From the answer, extract the first item listed under the editors' 'Highs' (pros) in the Car and Driver review summary for the 2024 Subaru Outback.

Return:
- highs_first_item: the first 'Highs' item text exactly as written. If not present, set it to null.
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


def looks_like_time_2pm(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    # Accept various formats: "2 pm", "2pm", "2 p.m.", "2:00 pm", "14:00", "14"
    patterns = [
        r'\b2\s*(p\.?m\.?)\b',
        r'\b2\s*:?\s*00\s*(p\.?m\.?)\b',
        r'\b2pm\b',
        r'\b2:00\s*pm\b',
        r'\b14:00\b',
        r'\b14\b'
    ]
    return any(re.search(p, t) for p in patterns)


def looks_like_temperature(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if contains a number and optional degree symbol or unit
    # e.g., "75°F", "24 C", "around 70", etc.
    if not contains_digits(text):
        return False
    return True


def looks_like_wind_speed(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept if contains a number and commonly used units or mentions 'wind'
    if not contains_digits(text):
        return False
    units_ok = has_any_ci(text, ['mph', 'km/h', 'kph', 'mi/h'])
    wind_word = ci_contains(text, 'wind')
    return units_ok or wind_word


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


def looks_like_inches(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['in', 'inch', 'inches', '"'])


def looks_like_cubic_feet(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['cu ft', 'cubic ft', 'cubic feet', 'ft³', 'ft^3'])


def mentions_rear_seats_folded(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['rear seats folded', 'seats folded', 'rear seat folded', 'with seats folded', 'with the rear seats folded'])


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
    weather_info = await evaluator.extract(
        prompt=prompt_extract_weather_from_answer(),
        template_class=WeatherAtTwoPM,
        extraction_name="weather_at_two_pm"
    )

    specs_info = await evaluator.extract(
        prompt=prompt_extract_specs_from_answer(),
        template_class=VehicleSpecs,
        extraction_name="vehicle_specs"
    )

    highs_info = await evaluator.extract(
        prompt=prompt_extract_highs_first_item_from_answer(),
        template_class=HighsInfo,
        extraction_name="highs_first_item"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 AccuWeather part
    accuweather_node = evaluator.add_sequential(
        id="accuweather_section",
        desc="AccuWeather hourly forecast for Grand Canyon Village - Tomorrow at 2 PM",
        parent=root,
        critical=False
    )

    # [Action Node] accuweather.com:F2:A2 - Navigate to AccuWeather hourly
    accuweather_action_ok = (has_any_ci(answer, ['accuweather']) and
                             (has_any_ci(answer, ['hourly']) or has_any_ci(answer, ['2 pm', '2pm', '2 p.m.', '14:00', '14'])))
    evaluator.add_custom_node(
        result=bool(accuweather_action_ok),
        id="accuweather_action_hourly",
        desc="[Action Node] accuweather.com:F2:A2 - Navigate to AccuWeather and open the Hourly tab for Grand Canyon Village",
        parent=accuweather_node,
        critical=False
    )

    # [Perception Node] accuweather.com:F2:P2 - Extract temperature and wind at 2 PM
    time_ok = looks_like_time_2pm(weather_info.time_label) or has_any_ci(answer, ['2 pm', '2pm', '2 p.m.', '14:00', '14'])
    temp_ok = looks_like_temperature(weather_info.temperature_text)
    wind_ok = looks_like_wind_speed(weather_info.wind_text)

    evaluator.add_custom_node(
        result=bool(time_ok and temp_ok and wind_ok),
        id="accuweather_perception_2pm_values",
        desc="[Perception Node] accuweather.com:F2:P2 - Extract temperature and wind speed for tomorrow at around 2 PM",
        parent=accuweather_node,
        critical=False
    )

    # Extra lenient check: mentions "tomorrow" in weather context (non-prefixed)
    mentions_tomorrow = has_any_ci(answer, ['tomorrow'])
    evaluator.add_custom_node(
        result=bool(mentions_tomorrow),
        id="accuweather_mentions_tomorrow",
        desc="Mentions the 'tomorrow' context for the 2 PM forecast",
        parent=accuweather_node,
        critical=False
    )

    # 3.2 Car and Driver part
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver review & specs for 2024 Subaru Outback",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search for model on Car and Driver
    caranddriver_action_ok = (has_any_ci(answer, ['car and driver']) and
                              has_any_ci(answer, ['2024 subaru outback']))
    evaluator.add_custom_node(
        result=bool(caranddriver_action_ok),
        id="caranddriver_action_search",
        desc="[Action Node] caranddriver.com:F2:A1 - Locate the 2024 Subaru Outback review on Car and Driver (via search or navigation)",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F2:P3 - Extract deep specs: ground clearance & cargo volume (rear seats folded)
    gc_num = extract_float(specs_info.ground_clearance_text)
    cv_num = extract_float(specs_info.cargo_volume_rear_folded_text)
    gc_units_ok = looks_like_inches(specs_info.ground_clearance_text)
    cv_units_ok = looks_like_cubic_feet(specs_info.cargo_volume_rear_folded_text)
    folded_context_ok = mentions_rear_seats_folded(answer)

    specs_ok = (gc_num is not None) and gc_units_ok and (cv_num is not None) and cv_units_ok and folded_context_ok

    evaluator.add_custom_node(
        result=bool(specs_ok),
        id="caranddriver_perception_specs",
        desc="[Perception Node] caranddriver.com:F2:P3 - Extract ground clearance (inches) and max cargo volume (cubic feet) with rear seats folded",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Identify first item under 'Highs'
    highs_text_ok = bool(highs_info and highs_info.highs_first_item and highs_info.highs_first_item.strip())
    highs_keyword_ok = has_any_ci(answer, ['highs'])
    evaluator.add_custom_node(
        result=bool(highs_text_ok and highs_keyword_ok),
        id="caranddriver_perception_highs",
        desc="[Perception Node] caranddriver.com:F1:P2 - Identify the first 'Highs' item from the editors' review summary",
        parent=caranddriver_node,
        critical=False
    )

    # Optional lenient node: Mentions 'Specs' or 'Specifications' page (non-prefixed)
    specs_mention_ok = has_any_ci(answer, ['specs', 'specifications'])
    evaluator.add_custom_node(
        result=bool(specs_mention_ok),
        id="caranddriver_mentions_specs",
        desc="Mentions visiting the 'Specs' or 'Specifications' section/page for details",
        parent=caranddriver_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()