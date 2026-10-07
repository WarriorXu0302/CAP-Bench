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
TASK_ID = "task-ed63b5"
TASK_DESCRIPTION = 'I’m planning to drive a Hyundai Ioniq 5 AWD from Griffith Observatory in Los Angeles to the Grand Canyon Visitor Center at the end of next month. I don’t fully trust the official range figures, so I need you to create a conservative charging plan for me.\n\nFirst, go to Car and Driver and find the tested **“75 mph highway range”** for this model (or the same-generation AWD version).  \nThen, go to YouTube and find a real-world winter/cold-weather range test video for this model, and note how many miles it achieved (use the **lowest tested value** mentioned in the video title or description).\n\nCompare these two numbers and use the smaller one as my **“safe range”**.\n\nFinally, plan the route in Google Maps and include **3 specific CCS charging stations** as stops along the way. Requirement: each leg of the trip (**start → stop 1 → stop 2 → stop 3 → destination**) must have a driving distance that does not exceed the “safe range.”\n\nOutput requirements:\n1. Car and Driver’s measured 75 mph range and the article link;  \n2. The winter tested range from the YouTube video, plus the video title and link;  \n3. The final “safe range” value;  \n4. The names of the 3 planned charging stations;  \n5. The driving distance (miles) for each of the 4 legs shown in Google Maps;  \n6. The final Google Maps route share link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CarAndDriverInfo(BaseModel):
    """Car and Driver 75 mph highway range data"""
    range_75mph_text: Optional[str] = None
    article_link: Optional[str] = None


class YouTubeWinterTest(BaseModel):
    """YouTube winter/cold-weather range test data"""
    winter_range_text: Optional[str] = None
    video_title: Optional[str] = None
    video_link: Optional[str] = None


class SafeRangeInfo(BaseModel):
    """Final safe range value"""
    safe_range_text: Optional[str] = None


class ChargingStations(BaseModel):
    """List of charging stations"""
    station_names: Optional[List[str]] = Field(default_factory=list)


class LegDistances(BaseModel):
    """Driving distances for each leg"""
    leg_distances: Optional[List[str]] = Field(default_factory=list)


class GoogleMapsRoute(BaseModel):
    """Google Maps route share link"""
    route_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_caranddriver_info() -> str:
    return """
Extract the Car and Driver 75 mph highway range test data for the Hyundai Ioniq 5 AWD from the answer.

Return:
- range_75mph_text: the 75 mph highway range value exactly as stated (include units if present).
- article_link: the URL to the Car and Driver article.

If any field is missing, set it to null.
"""


def prompt_extract_youtube_winter_test() -> str:
    return """
Extract the YouTube winter/cold-weather range test data for the Hyundai Ioniq 5 AWD from the answer.

Return:
- winter_range_text: the winter/cold-weather range value exactly as stated (include units if present).
- video_title: the title of the YouTube video.
- video_link: the URL to the YouTube video.

If any field is missing, set it to null.
"""


def prompt_extract_safe_range() -> str:
    return """
Extract the final "safe range" value from the answer.

Return:
- safe_range_text: the safe range value exactly as stated (include units if present).

If missing, set it to null.
"""


def prompt_extract_charging_stations() -> str:
    return """
Extract the names of the 3 CCS charging stations planned along the route from the answer.

Return:
- station_names: a list of 3 charging station names.

If missing or incomplete, return an empty list or partial list.
"""


def prompt_extract_leg_distances() -> str:
    return """
Extract the driving distances for each of the 4 legs of the trip from the answer.

Return:
- leg_distances: a list of 4 distance values as strings (include units if present).

If missing or incomplete, return an empty list or partial list.
"""


def prompt_extract_maps_route() -> str:
    return """
Extract the Google Maps route share link from the answer.

Return:
- route_link: the Google Maps route URL.

If missing, set it to null.
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


def looks_like_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and (ci_contains(text, 'http://') or ci_contains(text, 'https://'))


def looks_like_range_value(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and has_any_ci(text, ['mile', 'mi', 'km'])


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
    caranddriver_info = await evaluator.extract(
        prompt=prompt_extract_caranddriver_info(),
        template_class=CarAndDriverInfo,
        extraction_name="caranddriver_info"
    )

    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube_winter_test(),
        template_class=YouTubeWinterTest,
        extraction_name="youtube_winter_test"
    )

    safe_range_info = await evaluator.extract(
        prompt=prompt_extract_safe_range(),
        template_class=SafeRangeInfo,
        extraction_name="safe_range"
    )

    charging_stations = await evaluator.extract(
        prompt=prompt_extract_charging_stations(),
        template_class=ChargingStations,
        extraction_name="charging_stations"
    )

    leg_distances = await evaluator.extract(
        prompt=prompt_extract_leg_distances(),
        template_class=LegDistances,
        extraction_name="leg_distances"
    )

    maps_route = await evaluator.extract(
        prompt=prompt_extract_maps_route(),
        template_class=GoogleMapsRoute,
        extraction_name="maps_route"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver 75 mph highway range test for Hyundai Ioniq 5 AWD",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search for the model
    caranddriver_search_ok = (
        has_any_ci(answer, ['car and driver']) and
        has_any_ci(answer, ['ioniq 5', 'ioniq5']) and
        has_any_ci(answer, ['awd'])
    )
    evaluator.add_custom_node(
        result=bool(caranddriver_search_ok),
        id="caranddriver_search",
        desc="[Action Node] caranddriver.com:F2:A1 - Search for Hyundai Ioniq 5 AWD on Car and Driver",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Extract 75 mph highway range
    range_value_ok = looks_like_range_value(caranddriver_info.range_75mph_text)
    range_mentions_75mph = has_any_ci(answer, ['75 mph', '75mph', '75-mph'])
    article_link_ok = looks_like_url(caranddriver_info.article_link, 'caranddriver.com')

    evaluator.add_custom_node(
        result=bool(range_value_ok and range_mentions_75mph),
        id="caranddriver_extract_range",
        desc="[Perception Node] caranddriver.com:F1:P2 - Extract the 75 mph highway range value from the test data",
        parent=caranddriver_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(article_link_ok),
        id="caranddriver_article_link",
        desc="Provides the Car and Driver article link",
        parent=caranddriver_node,
        critical=False
    )

    # 3.2 YouTube section
    youtube_node = evaluator.add_sequential(
        id="youtube_section",
        desc="YouTube winter/cold-weather range test for Hyundai Ioniq 5 AWD",
        parent=root,
        critical=False
    )

    # [Action Node] youtube.com:F1:A22 - Click into video details
    youtube_video_ok = (
        has_any_ci(answer, ['youtube']) and
        has_any_ci(answer, ['ioniq 5', 'ioniq5'])
    )
    evaluator.add_custom_node(
        result=bool(youtube_video_ok),
        id="youtube_find_video",
        desc="[Action Node] youtube.com:F1:A22 - Find and access a Hyundai Ioniq 5 range test video",
        parent=youtube_node,
        critical=False
    )

    # [Perception Node] youtube.com:F1:P4 - Identify winter/cold-weather content
    winter_keywords = ['winter', 'cold', 'cold weather', 'cold-weather', 'freezing']
    winter_context_ok = has_any_ci(answer, winter_keywords)
    winter_range_ok = looks_like_range_value(youtube_info.winter_range_text)
    video_link_ok = looks_like_url(youtube_info.video_link, 'youtube.com') or looks_like_url(youtube_info.video_link, 'youtu.be')

    evaluator.add_custom_node(
        result=bool(winter_context_ok and winter_range_ok),
        id="youtube_extract_winter_range",
        desc="[Perception Node] youtube.com:F1:P4 - Identify winter/cold-weather range test and extract the lowest tested value",
        parent=youtube_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(video_link_ok),
        id="youtube_video_link",
        desc="Provides the YouTube video link",
        parent=youtube_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(youtube_info.video_title and len(youtube_info.video_title.strip()) > 0),
        id="youtube_video_title",
        desc="Provides the YouTube video title",
        parent=youtube_node,
        critical=False
    )

    # 3.3 Safe range calculation
    safe_range_node = evaluator.add_sequential(
        id="safe_range_section",
        desc="Calculate safe range by comparing Car and Driver and YouTube values",
        parent=root,
        critical=False
    )

    safe_range_value_ok = looks_like_range_value(safe_range_info.safe_range_text)
    safe_range_comparison_mentioned = has_any_ci(answer, ['smaller', 'lower', 'minimum', 'conservative', 'safe'])

    evaluator.add_custom_node(
        result=bool(safe_range_value_ok and safe_range_comparison_mentioned),
        id="safe_range_calculation",
        desc="Calculate and state the 'safe range' by taking the smaller of the two tested values",
        parent=safe_range_node,
        critical=False
    )

    # 3.4 Google Maps route planning
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps route planning with charging stations",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A5 - Route planning input
    maps_route_ok = (
        has_any_ci(answer, ['google maps', 'maps.google']) and
        has_any_ci(answer, ['griffith observatory', 'griffith']) and
        has_any_ci(answer, ['grand canyon'])
    )
    evaluator.add_custom_node(
        result=bool(maps_route_ok),
        id="maps_route_planning",
        desc="[Action Node] maps.google.com:F2:A5 - Plan route from Griffith Observatory to Grand Canyon Visitor Center in Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A20 - Search along route
    # [Action Node] maps.google.com:F2:A18 - Add waypoints
    stations_list = charging_stations.station_names if charging_stations.station_names else []
    has_three_stations = len(stations_list) == 3
    mentions_ccs = has_any_ci(answer, ['ccs', 'ccs charging', 'charging station'])

    evaluator.add_custom_node(
        result=bool(mentions_ccs),
        id="maps_search_charging",
        desc="[Action Node] maps.google.com:F2:A20 - Search for CCS charging stations along the route",
        parent=maps_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_three_stations),
        id="maps_add_waypoints",
        desc="[Action Node] maps.google.com:F2:A18 - Add 3 charging stations as waypoints to the route",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F11:P3 - Extract distances and verify constraints
    legs_list = leg_distances.leg_distances if leg_distances.leg_distances else []
    has_four_legs = len(legs_list) == 4

    evaluator.add_custom_node(
        result=bool(has_four_legs),
        id="maps_extract_distances",
        desc="[Perception Node] maps.google.com:F11:P3 - Extract driving distances for all 4 legs of the trip",
        parent=maps_node,
        critical=False
    )

    # Verify that each leg does not exceed safe range
    safe_range_float = extract_float(safe_range_info.safe_range_text)
    all_legs_within_safe_range = False
    if has_four_legs and safe_range_float is not None:
        all_legs_ok = True
        for leg_text in legs_list:
            leg_float = extract_float(leg_text)
            if leg_float is None or leg_float > safe_range_float:
                all_legs_ok = False
                break
        all_legs_within_safe_range = all_legs_ok

    evaluator.add_custom_node(
        result=bool(all_legs_within_safe_range),
        id="maps_distance_constraint",
        desc="Verify that each leg's distance does not exceed the safe range",
        parent=maps_node,
        critical=False
    )

    # Google Maps share link
    maps_link_ok = looks_like_url(maps_route.route_link, 'google.com/maps')
    evaluator.add_custom_node(
        result=bool(maps_link_ok),
        id="maps_share_link",
        desc="Provides the Google Maps route share link",
        parent=maps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
