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
TASK_ID = "task-328b3d"
TASK_DESCRIPTION = 'I am planning a "Game vs. Reality" pilgrimage video for "Like a Dragon: Infinite Wealth," focusing on Honolulu, Hawaii.\n\nFirst, navigate to IGN and search for "Real Life Locations" or a similar guide for the game. Identify a list that maps in-game establishments/landmarks to their real-world counterparts.\n\nFrom this list, select four real-world commercial establishments (restaurants, shops, or bars, excluding generic parks or streets). Each selected location must have a Google Maps rating of 4.0 stars or higher to ensure their quality and appeal.\n\nFor each of these four selected locations, search for their real-world names on Google Maps. Verify their specific "price level" (e.g., $, $$) and their current "operating status" (e.g., Open or Closed).\n\nOutput the following for each location: In-game Name, Real-world Counterpart Name, Real-world Address, Google Maps Rating, Price Level, Operating Status, and the respective links to the IGN guide page and the Google Maps detail page.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class LocationInfo(BaseModel):
    """Information for a single location extracted from the answer"""
    in_game_name: Optional[str] = None
    real_world_name: Optional[str] = None
    real_world_address: Optional[str] = None
    google_maps_rating: Optional[str] = None
    price_level: Optional[str] = None
    operating_status: Optional[str] = None
    ign_guide_link: Optional[str] = None
    google_maps_link: Optional[str] = None


class ExtractedLocations(BaseModel):
    """All locations extracted from the answer"""
    locations: List[LocationInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_locations_from_answer() -> str:
    return """
Extract all location information reported by the user from the answer for the "Like a Dragon: Infinite Wealth" pilgrimage video.

For each location, extract:
- in_game_name: the name used in the game
- real_world_name: the real-world counterpart name
- real_world_address: the physical address
- google_maps_rating: the rating value exactly as stated (include units/format if present)
- price_level: the price level exactly as stated (e.g., "$", "$$", "$$$")
- operating_status: the current operating status exactly as stated (e.g., "Open", "Closed", "Permanently closed")
- ign_guide_link: the URL to the IGN guide page
- google_maps_link: the URL to the Google Maps detail page

Return a list of all locations found. If any field is missing for a location, set it to null.
Also provide total_count: the total number of locations extracted.
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


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Accept ratings in reasonable range (e.g., 0-5)
    return 0 <= num <= 5


def looks_like_price_level(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept $, $$, $$$, $$$$ or similar patterns
    return bool(re.search(r'\$+', text))


def looks_like_operating_status(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept common status keywords
    return has_any_ci(text, ['open', 'closed', 'permanently', 'temporarily'])


def looks_like_url(text: Optional[str], domain: Optional[str] = None) -> bool:
    if not text:
        return False
    url_pattern = r'https?://'
    if not re.search(url_pattern, text, re.IGNORECASE):
        return False
    if domain:
        return domain.lower() in text.lower()
    return True


def is_commercial_establishment(in_game_name: Optional[str], real_world_name: Optional[str]) -> bool:
    """Check if the location appears to be a commercial establishment (restaurant, shop, bar)"""
    combined = f"{in_game_name or ''} {real_world_name or ''}".lower()
    # Exclude generic locations
    exclude_keywords = ['park', 'street', 'avenue', 'road', 'beach', 'plaza', 'square']
    if any(keyword in combined for keyword in exclude_keywords):
        return False
    # Include commercial keywords
    include_keywords = ['restaurant', 'bar', 'shop', 'store', 'cafe', 'grill', 'diner', 'market', 'boutique']
    return any(keyword in combined for keyword in include_keywords) or len(combined.strip()) > 0


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
    Restrict evaluator.verify to at most one usage (not used here).
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_locations_from_answer(),
        template_class=ExtractedLocations,
        extraction_name="extracted_locations"
    )

    locations = extracted.locations if extracted and extracted.locations else []
    total_count = len(locations)

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IGN search and navigation section
    ign_section = evaluator.add_sequential(
        id="ign_section",
        desc="IGN guide search and content extraction for Like a Dragon: Infinite Wealth",
        parent=root,
        critical=False
    )

    # [Action Node] ign.com:F1:A4 - Search results browsing
    ign_search_ok = (has_any_ci(answer, ['ign']) and
                     has_any_ci(answer, ['like a dragon', 'infinite wealth']) and
                     has_any_ci(answer, ['real life', 'location', 'guide']))
    evaluator.add_custom_node(
        result=bool(ign_search_ok),
        id="ign_search_action",
        desc="[Action Node] ign.com:F1:A4 - Search for 'Real Life Locations' guide on IGN for Like a Dragon: Infinite Wealth",
        parent=ign_section,
        critical=False
    )

    # [Perception Node] ign.com:F1:P1 - Content type identification
    ign_guide_link_ok = any(
        loc.ign_guide_link and looks_like_url(loc.ign_guide_link, 'ign.com')
        for loc in locations
    ) if locations else has_any_ci(answer, ['ign.com'])
    evaluator.add_custom_node(
        result=bool(ign_guide_link_ok),
        id="ign_content_type",
        desc="[Perception Node] ign.com:F1:P1 - Identify the correct guide/article type from search results",
        parent=ign_section,
        critical=False
    )

    # [Action Node] ign.com:F1:A11 - Scrolling to view long content
    has_multiple_locations = total_count >= 4
    evaluator.add_custom_node(
        result=bool(has_multiple_locations),
        id="ign_scroll_action",
        desc="[Action Node] ign.com:F1:A11 - Scroll through the guide page to view the complete location list",
        parent=ign_section,
        critical=False
    )

    # [Perception Node] ign.com:F3:P4 - Content understanding and mapping extraction
    has_mapping_pairs = total_count > 0 and all(
        loc.in_game_name and loc.real_world_name
        for loc in locations
    )
    evaluator.add_custom_node(
        result=bool(has_mapping_pairs),
        id="ign_mapping_extraction",
        desc="[Perception Node] ign.com:F3:P4 - Extract in-game to real-world location mappings",
        parent=ign_section,
        critical=False
    )

    # 3.2 Google Maps search and verification section
    gmaps_section = evaluator.add_sequential(
        id="gmaps_section",
        desc="Google Maps search and detail verification for selected locations",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search
    gmaps_search_ok = any(
        loc.google_maps_link and looks_like_url(loc.google_maps_link, 'google.com/maps')
        for loc in locations
    ) if locations else has_any_ci(answer, ['google maps'])
    evaluator.add_custom_node(
        result=bool(gmaps_search_ok),
        id="gmaps_search_action",
        desc="[Action Node] maps.google.com:F1:A2 - Search for real-world location names on Google Maps",
        parent=gmaps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F5:A8 - Click to enter detail page
    has_detail_info = any(
        loc.price_level or loc.operating_status
        for loc in locations
    )
    evaluator.add_custom_node(
        result=bool(has_detail_info),
        id="gmaps_detail_action",
        desc="[Action Node] maps.google.com:F5:A8 - Enter location detail page to view price level and operating status",
        parent=gmaps_section,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P3 - Rating extraction and filtering
    ratings_valid = []
    for loc in locations:
        if loc.google_maps_rating and looks_like_rating(loc.google_maps_rating):
            rating_val = extract_float(loc.google_maps_rating)
            if rating_val and rating_val >= 4.0:
                ratings_valid.append(True)
            else:
                ratings_valid.append(False)
        else:
            ratings_valid.append(False)

    rating_filtering_ok = len(ratings_valid) >= 4 and all(ratings_valid[:4])
    evaluator.add_custom_node(
        result=bool(rating_filtering_ok),
        id="gmaps_rating_perception",
        desc="[Perception Node] maps.google.com:F5:P3 - Extract and verify ratings are 4.0 or higher",
        parent=gmaps_section,
        critical=False
    )

    # [Action Node] maps.google.com:F5:A6 - Tab switching for price level
    price_levels_found = [
        loc.price_level for loc in locations if loc.price_level and looks_like_price_level(loc.price_level)
    ]
    price_level_ok = len(price_levels_found) >= 4
    evaluator.add_custom_node(
        result=bool(price_level_ok),
        id="gmaps_price_level_action",
        desc="[Action Node] maps.google.com:F5:A6 - Switch tabs to find price level information",
        parent=gmaps_section,
        critical=False
    )

    # [Perception Node] maps.google.com:F5:P15 - Operating status perception
    operating_statuses_found = [
        loc.operating_status for loc in locations if loc.operating_status and looks_like_operating_status(loc.operating_status)
    ]
    operating_status_ok = len(operating_statuses_found) >= 4
    evaluator.add_custom_node(
        result=bool(operating_status_ok),
        id="gmaps_operating_status_perception",
        desc="[Perception Node] maps.google.com:F5:P15 - Extract current operating status (Open/Closed)",
        parent=gmaps_section,
        critical=False
    )

    # 3.3 Overall completeness checks (non-prefixed)
    completeness_section = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall task completeness validation",
        parent=root,
        critical=False
    )

    # Check if four locations are provided
    four_locations_ok = total_count >= 4
    evaluator.add_custom_node(
        result=bool(four_locations_ok),
        id="four_locations_provided",
        desc="Four locations are provided as required",
        parent=completeness_section,
        critical=False
    )

    # Check if locations are commercial establishments
    commercial_count = sum(
        1 for loc in locations[:4] if is_commercial_establishment(loc.in_game_name, loc.real_world_name)
    )
    commercial_ok = commercial_count >= 4
    evaluator.add_custom_node(
        result=bool(commercial_ok),
        id="commercial_establishments",
        desc="Selected locations are commercial establishments (restaurants, shops, bars)",
        parent=completeness_section,
        critical=False
    )

    # Check if all required fields are present for each location
    complete_entries = 0
    for loc in locations[:4]:
        if all([
            loc.in_game_name,
            loc.real_world_name,
            loc.real_world_address,
            loc.google_maps_rating and looks_like_rating(loc.google_maps_rating),
            loc.price_level and looks_like_price_level(loc.price_level),
            loc.operating_status and looks_like_operating_status(loc.operating_status),
            loc.ign_guide_link and looks_like_url(loc.ign_guide_link),
            loc.google_maps_link and looks_like_url(loc.google_maps_link)
        ]):
            complete_entries += 1

    all_fields_ok = complete_entries >= 4
    evaluator.add_custom_node(
        result=bool(all_fields_ok),
        id="all_required_fields",
        desc="All required fields are provided for each location",
        parent=completeness_section,
        critical=False
    )

    # Check Honolulu/Hawaii context
    honolulu_context = has_any_ci(answer, ['honolulu', 'hawaii'])
    evaluator.add_custom_node(
        result=bool(honolulu_context),
        id="honolulu_hawaii_context",
        desc="Locations are in Honolulu, Hawaii context",
        parent=completeness_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
