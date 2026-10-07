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
TASK_ID = "task-74524f"
TASK_DESCRIPTION = 'As a plant photography enthusiast, I plan to visit Kew Gardens in London to photograph rare tropical Orchidaceae.\n\nFirst, on Kew Gardens\' official scientific database POWO (powo.science.kew.org), search for plants with the genus "Bulbophyllum". From the results, identify 3 species that are "Native to" "Madagascar" and have "Images". For these 3 plants, query Wikipedia to find their "Common name" (if none, mark as such) and whether they are "Epiphyte".\n\nNext, using Google Maps and starting from Kew Gardens, plan a walking tour route: first visit the "Princess of Wales Conservatory" (where such plants are typically displayed), then proceed to the nearby "The Hive" art installation. Finally, find a cafe within a 10-minute walk from "Kew Gardens Victoria Gate" with a rating above 4.5 for a break.\n\nOutput the scientific name, native origin, Wikipedia link, and epiphyte status for each of the 3 plants, along with the total distance of the walking route and the name of the cafe.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PlantInfo(BaseModel):
    """Information for a single plant species"""
    scientific_name: Optional[str] = None
    native_origin: Optional[str] = None
    wikipedia_link: Optional[str] = None
    common_name: Optional[str] = None
    epiphyte_status: Optional[str] = None


class PlantsList(BaseModel):
    """List of plant information extracted from the answer"""
    plants: List[PlantInfo] = Field(default_factory=list)


class RouteInfo(BaseModel):
    """Route and cafe information extracted from the answer"""
    total_distance: Optional[str] = None
    cafe_name: Optional[str] = None
    cafe_rating: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_plants_from_answer() -> str:
    return """
Extract information about the 3 Bulbophyllum species native to Madagascar from the answer.

For each plant, return:
- scientific_name: the full scientific name (genus + species)
- native_origin: the native origin information (should mention Madagascar)
- wikipedia_link: the Wikipedia URL for this species
- common_name: the common name if available, or indication if none exists
- epiphyte_status: whether the plant is an epiphyte (yes/no or descriptive text)

Return a list of up to 3 plants. If any field is missing, set it to null.
"""


def prompt_extract_route_from_answer() -> str:
    return """
Extract the walking route and cafe information from the answer.

Return:
- total_distance: the total distance of the walking route (include units)
- cafe_name: the name of the recommended cafe near Kew Gardens Victoria Gate
- cafe_rating: the rating of the cafe (if mentioned)

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


def looks_like_scientific_name(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain "Bulbophyllum" and another word (species epithet)
    return ci_contains(text, 'bulbophyllum') and len(text.split()) >= 2


def mentions_madagascar(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'madagascar')


def looks_like_wikipedia_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['wikipedia.org', 'en.wikipedia.org', 'wiki'])


def mentions_epiphyte(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['epiphyte', 'lithophyte', 'epiphytic'])


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    if not contains_digits(text):
        return False
    return has_any_ci(text, ['km', 'kilometer', 'metre', 'meter', 'mile', 'mi', 'm', 'km'])


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
    return 0 <= num <= 5


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
    plants_info = await evaluator.extract(
        prompt=prompt_extract_plants_from_answer(),
        template_class=PlantsList,
        extraction_name="plants_list"
    )

    route_info = await evaluator.extract(
        prompt=prompt_extract_route_from_answer(),
        template_class=RouteInfo,
        extraction_name="route_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 POWO database section
    powo_node = evaluator.add_sequential(
        id="powo_section",
        desc="POWO database search and filtering for Bulbophyllum species native to Madagascar",
        parent=root,
        critical=False
    )

    # [Action Node] powo.science.kew.org:F2:A1 - Search for genus Bulbophyllum
    powo_search_ok = has_any_ci(answer, ['powo', 'bulbophyllum'])
    evaluator.add_custom_node(
        result=bool(powo_search_ok),
        id="powo_action_search",
        desc="[Action Node] powo.science.kew.org:F2:A1 - Search for genus 'Bulbophyllum' on POWO",
        parent=powo_node,
        critical=False
    )

    # [Action Node] powo.science.kew.org:F3:A10 - Click into species detail pages
    has_species_details = False
    if plants_info and plants_info.plants:
        for plant in plants_info.plants:
            if plant.scientific_name and looks_like_scientific_name(plant.scientific_name):
                has_species_details = True
                break
    evaluator.add_custom_node(
        result=bool(has_species_details),
        id="powo_action_detail_click",
        desc="[Action Node] powo.science.kew.org:F3:A10 - Click into species detail pages to check native origin",
        parent=powo_node,
        critical=False
    )

    # [Perception Node] powo.science.kew.org:F3:P12 - Extract native origin (Madagascar)
    madagascar_count = 0
    if plants_info and plants_info.plants:
        for plant in plants_info.plants:
            if mentions_madagascar(plant.native_origin):
                madagascar_count += 1
    madagascar_ok = madagascar_count >= 3
    evaluator.add_custom_node(
        result=bool(madagascar_ok),
        id="powo_perception_madagascar",
        desc="[Perception Node] powo.science.kew.org:F3:P12 - Extract and verify native origin includes Madagascar for 3 species",
        parent=powo_node,
        critical=False
    )

    # Check for 3 species mentioned
    has_three_species = plants_info and len(plants_info.plants) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_species),
        id="powo_three_species",
        desc="Identified 3 Bulbophyllum species with required attributes",
        parent=powo_node,
        critical=False
    )

    # 3.2 Wikipedia section
    wikipedia_node = evaluator.add_sequential(
        id="wikipedia_section",
        desc="Wikipedia queries for common names and epiphyte status",
        parent=root,
        critical=False
    )

    # [Action Node] wikipedia.org:F1:A20 - Search and navigate to Wikipedia
    wiki_links_count = 0
    if plants_info and plants_info.plants:
        for plant in plants_info.plants:
            if looks_like_wikipedia_url(plant.wikipedia_link):
                wiki_links_count += 1
    wiki_navigation_ok = wiki_links_count >= 3
    evaluator.add_custom_node(
        result=bool(wiki_navigation_ok),
        id="wikipedia_action_navigate",
        desc="[Action Node] wikipedia.org:F1:A20 - Navigate to Wikipedia pages for the 3 species",
        parent=wikipedia_node,
        critical=False
    )

    # [Perception Node] wikipedia.org:F2:P2 - Extract epiphyte information
    epiphyte_count = 0
    if plants_info and plants_info.plants:
        for plant in plants_info.plants:
            if mentions_epiphyte(plant.epiphyte_status):
                epiphyte_count += 1
    epiphyte_ok = epiphyte_count >= 3
    evaluator.add_custom_node(
        result=bool(epiphyte_ok),
        id="wikipedia_perception_epiphyte",
        desc="[Perception Node] wikipedia.org:F2:P2 - Extract whether species are epiphytes from Wikipedia content",
        parent=wikipedia_node,
        critical=False
    )

    # Check for common names (optional, lenient)
    has_common_names = False
    if plants_info and plants_info.plants:
        for plant in plants_info.plants:
            if plant.common_name and plant.common_name.strip():
                has_common_names = True
                break
    evaluator.add_custom_node(
        result=bool(has_common_names),
        id="wikipedia_common_names",
        desc="Extracted common names or noted their absence",
        parent=wikipedia_node,
        critical=False
    )

    # 3.3 Google Maps section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps route planning and cafe search",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A7 - Select walking mode
    walking_mentioned = has_any_ci(answer, ['walk', 'walking', 'on foot'])
    evaluator.add_custom_node(
        result=bool(walking_mentioned),
        id="maps_action_walking_mode",
        desc="[Action Node] maps.google.com:F2:A7 - Select walking as transportation mode",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F2:A18 - Multi-point route planning
    princess_mentioned = has_any_ci(answer, ['princess of wales conservatory'])
    hive_mentioned = has_any_ci(answer, ['the hive', 'hive'])
    multi_stop_ok = princess_mentioned and hive_mentioned
    evaluator.add_custom_node(
        result=bool(multi_stop_ok),
        id="maps_action_multi_point_route",
        desc="[Action Node] maps.google.com:F2:A18 - Plan multi-point route (Princess of Wales Conservatory → The Hive)",
        parent=maps_node,
        critical=False
    )

    # Check total distance provided
    distance_ok = looks_like_distance(route_info.total_distance)
    evaluator.add_custom_node(
        result=bool(distance_ok),
        id="maps_route_distance",
        desc="Total walking route distance provided with units",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F11:A20 - Search for nearby cafe
    victoria_gate_mentioned = has_any_ci(answer, ['victoria gate', 'kew gardens victoria gate'])
    cafe_mentioned = has_any_ci(answer, ['cafe', 'coffee'])
    nearby_search_ok = victoria_gate_mentioned and cafe_mentioned
    evaluator.add_custom_node(
        result=bool(nearby_search_ok),
        id="maps_action_nearby_cafe",
        desc="[Action Node] maps.google.com:F11:A20 - Search for cafe near Kew Gardens Victoria Gate within 10-minute walk",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F1:P1 - Filter by rating above 4.5
    cafe_name_ok = route_info.cafe_name and route_info.cafe_name.strip()
    rating_num = extract_float(route_info.cafe_rating) if route_info.cafe_rating else None
    rating_mentioned = has_any_ci(answer, ['rating', '4.5', '4.6', '4.7', '4.8', '4.9', '5.0'])
    rating_ok = (rating_num and rating_num >= 4.5) or (cafe_name_ok and rating_mentioned)
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="maps_perception_rating_filter",
        desc="[Perception Node] maps.google.com:F1:P1 - Filter and select cafe with rating above 4.5",
        parent=maps_node,
        critical=False
    )

    # Check cafe name provided
    evaluator.add_custom_node(
        result=bool(cafe_name_ok),
        id="maps_cafe_name",
        desc="Cafe name provided",
        parent=maps_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
