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
TASK_ID = "task-e6941e"
TASK_DESCRIPTION = 'I am a site selection specialist for a health management service company. Our company plans to establish new chronic disease management clinics in the United States in 2026, primarily targeting patients with diabetes and heart disease.\n\nFirst, access the CDC WONDER database to query diabetes and heart disease mortality data for all US states from 2022-2024. Identify the top 3 states with the highest mortality rates (indicating a heavy disease burden and a large potential customer base).\n\nNext, search US News Health for endocrinologist and cardiologist resources in these 3 states. For each of these states, identify the top 2 major cities with a relatively lower physician density (indicating comparatively underserved medical resources and less competition).\n\nFinally, on Zillow, search for commercial real estate rentals in these cities. For each city, find one office/commercial space suitable for medical use, with a monthly rent between $3,000-$6,000 and an area between 1,500-3,000 square feet.\n\nOutput:\n*   For each state, the diabetes mortality rate and heart disease mortality rate (per 100,000 population), along with the link to the CDC data page.\n*   For each recommended city, the number of endocrinologists, the number of cardiologists, and the link to the US News search results page.\n*   For each commercial property, the address, monthly rent, area, and the link to the Zillow listing.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class StateData(BaseModel):
    """Mortality data for a single state"""
    state_name: Optional[str] = None
    diabetes_mortality_rate: Optional[str] = None
    heart_disease_mortality_rate: Optional[str] = None
    cdc_link: Optional[str] = None


class CityData(BaseModel):
    """Physician data for a single city"""
    city_name: Optional[str] = None
    state_name: Optional[str] = None
    endocrinologist_count: Optional[str] = None
    cardiologist_count: Optional[str] = None
    usnews_link: Optional[str] = None


class PropertyData(BaseModel):
    """Commercial property data"""
    city_name: Optional[str] = None
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    area_sqft: Optional[str] = None
    zillow_link: Optional[str] = None


class ExtractedData(BaseModel):
    """All extracted data from the answer"""
    states: List[StateData] = Field(default_factory=list)
    cities: List[CityData] = Field(default_factory=list)
    properties: List[PropertyData] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_data() -> str:
    return """
Extract all the information reported in the answer about states, cities, and commercial properties.

For states (up to 3):
- state_name: the state name
- diabetes_mortality_rate: the diabetes mortality rate per 100,000 population exactly as stated
- heart_disease_mortality_rate: the heart disease mortality rate per 100,000 population exactly as stated
- cdc_link: the CDC WONDER data page link

For cities (up to 6, ideally 2 per state):
- city_name: the city name
- state_name: the state the city belongs to
- endocrinologist_count: the number of endocrinologists exactly as stated
- cardiologist_count: the number of cardiologists exactly as stated
- usnews_link: the US News Health search results link

For properties (up to 6, one per city):
- city_name: the city the property is in
- address: the property address
- monthly_rent: the monthly rent exactly as stated (include currency and units)
- area_sqft: the area in square feet exactly as stated (include units)
- zillow_link: the Zillow listing link

If any field is missing, set it to null. Return empty lists if no data is found.
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


def looks_like_mortality_rate(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 10.0 <= num <= 500.0


def looks_like_physician_count(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    return num is not None and num >= 0


def looks_like_rent(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 3000 <= num <= 6000


def looks_like_sqft(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 1500 <= num <= 3000


def is_valid_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return ci_contains(text, domain) and (text.startswith('http://') or text.startswith('https://'))


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_data(),
        template_class=ExtractedData,
        extraction_name="extracted_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 CDC WONDER section
    cdc_node = evaluator.add_sequential(
        id="cdc_wonder_section",
        desc="CDC WONDER database query for state-level diabetes and heart disease mortality (2022-2024)",
        parent=root,
        critical=False
    )

    # [Action Node] cdc.gov:F5:A9 - Multi-dimensional query parameters
    cdc_mentions = has_any_ci(answer, ['cdc wonder', 'cdc.gov', 'wonder.cdc.gov'])
    year_mentions = has_any_ci(answer, ['2022', '2023', '2024'])
    disease_mentions = has_any_ci(answer, ['diabetes', 'heart disease', 'heart'])
    cdc_action_ok = cdc_mentions and year_mentions and disease_mentions

    evaluator.add_custom_node(
        result=bool(cdc_action_ok),
        id="cdc_query_setup",
        desc="[Action Node] cdc.gov:F5:A9 - Set up CDC WONDER query with year range (2022-2024), disease types (diabetes, heart disease), and geographic region parameters",
        parent=cdc_node,
        critical=False
    )

    # [Perception Node] cdc.gov:F5:P11 - Identify top 3 states with highest mortality rates
    has_three_states = len(extracted.states) >= 3
    states_have_rates = all(
        looks_like_mortality_rate(s.diabetes_mortality_rate) and
        looks_like_mortality_rate(s.heart_disease_mortality_rate)
        for s in extracted.states[:3]
    ) if has_three_states else False

    evaluator.add_custom_node(
        result=bool(has_three_states and states_have_rates),
        id="cdc_identify_top_states",
        desc="[Perception Node] cdc.gov:F5:P11 - Identify the 3 states with highest mortality rates from CDC WONDER data visualization or tables",
        parent=cdc_node,
        critical=False
    )

    # CDC links validation
    cdc_links_ok = all(
        is_valid_url(s.cdc_link, 'cdc.gov')
        for s in extracted.states[:3]
    ) if has_three_states else False

    evaluator.add_custom_node(
        result=bool(cdc_links_ok),
        id="cdc_links_provided",
        desc="Valid CDC WONDER data page links are provided for the identified states",
        parent=cdc_node,
        critical=False
    )

    # 3.2 US News Health section
    usnews_node = evaluator.add_sequential(
        id="usnews_health_section",
        desc="US News Health search for endocrinologist and cardiologist resources in selected states",
        parent=root,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A5 - Advanced multi-field search
    usnews_mentions = has_any_ci(answer, ['us news', 'usnews', 'health.usnews.com'])
    specialty_mentions = has_any_ci(answer, ['endocrinologist', 'cardiologist'])
    usnews_action_ok = usnews_mentions and specialty_mentions

    evaluator.add_custom_node(
        result=bool(usnews_action_ok),
        id="usnews_advanced_search",
        desc="[Action Node] health.usnews.com:F2:A5 - Perform advanced search on US News Health with specialty type and geographic location filters",
        parent=usnews_node,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A6 - Multi-condition filtering
    has_cities = len(extracted.cities) >= 6
    cities_have_states = all(
        c.state_name is not None and c.state_name.strip()
        for c in extracted.cities[:6]
    ) if has_cities else False

    evaluator.add_custom_node(
        result=bool(has_cities and cities_have_states),
        id="usnews_filter_cities",
        desc="[Action Node] health.usnews.com:F2:A6 - Use More Filters to filter by geographic location, specialty type, and identify cities with lower physician density",
        parent=usnews_node,
        critical=False
    )

    # [Action Node] health.usnews.com:F2:A7 - Pagination browsing
    multiple_cities_ok = len(extracted.cities) >= 4
    evaluator.add_custom_node(
        result=bool(multiple_cities_ok),
        id="usnews_pagination",
        desc="[Action Node] health.usnews.com:F2:A7 - Browse through multiple pages of physician search results to collect data across different cities",
        parent=usnews_node,
        critical=False
    )

    # [Perception Node] health.usnews.com:F2:P6 - Extract list information
    cities_have_counts = all(
        looks_like_physician_count(c.endocrinologist_count) and
        looks_like_physician_count(c.cardiologist_count)
        for c in extracted.cities[:6]
    ) if has_cities else False

    evaluator.add_custom_node(
        result=bool(cities_have_counts),
        id="usnews_extract_physician_counts",
        desc="[Perception Node] health.usnews.com:F2:P6 - Extract and compare physician counts from search result lists for multiple cities",
        parent=usnews_node,
        critical=False
    )

    # US News links validation
    usnews_links_ok = all(
        is_valid_url(c.usnews_link, 'usnews.com')
        for c in extracted.cities[:6]
    ) if has_cities else False

    evaluator.add_custom_node(
        result=bool(usnews_links_ok),
        id="usnews_links_provided",
        desc="Valid US News Health search result links are provided for the recommended cities",
        parent=usnews_node,
        critical=False
    )

    # 3.3 Zillow section
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow commercial real estate rental search in selected cities",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A3 - Switch to rental listings
    zillow_mentions = has_any_ci(answer, ['zillow'])
    rental_mentions = has_any_ci(answer, ['rent', 'rental', 'for rent'])
    zillow_rental_ok = zillow_mentions and rental_mentions

    evaluator.add_custom_node(
        result=bool(zillow_rental_ok),
        id="zillow_switch_rental",
        desc="[Action Node] zillow.com:F1:A3 - Switch from For Sale to For Rent to view commercial rental listings",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A8 - Multi-selection for property types
    commercial_mentions = has_any_ci(answer, ['office', 'commercial', 'medical'])
    evaluator.add_custom_node(
        result=bool(commercial_mentions),
        id="zillow_select_property_types",
        desc="[Action Node] zillow.com:F2:A8 - Select multiple commercial property types (Office, Commercial) in Home Type filter",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A12 - Drag price range sliders
    has_properties = len(extracted.properties) >= 6
    properties_in_rent_range = all(
        looks_like_rent(p.monthly_rent)
        for p in extracted.properties[:6]
    ) if has_properties else False

    evaluator.add_custom_node(
        result=bool(properties_in_rent_range),
        id="zillow_set_price_range",
        desc="[Action Node] zillow.com:F2:A12 - Drag Price double sliders to set rent range ($3,000-$6,000)",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F2:A26 - Scroll to load more listings
    multiple_properties_ok = len(extracted.properties) >= 4
    evaluator.add_custom_node(
        result=bool(multiple_properties_ok),
        id="zillow_scroll_load",
        desc="[Action Node] zillow.com:F2:A26 - Scroll down to load more commercial property listings",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A18 - Click into property details
    properties_have_details = all(
        p.address and p.monthly_rent and p.area_sqft
        for p in extracted.properties[:6]
    ) if has_properties else False

    evaluator.add_custom_node(
        result=bool(properties_have_details),
        id="zillow_click_details",
        desc="[Action Node] zillow.com:F4:A18 - Click property cards to enter detail pages for complete information",
        parent=zillow_node,
        critical=False
    )

    # [Perception Node] zillow.com:F4:P18 - Understand property details
    properties_in_area_range = all(
        looks_like_sqft(p.area_sqft)
        for p in extracted.properties[:6]
    ) if has_properties else False

    evaluator.add_custom_node(
        result=bool(properties_in_area_range),
        id="zillow_understand_features",
        desc="[Perception Node] zillow.com:F4:P18 - Understand area, usage, and facility features from Facts and features section (1,500-3,000 sqft, suitable for medical use)",
        parent=zillow_node,
        critical=False
    )

    # Zillow links validation
    zillow_links_ok = all(
        is_valid_url(p.zillow_link, 'zillow.com')
        for p in extracted.properties[:6]
    ) if has_properties else False

    evaluator.add_custom_node(
        result=bool(zillow_links_ok),
        id="zillow_links_provided",
        desc="Valid Zillow listing links are provided for the recommended commercial properties",
        parent=zillow_node,
        critical=False
    )

    # 3.4 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="completeness_checks",
        desc="Overall task completeness verification",
        parent=root,
        critical=False
    )

    # Check if the information flow is complete
    has_all_states = len(extracted.states) >= 3
    has_all_cities = len(extracted.cities) >= 6
    has_all_properties = len(extracted.properties) >= 6

    evaluator.add_custom_node(
        result=bool(has_all_states),
        id="complete_state_data",
        desc="All 3 states with mortality data are provided",
        parent=completeness_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_all_cities),
        id="complete_city_data",
        desc="All 6 cities (2 per state) with physician data are provided",
        parent=completeness_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(has_all_properties),
        id="complete_property_data",
        desc="All 6 commercial properties (1 per city) are provided",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
