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
TASK_ID = "task-a4b7d7"
TASK_DESCRIPTION = "I am considering investing in a single-family home in Seattle for long-term rental.\n\nFirst, search on Zillow for single-family homes for sale in Seattle, priced between $400,000 and $600,000. Filter for 3 eligible properties and record each property's address, asking price, square footage, Zestimate valuation, and corresponding ZIP code. Then, view the price history chart for each property and record the price change trend over the past 5 years (starting price, current price, and percentage change/gain/loss).\n\nNext, use the QuickFacts tool on Census.gov to retrieve demographic data for the communities corresponding to these 3 ZIP codes. Focus on median household income, population count, and educational attainment (percentage of residents with a bachelor's degree or higher). Afterward, visit EPA.gov to check the air quality for these 3 ZIP codes. Record the current AQI index and its rating (e.g., Good/Moderate/Unhealthy for Sensitive Groups, etc.).\n\nFinally, go to Gapminder Tools to examine the economic development trends of the Seattle metropolitan area over the past 10 years. Switch to the 'Trends' line chart, select Washington state, and view the historical change curves for GDP per capita and life expectancy. Record the values for 2015 and 2024 for comparison.\n\n**Output:**\n*   For each property: address, asking price, square footage, Zestimate valuation, ZIP code, and Zillow detail page link;\n*   For each property: 5-year price history (starting price, current price, percentage change/gain/loss);\n*   For each ZIP code community: median household income, population count, percentage of residents with a bachelor's degree or higher, and Census.gov search results page link;\n*   For each ZIP code: current AQI index, air quality rating, and EPA.gov search results page link;\n*   For Washington state: GDP per capita and life expectancy values for 2015 and 2024, and Gapminder chart link.\n\nFinally, based on all the data above, recommend the single best property for investment and explain the reasoning (comprehensively considering property appreciation potential, community income level, educational attainment, air quality, and regional economic growth trends)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PropertyInfo(BaseModel):
    """Information for a single property"""
    address: Optional[str] = None
    asking_price: Optional[str] = None
    square_footage: Optional[str] = None
    zestimate: Optional[str] = None
    zip_code: Optional[str] = None
    zillow_link: Optional[str] = None


class PriceHistory(BaseModel):
    """Price history for a single property"""
    starting_price: Optional[str] = None
    current_price: Optional[str] = None
    percentage_change: Optional[str] = None


class CensusData(BaseModel):
    """Census data for a ZIP code"""
    zip_code: Optional[str] = None
    median_household_income: Optional[str] = None
    population: Optional[str] = None
    bachelors_percentage: Optional[str] = None
    census_link: Optional[str] = None


class AirQualityData(BaseModel):
    """Air quality data for a ZIP code"""
    zip_code: Optional[str] = None
    aqi_index: Optional[str] = None
    air_quality_rating: Optional[str] = None
    epa_link: Optional[str] = None


class GapminderData(BaseModel):
    """Gapminder economic data for Washington state"""
    gdp_per_capita_2015: Optional[str] = None
    gdp_per_capita_2024: Optional[str] = None
    life_expectancy_2015: Optional[str] = None
    life_expectancy_2024: Optional[str] = None
    gapminder_link: Optional[str] = None


class InvestmentRecommendation(BaseModel):
    """Investment recommendation"""
    recommended_property: Optional[str] = None
    reasoning: Optional[str] = None


class ExtractedData(BaseModel):
    """All extracted data from the answer"""
    properties: List[PropertyInfo] = Field(default_factory=list)
    price_histories: List[PriceHistory] = Field(default_factory=list)
    census_data: List[CensusData] = Field(default_factory=list)
    air_quality_data: List[AirQualityData] = Field(default_factory=list)
    gapminder_data: Optional[GapminderData] = None
    recommendation: Optional[InvestmentRecommendation] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_data() -> str:
    return """
Extract all information from the answer related to the Seattle property investment task.

Return:
- properties: a list of up to 3 properties, each with address, asking_price, square_footage, zestimate, zip_code, and zillow_link
- price_histories: a list of up to 3 price histories, each with starting_price, current_price, and percentage_change
- census_data: a list of up to 3 census records, each with zip_code, median_household_income, population, bachelors_percentage, and census_link
- air_quality_data: a list of up to 3 air quality records, each with zip_code, aqi_index, air_quality_rating, and epa_link
- gapminder_data: Washington state data with gdp_per_capita_2015, gdp_per_capita_2024, life_expectancy_2015, life_expectancy_2024, and gapminder_link
- recommendation: the recommended_property address and reasoning

Set fields to null if not present in the answer.
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
    m = re.findall(r'(\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0].replace(',', ''))
    except Exception:
        return None


def is_seattle_address(address: Optional[str]) -> bool:
    if not address:
        return False
    return ci_contains(address, 'seattle') or ci_contains(address, 'wa')


def is_price_in_range(price_text: Optional[str]) -> bool:
    if not price_text:
        return False
    price = extract_float(price_text)
    if price is None:
        return False
    return 400000 <= price <= 600000


def has_zillow_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'zillow.com')


def has_census_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'census.gov')


def has_epa_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'epa.gov')


def has_gapminder_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'gapminder.org')


def is_valid_zip(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    return bool(re.search(r'\d{5}', zip_code))


def has_percentage(text: Optional[str]) -> bool:
    if not text:
        return False
    return '%' in text or ci_contains(text, 'percent')


def is_aqi_rating(rating: Optional[str]) -> bool:
    if not rating:
        return False
    valid_ratings = ['good', 'moderate', 'unhealthy for sensitive groups', 'unhealthy', 'very unhealthy', 'hazardous']
    return any(ci_contains(rating, r) for r in valid_ratings)


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
    data = await evaluator.extract(
        prompt=prompt_extract_all_data(),
        template_class=ExtractedData,
        extraction_name="all_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Zillow property search and details
    zillow_node = evaluator.add_sequential(
        id="zillow_section",
        desc="Zillow property search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F1:A35 - Search box input filtering
    search_seattle = has_any_ci(answer, ['zillow', 'seattle'])
    evaluator.add_custom_node(
        result=bool(search_seattle),
        id="zillow_search_seattle",
        desc="[Action Node] zillow.com:F1:A35 - Search for properties in Seattle on Zillow",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:A3 - Property type dropdown selection
    for_sale_check = has_any_ci(answer, ['for sale', 'sale', 'asking price'])
    evaluator.add_custom_node(
        result=bool(for_sale_check),
        id="zillow_for_sale",
        desc="[Action Node] zillow.com:F1:A3 - Filter for 'For Sale' properties (not rentals)",
        parent=zillow_node,
        critical=False
    )

    # Check if 3 properties with valid data
    has_three_properties = len(data.properties) >= 3
    properties_have_addresses = sum(1 for p in data.properties if p.address and is_seattle_address(p.address)) >= 3

    evaluator.add_custom_node(
        result=bool(has_three_properties and properties_have_addresses),
        id="zillow_three_properties",
        desc="Extract information for 3 eligible properties in Seattle",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:P3 - Card status marker recognition
    properties_in_price_range = sum(1 for p in data.properties if is_price_in_range(p.asking_price)) >= 3
    evaluator.add_custom_node(
        result=bool(properties_in_price_range),
        id="zillow_price_filter",
        desc="[Perception Node] zillow.com:F1:P3 - Identify properties in $400k-$600k price range",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F1:A26 - List scroll loading
    evaluator.add_custom_node(
        result=bool(has_three_properties),
        id="zillow_scroll_results",
        desc="[Action Node] zillow.com:F1:A26 - Browse multiple search results to find 3 properties",
        parent=zillow_node,
        critical=False
    )

    # [Action Node] zillow.com:F4:A18 - Click card to enter details
    properties_have_zestimate = sum(1 for p in data.properties if p.zestimate and contains_digits(p.zestimate)) >= 3
    evaluator.add_custom_node(
        result=bool(properties_have_zestimate),
        id="zillow_property_details",
        desc="[Action Node] zillow.com:F4:A18 - Navigate to property detail pages to extract Zestimate",
        parent=zillow_node,
        critical=False
    )

    # Verify all required property fields
    properties_complete = sum(1 for p in data.properties if all([
        p.address,
        p.asking_price and contains_digits(p.asking_price),
        p.square_footage and contains_digits(p.square_footage),
        p.zestimate,
        p.zip_code and is_valid_zip(p.zip_code)
    ])) >= 3

    evaluator.add_custom_node(
        result=bool(properties_complete),
        id="zillow_complete_info",
        desc="All 3 properties have address, asking price, square footage, Zestimate, and ZIP code",
        parent=zillow_node,
        critical=False
    )

    # Zillow links present
    zillow_links = sum(1 for p in data.properties if has_zillow_link(p.zillow_link)) >= 3
    evaluator.add_custom_node(
        result=bool(zillow_links),
        id="zillow_links",
        desc="Zillow detail page links provided for properties",
        parent=zillow_node,
        critical=False
    )

    # 3.2 Zillow price history
    price_history_node = evaluator.add_sequential(
        id="price_history_section",
        desc="Zillow price history analysis for each property",
        parent=root,
        critical=False
    )

    # [Action Node] zillow.com:F5:A6 - Tab switching
    price_history_mention = has_any_ci(answer, ['price history', 'price change', 'historical'])
    evaluator.add_custom_node(
        result=bool(price_history_mention),
        id="zillow_price_history_tab",
        desc="[Action Node] zillow.com:F5:A6 - Switch to Price History tab",
        parent=price_history_node,
        critical=False
    )

    # [Action Node] zillow.com:F5:A28 - Chart hover to view values
    has_three_histories = len(data.price_histories) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_histories),
        id="zillow_hover_chart",
        desc="[Action Node] zillow.com:F5:A28 - Hover over chart to extract data points",
        parent=price_history_node,
        critical=False
    )

    # [Perception Node] zillow.com:F5:P13 - Chart value reading
    histories_have_prices = sum(1 for h in data.price_histories if all([
        h.starting_price and contains_digits(h.starting_price),
        h.current_price and contains_digits(h.current_price)
    ])) >= 3

    evaluator.add_custom_node(
        result=bool(histories_have_prices),
        id="zillow_read_prices",
        desc="[Perception Node] zillow.com:F5:P13 - Extract specific starting and current price values",
        parent=price_history_node,
        critical=False
    )

    # [Perception Node] zillow.com:F5:P14 - Trend judgment
    histories_have_percentage = sum(1 for h in data.price_histories if h.percentage_change and has_percentage(h.percentage_change)) >= 3
    evaluator.add_custom_node(
        result=bool(histories_have_percentage),
        id="zillow_price_trend",
        desc="[Perception Node] zillow.com:F5:P14 - Calculate percentage change (gain/loss) over 5 years",
        parent=price_history_node,
        critical=False
    )

    # 3.3 Census.gov demographic data
    census_node = evaluator.add_sequential(
        id="census_section",
        desc="Census.gov QuickFacts demographic data for ZIP codes",
        parent=root,
        critical=False
    )

    # [Action Node] census.gov:F3:A13 - QuickFacts text input
    census_mention = has_any_ci(answer, ['census', 'quickfacts'])
    evaluator.add_custom_node(
        result=bool(census_mention),
        id="census_search",
        desc="[Action Node] census.gov:F3:A13 - Use QuickFacts tool to search by ZIP code",
        parent=census_node,
        critical=False
    )

    # [Perception Node] census.gov:F3:P6 - Table data understanding
    has_three_census = len(data.census_data) >= 3
    census_complete = sum(1 for c in data.census_data if all([
        c.zip_code and is_valid_zip(c.zip_code),
        c.median_household_income and contains_digits(c.median_household_income),
        c.population and contains_digits(c.population),
        c.bachelors_percentage and has_percentage(c.bachelors_percentage)
    ])) >= 3

    evaluator.add_custom_node(
        result=bool(census_complete),
        id="census_extract_data",
        desc="[Perception Node] census.gov:F3:P6 - Extract median income, population, and bachelor's degree percentage",
        parent=census_node,
        critical=False
    )

    census_links = sum(1 for c in data.census_data if has_census_link(c.census_link)) >= 3
    evaluator.add_custom_node(
        result=bool(census_links),
        id="census_links",
        desc="Census.gov search results page links provided",
        parent=census_node,
        critical=False
    )

    # 3.4 EPA.gov air quality data
    epa_node = evaluator.add_sequential(
        id="epa_section",
        desc="EPA.gov air quality data for ZIP codes",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F4:A13 - Air quality search input
    epa_mention = has_any_ci(answer, ['epa', 'air quality', 'aqi'])
    evaluator.add_custom_node(
        result=bool(epa_mention),
        id="epa_search",
        desc="[Action Node] epa.gov:F4:A13 - Search for air quality by ZIP code",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F4:P7 - Status label recognition
    has_three_aqi = len(data.air_quality_data) >= 3
    aqi_complete = sum(1 for a in data.air_quality_data if all([
        a.zip_code and is_valid_zip(a.zip_code),
        a.aqi_index and contains_digits(a.aqi_index),
        a.air_quality_rating and is_aqi_rating(a.air_quality_rating)
    ])) >= 3

    evaluator.add_custom_node(
        result=bool(aqi_complete),
        id="epa_extract_aqi",
        desc="[Perception Node] epa.gov:F4:P7 - Extract AQI index and rating (Good/Moderate/etc.)",
        parent=epa_node,
        critical=False
    )

    epa_links = sum(1 for a in data.air_quality_data if has_epa_link(a.epa_link)) >= 3
    evaluator.add_custom_node(
        result=bool(epa_links),
        id="epa_links",
        desc="EPA.gov search results page links provided",
        parent=epa_node,
        critical=False
    )

    # 3.5 Gapminder economic trends
    gapminder_node = evaluator.add_sequential(
        id="gapminder_section",
        desc="Gapminder economic development trends for Washington state",
        parent=root,
        critical=False
    )

    gapminder_mention = has_any_ci(answer, ['gapminder'])

    # [Action Node] gapminder.org:F4:A7 - Chart type tab switching
    trends_mention = has_any_ci(answer, ['trends', 'line chart', 'historical'])
    evaluator.add_custom_node(
        result=bool(gapminder_mention and trends_mention),
        id="gapminder_trends_tab",
        desc="[Action Node] gapminder.org:F4:A7 - Switch to Trends line chart view",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F4:A3 - Country/region multi-select
    washington_mention = has_any_ci(answer, ['washington', 'wa'])
    evaluator.add_custom_node(
        result=bool(washington_mention),
        id="gapminder_select_washington",
        desc="[Action Node] gapminder.org:F4:A3 - Select Washington state in the region selector",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F4:A1 - Timeline slider drag
    has_2015_2024 = has_any_ci(answer, ['2015']) and has_any_ci(answer, ['2024'])
    evaluator.add_custom_node(
        result=bool(has_2015_2024),
        id="gapminder_timeline",
        desc="[Action Node] gapminder.org:F4:A1 - Drag timeline slider to view 2015 and 2024 data",
        parent=gapminder_node,
        critical=False
    )

    # [Action Node] gapminder.org:F4:A6 - Chart hover to view values
    gm = data.gapminder_data
    gapminder_complete = gm and all([
        gm.gdp_per_capita_2015 and contains_digits(gm.gdp_per_capita_2015),
        gm.gdp_per_capita_2024 and contains_digits(gm.gdp_per_capita_2024),
        gm.life_expectancy_2015 and contains_digits(gm.life_expectancy_2015),
        gm.life_expectancy_2024 and contains_digits(gm.life_expectancy_2024)
    ])

    evaluator.add_custom_node(
        result=bool(gapminder_complete),
        id="gapminder_extract_values",
        desc="[Action Node] gapminder.org:F4:A6 - Hover over chart to extract GDP per capita and life expectancy values",
        parent=gapminder_node,
        critical=False
    )

    # [Perception Node] gapminder.org:F4:P1 - Trend judgment
    gdp_mention = has_any_ci(answer, ['gdp', 'economic growth'])
    life_expectancy_mention = has_any_ci(answer, ['life expectancy'])
    evaluator.add_custom_node(
        result=bool(gdp_mention and life_expectancy_mention and gapminder_complete),
        id="gapminder_trend_judgment",
        desc="[Perception Node] gapminder.org:F4:P1 - Identify economic growth trends over 10 years",
        parent=gapminder_node,
        critical=False
    )

    # [Perception Node] gapminder.org:F4:P2 - Multi-series data comparison
    has_comparison = gm and has_any_ci(answer, ['comparison', 'compare', 'change'])
    evaluator.add_custom_node(
        result=bool(has_comparison and gapminder_complete),
        id="gapminder_comparison",
        desc="[Perception Node] gapminder.org:F4:P2 - Compare 2015 and 2024 values across metrics",
        parent=gapminder_node,
        critical=False
    )

    gapminder_link = gm and has_gapminder_link(gm.gapminder_link)
    evaluator.add_custom_node(
        result=bool(gapminder_link),
        id="gapminder_link",
        desc="Gapminder chart link provided",
        parent=gapminder_node,
        critical=False
    )

    # 3.6 Investment recommendation
    recommendation_node = evaluator.add_sequential(
        id="recommendation_section",
        desc="Investment recommendation based on comprehensive analysis",
        parent=root,
        critical=False
    )

    rec = data.recommendation
    has_recommendation = rec and rec.recommended_property and rec.reasoning
    evaluator.add_custom_node(
        result=bool(has_recommendation),
        id="has_recommendation",
        desc="Provides a specific property recommendation with reasoning",
        parent=recommendation_node,
        critical=False
    )

    if has_recommendation and rec:
        # Check if reasoning mentions multiple factors
        reasoning_factors = [
            has_any_ci(rec.reasoning, ['appreciation', 'price', 'value']),
            has_any_ci(rec.reasoning, ['income', 'household']),
            has_any_ci(rec.reasoning, ['education', 'bachelor']),
            has_any_ci(rec.reasoning, ['air quality', 'aqi']),
            has_any_ci(rec.reasoning, ['economic', 'gdp', 'growth'])
        ]
        comprehensive_reasoning = sum(reasoning_factors) >= 3
    else:
        comprehensive_reasoning = False

    evaluator.add_custom_node(
        result=bool(comprehensive_reasoning),
        id="comprehensive_reasoning",
        desc="Recommendation considers multiple factors (appreciation, income, education, air quality, economic trends)",
        parent=recommendation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
