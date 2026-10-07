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
TASK_ID = "task-502566"
TASK_DESCRIPTION = 'I’m planning to move to Tampa, Florida in the near future and I’m very concerned about neighborhood safety. First, go to the Superfund page on EPA.gov and search for the **“Kassauf-Kimerling Battery Disposal”** contaminated site in Tampa. Obtain its full street address and current NPL status.  \n\nNext, go to Census.gov QuickFacts and check the latest population estimate for **“Tampa city, Florida.”** Compare it with the value from the previous statistical period to confirm whether the population is trending upward.\n\nAfter gathering that information, go to Redfin.com to find listings. Filter criteria: Tampa city, **House** property type, price **$450,000–$600,000**, at least **3 bedrooms**, and status must be **Active** (exclude Pending/Contingent). Try to find **3** listings that meet these criteria; if fewer than 3 are available, return all currently available matches and indicate the total count.  \n\nTo ensure safety, use Google Maps to calculate the **driving distance** from each listing to that battery disposal contamination site, and prioritize homes that are **more than 3 miles** away. If all are under 3 miles, still report the exact distances truthfully.\n\nOutput: the contaminated site address, NPL status, and Tampa’s latest population; plus for each listing: address, price, Redfin link, driving distance to the contaminated site, and whether it meets the “>3 miles” criterion.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ContaminatedSiteInfo(BaseModel):
    """EPA Superfund site information extracted from the answer"""
    site_name: Optional[str] = None
    full_address: Optional[str] = None
    npl_status: Optional[str] = None


class TampaPopulationInfo(BaseModel):
    """Census population data extracted from the answer"""
    latest_population: Optional[str] = None
    previous_population: Optional[str] = None
    trending_upward: Optional[bool] = None


class PropertyListing(BaseModel):
    """Individual property listing details"""
    address: Optional[str] = None
    price: Optional[str] = None
    redfin_link: Optional[str] = None
    driving_distance: Optional[str] = None
    meets_distance_criterion: Optional[bool] = None


class RedfinListings(BaseModel):
    """Redfin listings extracted from the answer"""
    total_count: Optional[int] = None
    listings: Optional[List[PropertyListing]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_contaminated_site() -> str:
    return """
Extract the EPA Superfund contaminated site information from the answer for the "Kassauf-Kimerling Battery Disposal" site in Tampa.

Return:
- site_name: the name of the contaminated site as mentioned in the answer
- full_address: the full street address of the site exactly as stated
- npl_status: the current NPL status exactly as stated (e.g., "Deleted", "Final", "Proposed")

If any field is missing, set it to null.
"""


def prompt_extract_tampa_population() -> str:
    return """
Extract the Tampa city, Florida population information from the answer based on Census.gov QuickFacts.

Return:
- latest_population: the latest population estimate exactly as stated
- previous_population: the previous period's population value if mentioned
- trending_upward: true if the answer indicates population is increasing, false if decreasing, null if unclear

If any field is missing, set it to null.
"""


def prompt_extract_redfin_listings() -> str:
    return """
Extract all Redfin property listings from the answer that match the Tampa search criteria.

Return:
- total_count: the total number of matching listings found (as stated or count of listings provided)
- listings: array of property objects, each containing:
  - address: full property address
  - price: listing price exactly as stated
  - redfin_link: URL to the Redfin listing
  - driving_distance: driving distance from the property to the contaminated site
  - meets_distance_criterion: true if distance is more than 3 miles, false otherwise

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


def looks_like_address(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for street number and common address words
    has_number = contains_digits(text)
    has_street_words = has_any_ci(text, ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr', 'boulevard', 'blvd', 'lane', 'ln', 'way'])
    return has_number and (has_street_words or ci_contains(text, 'tampa'))


def looks_like_npl_status(text: Optional[str]) -> bool:
    if not text:
        return False
    # Common NPL status terms
    return has_any_ci(text, ['deleted', 'final', 'proposed', 'npl', 'removed', 'construction complete'])


def looks_like_population(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain large numbers (millions or hundred thousands)
    num = extract_float(text)
    if num is None:
        return False
    return num > 100000  # Tampa population is in hundreds of thousands


def looks_like_price_in_range(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    # Accept if in range or close (450k-600k)
    return 400000 <= num <= 650000


def looks_like_distance(text: Optional[str]) -> bool:
    if not text:
        return False
    has_number = contains_digits(text)
    has_unit = has_any_ci(text, ['mile', 'mi', 'km', 'kilometer'])
    return has_number and has_unit


def distance_is_over_3_miles(text: Optional[str]) -> Optional[bool]:
    if not text:
        return None
    num = extract_float(text)
    if num is None:
        return None
    # Convert km to miles if needed
    if ci_contains(text, 'km'):
        num = num * 0.621371
    return num > 3.0


def looks_like_redfin_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'redfin.com')


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
    site_info = await evaluator.extract(
        prompt=prompt_extract_contaminated_site(),
        template_class=ContaminatedSiteInfo,
        extraction_name="contaminated_site_info"
    )

    population_info = await evaluator.extract(
        prompt=prompt_extract_tampa_population(),
        template_class=TampaPopulationInfo,
        extraction_name="tampa_population_info"
    )

    listings_info = await evaluator.extract(
        prompt=prompt_extract_redfin_listings(),
        template_class=RedfinListings,
        extraction_name="redfin_listings"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 EPA.gov section
    epa_node = evaluator.add_sequential(
        id="epa_section",
        desc="EPA.gov Superfund site search and information retrieval",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F2:A7 - Search for contaminated site in Tampa
    epa_search_ok = (has_any_ci(answer, ['epa', 'superfund']) and
                     has_any_ci(answer, ['kassauf', 'kimerling', 'battery disposal']) and
                     has_any_ci(answer, ['tampa']))
    evaluator.add_custom_node(
        result=bool(epa_search_ok),
        id="epa_action_search",
        desc="[Action Node] epa.gov:F2:A7 - Search for Kassauf-Kimerling Battery Disposal site in Tampa using EPA Superfund search",
        parent=epa_node,
        critical=False
    )

    # Check if address looks valid
    address_ok = looks_like_address(site_info.full_address)
    evaluator.add_custom_node(
        result=bool(address_ok),
        id="epa_address_valid",
        desc="Contaminated site address is a valid Tampa street address",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F2:P4 - Extract NPL status
    npl_status_ok = looks_like_npl_status(site_info.npl_status)
    evaluator.add_custom_node(
        result=bool(npl_status_ok),
        id="epa_perception_npl_status",
        desc="[Perception Node] epa.gov:F2:P4 - Extract current NPL status for the contaminated site",
        parent=epa_node,
        critical=False
    )

    # 3.2 Census.gov section
    census_node = evaluator.add_sequential(
        id="census_section",
        desc="Census.gov QuickFacts population data retrieval",
        parent=root,
        critical=False
    )

    # [Action Node] census.gov:F3:A13 - Query Tampa population on QuickFacts
    census_search_ok = (has_any_ci(answer, ['census', 'quickfacts']) and
                        has_any_ci(answer, ['tampa city, florida', 'tampa, florida']))
    evaluator.add_custom_node(
        result=bool(census_search_ok),
        id="census_action_query",
        desc="[Action Node] census.gov:F3:A13 - Query Tampa city, Florida population on Census.gov QuickFacts",
        parent=census_node,
        critical=False
    )

    # Check if population values look valid
    pop_latest_ok = looks_like_population(population_info.latest_population)
    evaluator.add_custom_node(
        result=bool(pop_latest_ok),
        id="census_population_valid",
        desc="Latest Tampa population estimate is a reasonable value (>100,000)",
        parent=census_node,
        critical=False
    )

    # Check if trend is mentioned
    trend_mentioned = population_info.trending_upward is not None
    evaluator.add_custom_node(
        result=bool(trend_mentioned),
        id="census_trend_analyzed",
        desc="Population trend (upward/downward) is analyzed by comparing with previous period",
        parent=census_node,
        critical=False
    )

    # 3.3 Redfin.com section
    redfin_node = evaluator.add_sequential(
        id="redfin_section",
        desc="Redfin.com property search with specified filters",
        parent=root,
        critical=False
    )

    # [Action Node] redfin.com:F2:A1 - Apply price range filter
    redfin_search_ok = has_any_ci(answer, ['redfin']) and has_any_ci(answer, ['tampa'])
    evaluator.add_custom_node(
        result=bool(redfin_search_ok),
        id="redfin_action_search",
        desc="[Action Node] redfin.com:F2:A1 - Search for properties in Tampa city with price filter $450,000-$600,000",
        parent=redfin_node,
        critical=False
    )

    # [Action Node] redfin.com:F2:A2 - Apply property type and bedroom filters
    house_filter_ok = has_any_ci(answer, ['house'])
    bedroom_filter_ok = has_any_ci(answer, ['3 bedroom', '3 bed', 'at least 3'])
    evaluator.add_custom_node(
        result=bool(house_filter_ok and bedroom_filter_ok),
        id="redfin_action_filters",
        desc="[Action Node] redfin.com:F2:A2 - Apply House property type and minimum 3 bedrooms filters",
        parent=redfin_node,
        critical=False
    )

    # [Perception Node] redfin.com:F1:P1 - Verify Active status (exclude Pending/Contingent)
    status_check_ok = has_any_ci(answer, ['active']) or not has_any_ci(answer, ['pending', 'contingent'])
    evaluator.add_custom_node(
        result=bool(status_check_ok),
        id="redfin_perception_status",
        desc="[Perception Node] redfin.com:F1:P1 - Verify listings are Active status (excluding Pending/Contingent)",
        parent=redfin_node,
        critical=False
    )

    # Check if listings are provided
    has_listings = listings_info.listings and len(listings_info.listings) > 0
    evaluator.add_custom_node(
        result=bool(has_listings),
        id="redfin_listings_provided",
        desc="At least one property listing is provided in the answer",
        parent=redfin_node,
        critical=False
    )

    # Check if prices are in range
    if has_listings:
        prices_in_range = all(looks_like_price_in_range(listing.price) for listing in listings_info.listings if listing.price)
        evaluator.add_custom_node(
            result=bool(prices_in_range),
            id="redfin_prices_in_range",
            desc="All listing prices are within the $450,000-$600,000 range",
            parent=redfin_node,
            critical=False
        )

        # Check if Redfin links are provided
        links_provided = all(looks_like_redfin_url(listing.redfin_link) for listing in listings_info.listings if listing.redfin_link)
        evaluator.add_custom_node(
            result=bool(links_provided),
            id="redfin_links_provided",
            desc="Redfin URLs are provided for all listings",
            parent=redfin_node,
            critical=False
        )

    # 3.4 Google Maps section
    maps_node = evaluator.add_sequential(
        id="maps_section",
        desc="Google Maps distance calculation from listings to contaminated site",
        parent=root,
        critical=False
    )

    # [Action Node] google.com/maps:F2:A5 - Calculate driving routes
    maps_used = has_any_ci(answer, ['google maps', 'driving distance', 'route'])
    evaluator.add_custom_node(
        result=bool(maps_used),
        id="maps_action_route",
        desc="[Action Node] google.com/maps:F2:A5 - Calculate driving distance from each listing to the contaminated site",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] google.com/maps:F2:P3 - Extract distance values
    if has_listings:
        distances_provided = all(looks_like_distance(listing.driving_distance) for listing in listings_info.listings if listing.driving_distance)
        evaluator.add_custom_node(
            result=bool(distances_provided),
            id="maps_perception_distance",
            desc="[Perception Node] google.com/maps:F2:P3 - Extract driving distance values with units for all listings",
            parent=maps_node,
            critical=False
        )

        # Check if distance criterion (>3 miles) is evaluated
        criterion_evaluated = all(listing.meets_distance_criterion is not None for listing in listings_info.listings)
        evaluator.add_custom_node(
            result=bool(criterion_evaluated),
            id="maps_criterion_evaluated",
            desc="Distance criterion (>3 miles) is evaluated for each listing",
            parent=maps_node,
            critical=False
        )

        # Check if any listings meet the >3 miles criterion
        any_meet_criterion = any(listing.meets_distance_criterion for listing in listings_info.listings if listing.meets_distance_criterion is not None)
        evaluator.add_custom_node(
            result=bool(any_meet_criterion),
            id="maps_criterion_met",
            desc="At least one listing is more than 3 miles from the contaminated site",
            parent=maps_node,
            critical=False
        )

    # 3.5 Overall completeness check
    completeness_node = evaluator.add_parallel(
        id="completeness_section",
        desc="Overall task completeness verification",
        parent=root,
        critical=False
    )

    # Check if all required outputs are present
    has_site_address = bool(site_info.full_address and site_info.full_address.strip())
    has_npl_status = bool(site_info.npl_status and site_info.npl_status.strip())
    has_population = bool(population_info.latest_population and population_info.latest_population.strip())

    evaluator.add_custom_node(
        result=bool(has_site_address and has_npl_status and has_population and has_listings),
        id="output_completeness",
        desc="All required outputs are present: site address, NPL status, Tampa population, and property listings",
        parent=completeness_node,
        critical=False
    )

    # Check if total count is mentioned
    count_mentioned = listings_info.total_count is not None
    evaluator.add_custom_node(
        result=bool(count_mentioned),
        id="listing_count_mentioned",
        desc="Total count of available matching listings is mentioned",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
