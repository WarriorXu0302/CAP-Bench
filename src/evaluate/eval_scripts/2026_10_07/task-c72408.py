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
TASK_ID = "task-c72408"
TASK_DESCRIPTION = 'I’m planning to relocate to Portland for work next spring, with a budget of **$2,000–$2,500** for a **1-bed, 1-bath apartment**, and I’m especially concerned about environmental health risks.\n\nFirst, use Apartments.com to find **5 apartments** that match the budget and have a rating of **4.0 or higher**, and record their addresses and ZIP codes.\n\nThen, use the **EPA Safe Drinking Water Information System (SDWIS)** to look up, for each address:\n- the corresponding water system name,\n- the number of violations in the past 3 years,\n- and the date of the most recent inspection.\n\nNext, go to Yelp and search for community reviews within **1 mile** of each apartment address. Try to find resident reviews mentioning **“noise,” “air quality,” or “safety,”** with a target of **at least 3 reviews per apartment**. If fewer than 3 are available, keep what you find and note the actual count.\n\nFinally, use Google Maps to measure, for each apartment, the straight-line distance to:\n- **Portland International Airport (PDX)**, and\n- the nearest entrance to **I-5**.\n\nOutput the following for each apartment:\n- Apartments.com link  \n- Address  \n- Monthly rent  \n- Rating  \n- Water system name  \n- Number of violations in the past 3 years  \n- Most recent inspection date  \n- EPA SDWIS link  \n- Yelp community review summary (key points from relevant reviews, and the actual number of reviews collected for that apartment)  \n- Distance to airport  \n- Distance to highway entrance  \n- Google Maps distance-measurement screenshot link  \n\nAt the end, recommend the **2 apartments with the lowest overall risk** and explain why.\n\nIf you encounter login requirements or access restrictions on any site, record the restricted page(s) and the fields that could not be retrieved, then continue with all remaining accessible steps.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ApartmentInfo(BaseModel):
    """Information for a single apartment"""
    apartments_com_link: Optional[str] = None
    address: Optional[str] = None
    monthly_rent: Optional[str] = None
    rating: Optional[str] = None
    water_system_name: Optional[str] = None
    violations_count: Optional[str] = None
    recent_inspection_date: Optional[str] = None
    epa_sdwis_link: Optional[str] = None
    yelp_review_summary: Optional[str] = None
    yelp_review_count: Optional[str] = None
    distance_to_airport: Optional[str] = None
    distance_to_highway: Optional[str] = None
    google_maps_link: Optional[str] = None


class ExtractedApartments(BaseModel):
    """All apartments extracted from the answer"""
    apartments: List[ApartmentInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


class Recommendations(BaseModel):
    """Recommendations extracted from the answer"""
    recommended_apartments: Optional[str] = None
    explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_apartments_from_answer() -> str:
    return """
Extract all apartment listings from the answer. For each apartment, extract:
- apartments_com_link: the Apartments.com URL
- address: full street address including ZIP code
- monthly_rent: the monthly rent amount
- rating: the rating value
- water_system_name: the EPA water system name
- violations_count: number of violations in past 3 years
- recent_inspection_date: most recent inspection date
- epa_sdwis_link: EPA SDWIS link
- yelp_review_summary: summary of Yelp reviews
- yelp_review_count: actual number of reviews collected
- distance_to_airport: distance to PDX
- distance_to_highway: distance to I-5 entrance
- google_maps_link: Google Maps screenshot or measurement link

Also extract total_count: the total number of apartments listed.

If any field is missing for an apartment, set it to null.
"""


def prompt_extract_recommendations_from_answer() -> str:
    return """
Extract the recommendations from the answer:
- recommended_apartments: which 2 apartments were recommended
- explanation: the explanation for why they have the lowest risk

If not present, set fields to null.
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
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.findall(r'(\d+)', text)
    if not m:
        return None
    try:
        return int(m[0])
    except Exception:
        return None


def is_valid_rent_range(rent_text: Optional[str]) -> bool:
    if not rent_text:
        return False
    val = extract_float(rent_text)
    if val is None:
        return False
    return 2000 <= val <= 2500


def is_valid_rating(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    val = extract_float(rating_text)
    if val is None:
        return False
    return val >= 4.0


def contains_portland(text: Optional[str]) -> bool:
    return ci_contains(text, 'portland')


def contains_zipcode(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.search(r'\b\d{5}\b', text))


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{4}-\d{1,2}-\d{1,2}',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def contains_distance_unit(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['mile', 'mi', 'km', 'kilometer'])


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
    apartments_data = await evaluator.extract(
        prompt=prompt_extract_apartments_from_answer(),
        template_class=ExtractedApartments,
        extraction_name="apartments_list"
    )

    recommendations = await evaluator.extract(
        prompt=prompt_extract_recommendations_from_answer(),
        template_class=Recommendations,
        extraction_name="recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Apartments.com section
    apartments_com_node = evaluator.add_sequential(
        id="apartments_com_section",
        desc="Apartments.com search and listing extraction",
        parent=root,
        critical=False
    )

    # [Action Node] apartments.com:F1:A9 - City location search
    city_search_ok = contains_portland(answer)
    evaluator.add_custom_node(
        result=bool(city_search_ok),
        id="apartments_com_city_search",
        desc="[Action Node] apartments.com:F1:A9 - Search for apartments in Portland",
        parent=apartments_com_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A1 - Price filter
    rent_filter_ok = has_any_ci(answer, ['2000', '2500', 'budget', 'rent'])
    evaluator.add_custom_node(
        result=bool(rent_filter_ok),
        id="apartments_com_price_filter",
        desc="[Action Node] apartments.com:F1:A1 - Apply price filter for $2,000-$2,500 range",
        parent=apartments_com_node,
        critical=False
    )

    # [Action Node] apartments.com:F1:A2 - Bedroom filter
    bedroom_filter_ok = has_any_ci(answer, ['1 bed', '1-bed', '1bed', '1 bedroom', '1br'])
    evaluator.add_custom_node(
        result=bool(bedroom_filter_ok),
        id="apartments_com_bedroom_filter",
        desc="[Action Node] apartments.com:F1:A2 - Apply 1-bedroom filter",
        parent=apartments_com_node,
        critical=False
    )

    # [Perception Node] apartments.com:F1:P1 - Rating identification
    rating_mention_ok = has_any_ci(answer, ['rating', '4.0', '4.'])
    evaluator.add_custom_node(
        result=bool(rating_mention_ok),
        id="apartments_com_rating_perception",
        desc="[Perception Node] apartments.com:F1:P1 - Identify apartments with rating 4.0 or higher",
        parent=apartments_com_node,
        critical=False
    )

    # [Action Node] apartments.com:F2:A12 - Navigate to detail page
    detail_page_ok = apartments_data and apartments_data.apartments and len(apartments_data.apartments) > 0
    evaluator.add_custom_node(
        result=bool(detail_page_ok),
        id="apartments_com_detail_page",
        desc="[Action Node] apartments.com:F2:A12 - Navigate to apartment detail pages to extract full address and ZIP",
        parent=apartments_com_node,
        critical=False
    )

    # Check extracted apartment data quality
    apartments_quality_node = evaluator.add_parallel(
        id="apartments_data_quality",
        desc="Validate extracted apartment data meets requirements",
        parent=apartments_com_node,
        critical=False
    )

    # Check apartment count
    count_ok = apartments_data and apartments_data.apartments and len(apartments_data.apartments) >= 5
    evaluator.add_custom_node(
        result=bool(count_ok),
        id="apartments_count_check",
        desc="At least 5 apartments were found",
        parent=apartments_quality_node,
        critical=False
    )

    # Check individual apartment data
    if apartments_data and apartments_data.apartments:
        for idx, apt in enumerate(apartments_data.apartments[:5], 1):
            apt_node = evaluator.add_parallel(
                id=f"apartment_{idx}_validation",
                desc=f"Validate apartment {idx} data",
                parent=apartments_quality_node,
                critical=False
            )

            # Rent in range
            rent_ok = is_valid_rent_range(apt.monthly_rent)
            evaluator.add_custom_node(
                result=bool(rent_ok),
                id=f"apt_{idx}_rent_range",
                desc=f"Apartment {idx} rent is within $2,000-$2,500",
                parent=apt_node,
                critical=False
            )

            # Rating >= 4.0
            rating_ok = is_valid_rating(apt.rating)
            evaluator.add_custom_node(
                result=bool(rating_ok),
                id=f"apt_{idx}_rating",
                desc=f"Apartment {idx} rating is 4.0 or higher",
                parent=apt_node,
                critical=False
            )

            # Address includes Portland
            portland_ok = contains_portland(apt.address)
            evaluator.add_custom_node(
                result=bool(portland_ok),
                id=f"apt_{idx}_portland",
                desc=f"Apartment {idx} address includes Portland",
                parent=apt_node,
                critical=False
            )

            # ZIP code present
            zip_ok = contains_zipcode(apt.address)
            evaluator.add_custom_node(
                result=bool(zip_ok),
                id=f"apt_{idx}_zipcode",
                desc=f"Apartment {idx} address includes ZIP code",
                parent=apt_node,
                critical=False
            )

    # 3.2 EPA SDWIS section
    epa_node = evaluator.add_sequential(
        id="epa_sdwis_section",
        desc="EPA Safe Drinking Water Information System lookups",
        parent=root,
        critical=False
    )

    # [Action Node] epa.gov:F8:A18 - State/region selection
    oregon_ok = has_any_ci(answer, ['oregon', 'or'])
    evaluator.add_custom_node(
        result=bool(oregon_ok),
        id="epa_state_selection",
        desc="[Action Node] epa.gov:F8:A18 - Select Oregon state in SDWIS system",
        parent=epa_node,
        critical=False
    )

    # [Perception Node] epa.gov:F8:P14 - Water system data table understanding
    epa_data_ok = False
    if apartments_data and apartments_data.apartments:
        for apt in apartments_data.apartments:
            if apt.water_system_name or apt.violations_count or apt.recent_inspection_date:
                epa_data_ok = True
                break

    evaluator.add_custom_node(
        result=bool(epa_data_ok),
        id="epa_data_table_understanding",
        desc="[Perception Node] epa.gov:F8:P14 - Extract water system name, violations count, and inspection date from SDWIS data",
        parent=epa_node,
        critical=False
    )

    # Check EPA data quality for each apartment
    if apartments_data and apartments_data.apartments:
        epa_quality_node = evaluator.add_parallel(
            id="epa_data_quality",
            desc="Validate EPA data for each apartment",
            parent=epa_node,
            critical=False
        )

        for idx, apt in enumerate(apartments_data.apartments[:5], 1):
            apt_epa_node = evaluator.add_parallel(
                id=f"apartment_{idx}_epa_data",
                desc=f"EPA data for apartment {idx}",
                parent=epa_quality_node,
                critical=False
            )

            # Water system name present
            water_system_ok = bool(apt.water_system_name and apt.water_system_name.strip())
            evaluator.add_custom_node(
                result=bool(water_system_ok),
                id=f"apt_{idx}_water_system",
                desc=f"Apartment {idx} has water system name",
                parent=apt_epa_node,
                critical=False
            )

            # Violations count present (numeric)
            violations_ok = apt.violations_count is not None and extract_int(apt.violations_count) is not None
            evaluator.add_custom_node(
                result=bool(violations_ok),
                id=f"apt_{idx}_violations",
                desc=f"Apartment {idx} has violations count for past 3 years",
                parent=apt_epa_node,
                critical=False
            )

            # Inspection date present
            inspection_ok = looks_like_date(apt.recent_inspection_date)
            evaluator.add_custom_node(
                result=bool(inspection_ok),
                id=f"apt_{idx}_inspection_date",
                desc=f"Apartment {idx} has most recent inspection date",
                parent=apt_epa_node,
                critical=False
            )

    # 3.3 Yelp section
    yelp_node = evaluator.add_sequential(
        id="yelp_section",
        desc="Yelp community reviews for environmental concerns",
        parent=root,
        critical=False
    )

    # [Action Node] yelp.com:F1:A1 - Location + keyword search
    yelp_search_ok = has_any_ci(answer, ['yelp'])
    evaluator.add_custom_node(
        result=bool(yelp_search_ok),
        id="yelp_location_search",
        desc="[Action Node] yelp.com:F1:A1 - Search Yelp for reviews near each apartment address",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F1:A2 - Distance filter
    distance_filter_ok = has_any_ci(answer, ['1 mile', 'within 1 mile', '1-mile'])
    evaluator.add_custom_node(
        result=bool(distance_filter_ok),
        id="yelp_distance_filter",
        desc="[Action Node] yelp.com:F1:A2 - Apply 1-mile distance filter",
        parent=yelp_node,
        critical=False
    )

    # [Action Node] yelp.com:F2:A9 - Navigate to business detail page
    yelp_detail_ok = has_any_ci(answer, ['review', 'noise', 'air quality', 'safety'])
    evaluator.add_custom_node(
        result=bool(yelp_detail_ok),
        id="yelp_detail_page",
        desc="[Action Node] yelp.com:F2:A9 - Navigate to business detail pages to read full reviews",
        parent=yelp_node,
        critical=False
    )

    # [Perception Node] yelp.com:F2:P3 - Review content understanding
    keywords_ok = has_any_ci(answer, ['noise']) or has_any_ci(answer, ['air quality']) or has_any_ci(answer, ['safety'])
    evaluator.add_custom_node(
        result=bool(keywords_ok),
        id="yelp_review_understanding",
        desc="[Perception Node] yelp.com:F2:P3 - Understand and extract key points from reviews mentioning noise, air quality, or safety",
        parent=yelp_node,
        critical=False
    )

    # Check Yelp data quality for each apartment
    if apartments_data and apartments_data.apartments:
        yelp_quality_node = evaluator.add_parallel(
            id="yelp_data_quality",
            desc="Validate Yelp review data for each apartment",
            parent=yelp_node,
            critical=False
        )

        for idx, apt in enumerate(apartments_data.apartments[:5], 1):
            apt_yelp_node = evaluator.add_parallel(
                id=f"apartment_{idx}_yelp_data",
                desc=f"Yelp data for apartment {idx}",
                parent=yelp_quality_node,
                critical=False
            )

            # Review summary present
            summary_ok = bool(apt.yelp_review_summary and apt.yelp_review_summary.strip())
            evaluator.add_custom_node(
                result=bool(summary_ok),
                id=f"apt_{idx}_yelp_summary",
                desc=f"Apartment {idx} has Yelp review summary",
                parent=apt_yelp_node,
                critical=False
            )

            # Review count noted
            count_noted = bool(apt.yelp_review_count and apt.yelp_review_count.strip())
            evaluator.add_custom_node(
                result=bool(count_noted),
                id=f"apt_{idx}_yelp_count",
                desc=f"Apartment {idx} notes actual number of reviews collected",
                parent=apt_yelp_node,
                critical=False
            )

    # 3.4 Google Maps section
    maps_node = evaluator.add_sequential(
        id="google_maps_section",
        desc="Google Maps distance measurements",
        parent=root,
        critical=False
    )

    # [Action Node] maps.google.com:F1:A2 - Location search
    maps_search_ok = has_any_ci(answer, ['google maps', 'maps']) or has_any_ci(answer, ['distance'])
    evaluator.add_custom_node(
        result=bool(maps_search_ok),
        id="maps_location_search",
        desc="[Action Node] maps.google.com:F1:A2 - Search for apartment addresses, PDX airport, and I-5 entrance on Google Maps",
        parent=maps_node,
        critical=False
    )

    # [Action Node] maps.google.com:F3:A9 - Map zoom adjustment
    zoom_ok = has_any_ci(answer, ['distance', 'mile', 'km'])
    evaluator.add_custom_node(
        result=bool(zoom_ok),
        id="maps_zoom_adjustment",
        desc="[Action Node] maps.google.com:F3:A9 - Adjust map zoom to view full route/distance",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P12 - Map marker identification
    markers_ok = (has_any_ci(answer, ['pdx', 'airport']) and has_any_ci(answer, ['i-5', 'highway', 'interstate']))
    evaluator.add_custom_node(
        result=bool(markers_ok),
        id="maps_marker_identification",
        desc="[Perception Node] maps.google.com:F3:P12 - Identify map markers for apartments, airport, and highway entrance",
        parent=maps_node,
        critical=False
    )

    # [Perception Node] maps.google.com:F3:P13 - Map range understanding
    range_ok = has_any_ci(answer, ['straight-line', 'direct', 'distance'])
    evaluator.add_custom_node(
        result=bool(range_ok),
        id="maps_range_understanding",
        desc="[Perception Node] maps.google.com:F3:P13 - Understand map display range and scale for distance measurement",
        parent=maps_node,
        critical=False
    )

    # Check Google Maps data quality for each apartment
    if apartments_data and apartments_data.apartments:
        maps_quality_node = evaluator.add_parallel(
            id="maps_data_quality",
            desc="Validate Google Maps distance data for each apartment",
            parent=maps_node,
            critical=False
        )

        for idx, apt in enumerate(apartments_data.apartments[:5], 1):
            apt_maps_node = evaluator.add_parallel(
                id=f"apartment_{idx}_maps_data",
                desc=f"Google Maps data for apartment {idx}",
                parent=maps_quality_node,
                critical=False
            )

            # Airport distance present
            airport_dist_ok = bool(apt.distance_to_airport and contains_distance_unit(apt.distance_to_airport))
            evaluator.add_custom_node(
                result=bool(airport_dist_ok),
                id=f"apt_{idx}_airport_distance",
                desc=f"Apartment {idx} has distance to PDX airport with units",
                parent=apt_maps_node,
                critical=False
            )

            # Highway distance present
            highway_dist_ok = bool(apt.distance_to_highway and contains_distance_unit(apt.distance_to_highway))
            evaluator.add_custom_node(
                result=bool(highway_dist_ok),
                id=f"apt_{idx}_highway_distance",
                desc=f"Apartment {idx} has distance to I-5 entrance with units",
                parent=apt_maps_node,
                critical=False
            )

    # 3.5 Recommendations section
    recommendations_node = evaluator.add_sequential(
        id="recommendations_section",
        desc="Final recommendations for lowest-risk apartments",
        parent=root,
        critical=False
    )

    # Check if 2 apartments were recommended
    two_recommended = False
    if recommendations and recommendations.recommended_apartments:
        rec_text = recommendations.recommended_apartments.lower()
        # Look for "2" or "two" in recommendation
        two_recommended = bool(re.search(r'\b(2|two)\b', rec_text))

    evaluator.add_custom_node(
        result=bool(two_recommended),
        id="two_apartments_recommended",
        desc="Recommended 2 apartments with lowest overall risk",
        parent=recommendations_node,
        critical=False
    )

    # Check if explanation is provided
    explanation_ok = bool(recommendations and recommendations.explanation and len(recommendations.explanation.strip()) > 20)
    evaluator.add_custom_node(
        result=bool(explanation_ok),
        id="recommendation_explanation",
        desc="Provided explanation for why the recommended apartments have lowest risk",
        parent=recommendations_node,
        critical=False
    )

    # Check if explanation mentions environmental/health factors
    risk_factors_ok = False
    if recommendations and recommendations.explanation:
        risk_factors_ok = (
            has_any_ci(recommendations.explanation, ['water', 'violation']) or
            has_any_ci(recommendations.explanation, ['noise', 'air', 'safety']) or
            has_any_ci(recommendations.explanation, ['airport', 'highway', 'distance']) or
            has_any_ci(recommendations.explanation, ['environmental', 'health', 'risk'])
        )

    evaluator.add_custom_node(
        result=bool(risk_factors_ok),
        id="risk_factors_mentioned",
        desc="Explanation mentions environmental/health risk factors in decision",
        parent=recommendations_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
