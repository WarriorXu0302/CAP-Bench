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
TASK_ID = "task-fb1952"
TASK_DESCRIPTION = 'I just moved to Newark, New Jersey (ZIP code 07102), and I’m concerned about the quality of the local tap water. Please investigate and find a solution for me.\n\nFirst, go to the EWG Tap Water Database, enter the ZIP code, and identify the contaminant with the **highest “times over EWG Health Guideline.”**\n\nNext, go to the CDC website and use the A–Z index to find the health effects of that contaminant. If the CDC A–Z index does not have a standalone entry for the contaminant, then search within the CDC site for the health effects of the contaminant’s broader category (e.g., disinfection byproducts), and briefly note one key risk.\n\nFinally, go to Amazon and find a water filter pitcher that can filter this specific contaminant.\n\nRequirements:\n- Price must be between **$20 and $60**\n- Rating must be **4 stars or higher**\n- Brand must be **Brita, PUR, or ZeroWater**\n\nCarefully review product images or detailed technical specifications to ensure the product explicitly states it can remove or reduce this specific contaminant. If no product explicitly names the contaminant, choose a pitcher that clearly states it reduces related contaminant groups (e.g., disinfection byproducts, THMs/HAA5), and note in the output that this is a **“category-claim match.”**\n\nOutput required:\n- Contaminant name\n- Times over guideline\n- CDC health risk\n- Filter pitcher name\n- Price\n- Rating\n- Supporting text proving it filters the contaminant\n- Links for each step'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ContaminantInfo(BaseModel):
    """Contaminant information extracted from EWG Tap Water Database"""
    contaminant_name: Optional[str] = None
    times_over_guideline: Optional[str] = None


class HealthRiskInfo(BaseModel):
    """Health risk information extracted from CDC"""
    health_risk: Optional[str] = None


class FilterPitcherInfo(BaseModel):
    """Filter pitcher information extracted from Amazon"""
    pitcher_name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    supporting_text: Optional[str] = None


class LinksInfo(BaseModel):
    """Links extracted from the answer"""
    ewg_link: Optional[str] = None
    cdc_link: Optional[str] = None
    amazon_link: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_contaminant_from_answer() -> str:
    return """
Extract the contaminant information from the EWG Tap Water Database for Newark, NJ (ZIP 07102) as reported in the answer.

Return:
- contaminant_name: the name of the contaminant with the highest "times over EWG Health Guideline" exactly as written.
- times_over_guideline: the numeric value or text describing how many times over the guideline (include units if present).

If any field is missing in the answer, set it to null.
"""


def prompt_extract_health_risk_from_answer() -> str:
    return """
From the answer, extract the CDC health risk information for the contaminant identified from EWG.

Return:
- health_risk: the key health risk or health effect as stated in the answer (one key risk is sufficient).

If not present, set it to null.
"""


def prompt_extract_filter_pitcher_from_answer() -> str:
    return """
From the answer, extract the Amazon water filter pitcher information that can filter the identified contaminant.

Return:
- pitcher_name: the full product name of the filter pitcher.
- price: the price exactly as stated (include currency symbol if present).
- rating: the rating exactly as stated (e.g., "4.5 out of 5 stars").
- supporting_text: the text from product images or specifications that proves it filters the specific contaminant or related category.

If any field is missing, set it to null.
"""


def prompt_extract_links_from_answer() -> str:
    return """
From the answer, extract the URLs/links for each step of the investigation.

Return:
- ewg_link: the EWG Tap Water Database link used.
- cdc_link: the CDC website link used.
- amazon_link: the Amazon product link for the chosen filter pitcher.

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


def extract_price_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Match patterns like $29.99, 29.99, $29
    m = re.search(r'\$?\s*(\d+(?:\.\d{2})?)', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_rating_value(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Match patterns like "4.5 out of 5", "4.5", "4 stars"
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def is_valid_brand(pitcher_name: Optional[str]) -> bool:
    if not pitcher_name:
        return False
    valid_brands = ['brita', 'pur', 'zerowater']
    return any(brand in pitcher_name.lower() for brand in valid_brands)


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(re.match(r'https?://', text, re.IGNORECASE))


def mentions_newark_or_zipcode(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['newark', '07102'])


def mentions_category_claim(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['category-claim', 'category claim', 'related contaminant', 'disinfection byproducts', 'thm', 'haa5'])


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
    contaminant_info = await evaluator.extract(
        prompt=prompt_extract_contaminant_from_answer(),
        template_class=ContaminantInfo,
        extraction_name="contaminant_info"
    )

    health_risk_info = await evaluator.extract(
        prompt=prompt_extract_health_risk_from_answer(),
        template_class=HealthRiskInfo,
        extraction_name="health_risk_info"
    )

    filter_pitcher_info = await evaluator.extract(
        prompt=prompt_extract_filter_pitcher_from_answer(),
        template_class=FilterPitcherInfo,
        extraction_name="filter_pitcher_info"
    )

    links_info = await evaluator.extract(
        prompt=prompt_extract_links_from_answer(),
        template_class=LinksInfo,
        extraction_name="links_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 EWG Tap Water Database section
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG Tap Water Database - Identify highest contaminant for Newark ZIP 07102",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Form interaction: enter ZIP code
    ewg_action_ok = (has_any_ci(answer, ['ewg']) and
                     mentions_newark_or_zipcode(answer) and
                     bool(contaminant_info and contaminant_info.contaminant_name))
    evaluator.add_custom_node(
        result=bool(ewg_action_ok),
        id="ewg_form_interaction",
        desc="[Action Node] ewg.org:F1:A1 - Enter ZIP code 07102 in EWG Tap Water Database and retrieve contaminant data",
        parent=ewg_node,
        critical=False
    )

    # o1 verification: contaminant name is present and looks valid
    contaminant_name_ok = bool(contaminant_info and contaminant_info.contaminant_name and
                               len(contaminant_info.contaminant_name.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(contaminant_name_ok),
        id="ewg_contaminant_name",
        desc="Output o1: Contaminant name is present and non-empty",
        parent=ewg_node,
        critical=False
    )

    # o2 verification: times over guideline is present
    times_over_ok = bool(contaminant_info and contaminant_info.times_over_guideline and
                        contains_digits(contaminant_info.times_over_guideline))
    evaluator.add_custom_node(
        result=bool(times_over_ok),
        id="ewg_times_over_guideline",
        desc="Output o2: Times over guideline is present with numeric value",
        parent=ewg_node,
        critical=False
    )

    # 3.2 CDC website section
    cdc_node = evaluator.add_sequential(
        id="cdc_section",
        desc="CDC website - Find health effects via A-Z index or search",
        parent=root,
        critical=False
    )

    # [Action Node] cdc.gov:F1:A1 - Index navigation: use A-Z index
    cdc_action_ok = (has_any_ci(answer, ['cdc']) and
                     (has_any_ci(answer, ['a-z', 'a–z', 'index', 'a to z']) or
                      has_any_ci(answer, ['search'])))
    evaluator.add_custom_node(
        result=bool(cdc_action_ok),
        id="cdc_index_navigation",
        desc="[Action Node] cdc.gov:F1:A1 - Navigate CDC A-Z index or search for contaminant health effects",
        parent=cdc_node,
        critical=False
    )

    # o3 verification: health risk describes the contaminant from o1
    health_risk_present = bool(health_risk_info and health_risk_info.health_risk and
                              len(health_risk_info.health_risk.strip()) > 0)
    # Additional check: contaminant name appears in context or category mentioned
    health_risk_relevant = (health_risk_present and
                           (ci_contains(answer, contaminant_info.contaminant_name if contaminant_info else '') or
                            mentions_category_claim(answer)))
    evaluator.add_custom_node(
        result=bool(health_risk_relevant),
        id="cdc_health_risk",
        desc="Output o3: CDC health risk is present and relevant to the identified contaminant",
        parent=cdc_node,
        critical=False
    )

    # 3.3 Amazon filter pitcher section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon - Find water filter pitcher meeting all requirements",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F3:A17 - Price range slider/filter ($20-$60)
    price_val = extract_price_value(filter_pitcher_info.price if filter_pitcher_info else None)
    price_in_range = price_val is not None and 20 <= price_val <= 60
    evaluator.add_custom_node(
        result=bool(price_in_range),
        id="amazon_price_filter",
        desc="[Action Node] Amazon:F3:A17 - Apply price filter to show products between $20 and $60",
        parent=amazon_node,
        critical=False
    )

    # o5 verification: price is in range
    evaluator.add_custom_node(
        result=bool(price_in_range),
        id="amazon_price_value",
        desc="Output o5: Price is within $20-$60 range",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A14 - Multi-select brand filter (Brita, PUR, ZeroWater)
    brand_ok = is_valid_brand(filter_pitcher_info.pitcher_name if filter_pitcher_info else None)
    evaluator.add_custom_node(
        result=bool(brand_ok),
        id="amazon_brand_filter",
        desc="[Action Node] Amazon:F3:A14 - Apply brand filter for Brita, PUR, or ZeroWater",
        parent=amazon_node,
        critical=False
    )

    # o4 verification: product brand is in specified list
    evaluator.add_custom_node(
        result=bool(brand_ok),
        id="amazon_brand_value",
        desc="Output o4: Product brand is Brita, PUR, or ZeroWater",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Rating checkbox filter (4 stars and up)
    rating_val = extract_rating_value(filter_pitcher_info.rating if filter_pitcher_info else None)
    rating_ok = rating_val is not None and rating_val >= 4.0
    rating_mentions_stars = has_any_ci(filter_pitcher_info.rating if filter_pitcher_info else None, ['star', 'out of 5'])
    evaluator.add_custom_node(
        result=bool(rating_ok and rating_mentions_stars),
        id="amazon_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Apply rating filter for 4 stars or higher",
        parent=amazon_node,
        critical=False
    )

    # o6 verification: rating is 4 stars or higher
    evaluator.add_custom_node(
        result=bool(rating_ok),
        id="amazon_rating_value",
        desc="Output o6: Rating is 4 stars or higher",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F1:P3 - Image understanding (verify pitcher type)
    pitcher_type_ok = has_any_ci(filter_pitcher_info.pitcher_name if filter_pitcher_info else None, ['pitcher'])
    evaluator.add_custom_node(
        result=bool(pitcher_type_ok),
        id="amazon_image_understanding",
        desc="[Perception Node] Amazon:F1:P3 - Verify product is a pitcher type filter (not faucet-mount)",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A23 - Carousel interaction (switch images to find specs)
    # Lenient: if supporting text is present, assume carousel was used if needed
    carousel_likely = bool(filter_pitcher_info and filter_pitcher_info.supporting_text)
    evaluator.add_custom_node(
        result=bool(carousel_likely),
        id="amazon_carousel_interaction",
        desc="[Action Node] Amazon:F5:A23 - Navigate product image carousel to find specifications",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A24 - Collapsible panel interaction (Product Details)
    # Lenient: if supporting text mentions specs or details, assume panel was expanded
    panel_likely = bool(filter_pitcher_info and filter_pitcher_info.supporting_text and
                       has_any_ci(filter_pitcher_info.supporting_text, ['spec', 'detail', 'reduce', 'remove', 'filter']))
    evaluator.add_custom_node(
        result=bool(panel_likely),
        id="amazon_panel_interaction",
        desc="[Action Node] Amazon:F5:A24 - Expand Product Details or specifications panel",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P11 - OCR from product images
    # Lenient: if supporting text is present and references contaminant or category, assume OCR was used
    ocr_likely = bool(filter_pitcher_info and filter_pitcher_info.supporting_text and
                     (ci_contains(filter_pitcher_info.supporting_text, contaminant_info.contaminant_name if contaminant_info else '') or
                      mentions_category_claim(filter_pitcher_info.supporting_text if filter_pitcher_info else '')))
    evaluator.add_custom_node(
        result=bool(ocr_likely),
        id="amazon_ocr_perception",
        desc="[Perception Node] Amazon:F5:P11 - Extract text from product images to verify contaminant filtering claim",
        parent=amazon_node,
        critical=False
    )

    # o7 verification: supporting text contains contaminant name or category
    supporting_text_ok = bool(filter_pitcher_info and filter_pitcher_info.supporting_text and
                             (ci_contains(filter_pitcher_info.supporting_text, contaminant_info.contaminant_name if contaminant_info else '') or
                              mentions_category_claim(filter_pitcher_info.supporting_text if filter_pitcher_info else '')))
    evaluator.add_custom_node(
        result=bool(supporting_text_ok),
        id="amazon_supporting_text",
        desc="Output o7: Supporting text proves the product filters the specific contaminant or related category",
        parent=amazon_node,
        critical=False
    )

    # 3.4 Links verification section
    links_node = evaluator.add_parallel(
        id="links_section",
        desc="Verify all required links are provided",
        parent=root,
        critical=False
    )

    ewg_link_ok = looks_like_url(links_info.ewg_link if links_info else None) and has_any_ci(links_info.ewg_link if links_info else None, ['ewg'])
    evaluator.add_custom_node(
        result=bool(ewg_link_ok),
        id="link_ewg",
        desc="EWG Tap Water Database link is present and valid",
        parent=links_node,
        critical=False
    )

    cdc_link_ok = looks_like_url(links_info.cdc_link if links_info else None) and has_any_ci(links_info.cdc_link if links_info else None, ['cdc'])
    evaluator.add_custom_node(
        result=bool(cdc_link_ok),
        id="link_cdc",
        desc="CDC website link is present and valid",
        parent=links_node,
        critical=False
    )

    amazon_link_ok = looks_like_url(links_info.amazon_link if links_info else None) and has_any_ci(links_info.amazon_link if links_info else None, ['amazon'])
    evaluator.add_custom_node(
        result=bool(amazon_link_ok),
        id="link_amazon",
        desc="Amazon product link is present and valid",
        parent=links_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
