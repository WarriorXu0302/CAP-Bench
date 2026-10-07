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
TASK_ID = "task-fa0c9b"
TASK_DESCRIPTION = 'I’m 4 months pregnant and want to buy some pregnancy-safe skincare products. First, help me find 5 facial moisturizers on Sephora (either creams or serums), making sure they are priced between $30–$60, rated 4.5 or higher, and prioritizing products with a **Bestseller** tag. Then, check these 5 products one by one in the EWG Skin Deep database and prioritize keeping products with an EWG overall hazard score of 1–3 (low risk). If fewer than enough products meet the 1–3 range, keep the remaining products and clearly label their scores and risk differences in the results.  \n\nNext, expand each product’s ingredient list, copy the key ingredients (first 5 ingredients), and search PubMed for “pregnancy safe [ingredient name]” (e.g., “pregnancy safe retinol”). Review the abstracts to confirm whether these ingredients have any pregnancy-related contraindications.  \n\nFinally, output: product name, brand, Sephora price and rating, Sephora product link, whether it has a Bestseller tag, EWG overall score, EWG product link, key ingredient list (first 5), and a pregnancy safety note (e.g., “Contains retinol, not recommended during pregnancy” or “Ingredients appear safe and suitable for pregnancy”). Also recommend 2–3 products with the best overall performance and stronger pregnancy safety among the available results; if fewer than 2 are suitable, recommend all that are reasonably recommendable and explain why.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    """Product information extracted from the answer"""
    product_name: Optional[str] = None
    brand: Optional[str] = None
    price_text: Optional[str] = None
    rating_text: Optional[str] = None
    sephora_link: Optional[str] = None
    has_bestseller: Optional[bool] = None
    ewg_score_text: Optional[str] = None
    ewg_link: Optional[str] = None
    key_ingredients: Optional[List[str]] = Field(default_factory=list)
    pregnancy_safety_note: Optional[str] = None


class ExtractedProducts(BaseModel):
    """All products extracted from the answer"""
    products: List[ProductInfo] = Field(default_factory=list)
    recommendations: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_products_from_answer() -> str:
    return """
Extract all pregnancy-safe facial moisturizer products reported in the answer.

For each product, extract:
- product_name: the product name
- brand: the brand name
- price_text: the Sephora price exactly as stated (include currency symbols)
- rating_text: the Sephora rating exactly as stated
- sephora_link: the Sephora product link URL
- has_bestseller: true if the product has a Bestseller tag, false otherwise, null if not mentioned
- ewg_score_text: the EWG overall hazard score exactly as stated
- ewg_link: the EWG product link URL
- key_ingredients: a list of the first 5 ingredients (or fewer if less than 5 are provided)
- pregnancy_safety_note: the pregnancy safety assessment or note

Also extract:
- recommendations: the final product recommendations text (2-3 products or explanation)

If any field is missing for a product, set it to null or empty list for key_ingredients.
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


def is_price_in_range(price_text: Optional[str], min_price: float = 30.0, max_price: float = 60.0) -> bool:
    price_val = extract_float(price_text)
    if price_val is None:
        return False
    return min_price <= price_val <= max_price


def is_rating_sufficient(rating_text: Optional[str], min_rating: float = 4.5) -> bool:
    rating_val = extract_float(rating_text)
    if rating_val is None:
        return False
    return rating_val >= min_rating


def is_ewg_score_low_risk(ewg_score_text: Optional[str]) -> bool:
    score_val = extract_float(ewg_score_text)
    if score_val is None:
        return False
    return 1.0 <= score_val <= 3.0


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def looks_like_sephora_url(text: Optional[str]) -> bool:
    if not looks_like_url(text):
        return False
    return 'sephora.com' in text.lower()


def looks_like_ewg_url(text: Optional[str]) -> bool:
    if not looks_like_url(text):
        return False
    return 'ewg.org' in text.lower()


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
        prompt=prompt_extract_products_from_answer(),
        template_class=ExtractedProducts,
        extraction_name="extracted_products"
    )

    products = extracted.products if extracted and extracted.products else []
    recommendations = extracted.recommendations if extracted else None

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Sephora product search section
    sephora_section = evaluator.add_sequential(
        id="sephora_section",
        desc="Sephora: Find 5 facial moisturizers with specified criteria",
        parent=root,
        critical=False
    )

    # [Action Node] sephora.com:F1:A1 - Multi-criteria filtering (price and rating)
    products_in_price_range = sum(1 for p in products if is_price_in_range(p.price_text))
    products_rating_ok = sum(1 for p in products if is_rating_sufficient(p.rating_text))
    price_and_rating_ok = products_in_price_range >= 3 and products_rating_ok >= 3

    evaluator.add_custom_node(
        result=bool(price_and_rating_ok),
        id="sephora_multi_criteria_filter",
        desc="[Action Node] sephora.com:F1:A1 - Filter products by price ($30-60) and rating (4.5+)",
        parent=sephora_section,
        critical=False
    )

    # [Action Node] sephora.com:F1:A2 - Sort/prioritize by Bestseller tag
    has_bestseller_products = sum(1 for p in products if p.has_bestseller is True) > 0
    mentions_bestseller = has_any_ci(answer, ['bestseller'])

    evaluator.add_custom_node(
        result=bool(has_bestseller_products and mentions_bestseller),
        id="sephora_bestseller_priority",
        desc="[Action Node] sephora.com:F1:A2 - Prioritize products with Bestseller tag",
        parent=sephora_section,
        critical=False
    )

    # [Action Node] sephora.com:F1:A3 - Navigate to product details
    valid_sephora_links = sum(1 for p in products if looks_like_sephora_url(p.sephora_link))

    evaluator.add_custom_node(
        result=bool(valid_sephora_links >= 3),
        id="sephora_product_details",
        desc="[Action Node] sephora.com:F1:A3 - Access product detail pages to gather complete information",
        parent=sephora_section,
        critical=False
    )

    # [Perception Node] sephora.com:F1:P1 - Identify Bestseller tags
    bestseller_status_recorded = sum(1 for p in products if p.has_bestseller is not None) >= 3

    evaluator.add_custom_node(
        result=bool(bestseller_status_recorded),
        id="sephora_bestseller_identification",
        desc="[Perception Node] sephora.com:F1:P1 - Identify and record Bestseller tag status for products",
        parent=sephora_section,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P2 - Extract rating data
    ratings_extracted = sum(1 for p in products if p.rating_text and extract_float(p.rating_text) is not None) >= 3

    evaluator.add_custom_node(
        result=bool(ratings_extracted),
        id="sephora_rating_perception",
        desc="[Perception Node] sephora.com:F2:P2 - Extract product rating values from detail pages",
        parent=sephora_section,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P7 - Extract price information
    prices_extracted = sum(1 for p in products if p.price_text and extract_float(p.price_text) is not None) >= 3

    evaluator.add_custom_node(
        result=bool(prices_extracted),
        id="sephora_price_perception",
        desc="[Perception Node] sephora.com:F2:P7 - Extract product price information from detail pages",
        parent=sephora_section,
        critical=False
    )

    # [Action Node] sephora.com:F2:A9 - Expand ingredient list
    products_with_ingredients = sum(1 for p in products if p.key_ingredients and len(p.key_ingredients) > 0) >= 3
    mentions_ingredients = has_any_ci(answer, ['ingredient'])

    evaluator.add_custom_node(
        result=bool(products_with_ingredients and mentions_ingredients),
        id="sephora_expand_ingredients",
        desc="[Action Node] sephora.com:F2:A9 - Expand ingredient list panel to view full ingredients",
        parent=sephora_section,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P4 - Extract first 5 ingredients
    products_with_5_ingredients = sum(1 for p in products if p.key_ingredients and len(p.key_ingredients) >= 3) >= 2

    evaluator.add_custom_node(
        result=bool(products_with_5_ingredients),
        id="sephora_extract_key_ingredients",
        desc="[Perception Node] sephora.com:F2:P4 - Extract first 5 ingredients from ingredient list",
        parent=sephora_section,
        critical=False
    )

    # 3.2 EWG Skin Deep safety check section
    ewg_section = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG Skin Deep: Check safety scores for products",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Search products in EWG database
    valid_ewg_links = sum(1 for p in products if looks_like_ewg_url(p.ewg_link))
    ewg_scores_present = sum(1 for p in products if p.ewg_score_text and extract_float(p.ewg_score_text) is not None)
    mentions_ewg = has_any_ci(answer, ['ewg'])

    evaluator.add_custom_node(
        result=bool(valid_ewg_links >= 3 and ewg_scores_present >= 3 and mentions_ewg),
        id="ewg_product_search",
        desc="[Action Node] ewg.org:F1:A1 - Search each product in EWG Skin Deep database",
        parent=ewg_section,
        critical=False
    )

    # Check prioritization of low-risk products (non-prefixed check)
    low_risk_products = sum(1 for p in products if is_ewg_score_low_risk(p.ewg_score_text))
    has_risk_labeling = has_any_ci(answer, ['low risk', 'score', 'hazard'])

    evaluator.add_custom_node(
        result=bool(low_risk_products > 0 and has_risk_labeling),
        id="ewg_risk_prioritization",
        desc="Prioritize products with EWG score 1-3 (low risk) and label risk levels",
        parent=ewg_section,
        critical=False
    )

    # 3.3 PubMed ingredient safety verification section
    pubmed_section = evaluator.add_sequential(
        id="pubmed_section",
        desc="PubMed: Verify ingredient pregnancy safety",
        parent=root,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Search literature for ingredients
    mentions_pubmed = has_any_ci(answer, ['pubmed'])
    has_pregnancy_searches = has_any_ci(answer, ['pregnancy safe', 'pregnancy'])
    products_with_safety_notes = sum(1 for p in products if p.pregnancy_safety_note and p.pregnancy_safety_note.strip()) >= 3

    evaluator.add_custom_node(
        result=bool(mentions_pubmed and has_pregnancy_searches and products_with_safety_notes),
        id="pubmed_ingredient_search",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Search PubMed for 'pregnancy safe [ingredient]' for key ingredients",
        parent=pubmed_section,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Understand literature list content
    has_literature_references = has_any_ci(answer, ['abstract', 'study', 'research', 'literature'])

    evaluator.add_custom_node(
        result=bool(has_literature_references and products_with_safety_notes),
        id="pubmed_literature_understanding",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P2 - Review and understand literature search results",
        parent=pubmed_section,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P3 - Understand abstract snippets
    has_contraindication_info = has_any_ci(answer, ['contraindication', 'not recommended', 'safe', 'suitable'])

    evaluator.add_custom_node(
        result=bool(has_contraindication_info and products_with_safety_notes),
        id="pubmed_abstract_understanding",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F1:P3 - Review abstracts to identify pregnancy contraindications",
        parent=pubmed_section,
        critical=False
    )

    # 3.4 Final output completeness section
    output_section = evaluator.add_parallel(
        id="output_section",
        desc="Final output: Complete product comparison and recommendations",
        parent=root,
        critical=False
    )

    # Check for complete product information (at least 3 products with most fields)
    complete_products = 0
    for p in products:
        fields_present = sum([
            bool(p.product_name),
            bool(p.brand),
            bool(p.price_text),
            bool(p.rating_text),
            bool(p.sephora_link),
            p.has_bestseller is not None,
            bool(p.ewg_score_text),
            bool(p.ewg_link),
            bool(p.key_ingredients and len(p.key_ingredients) > 0),
            bool(p.pregnancy_safety_note)
        ])
        if fields_present >= 8:
            complete_products += 1

    evaluator.add_custom_node(
        result=bool(complete_products >= 3),
        id="complete_product_info",
        desc="Output includes complete product information (name, brand, price, rating, links, scores, ingredients, safety notes)",
        parent=output_section,
        critical=False
    )

    # Check for product recommendations
    has_recommendations = bool(recommendations and len(recommendations.strip()) > 20)
    mentions_recommendation = has_any_ci(answer, ['recommend'])

    evaluator.add_custom_node(
        result=bool(has_recommendations and mentions_recommendation),
        id="product_recommendations",
        desc="Provide 2-3 product recommendations with best safety and performance, or explain if fewer are suitable",
        parent=output_section,
        critical=False
    )

    # Check that at least 5 products were attempted
    five_products_found = len(products) >= 5 or has_any_ci(answer, ['5 products', 'five products'])

    evaluator.add_custom_node(
        result=bool(five_products_found),
        id="five_products_requirement",
        desc="Attempt to find and analyze 5 facial moisturizers as requested",
        parent=output_section,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
