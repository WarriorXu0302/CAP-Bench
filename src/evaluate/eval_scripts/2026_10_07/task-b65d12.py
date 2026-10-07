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
TASK_ID = "task-b65d12"
TASK_DESCRIPTION = 'I am 3 months pregnant and want to verify the safety of three skincare products I am currently using.\n\nFirst, go to the EWG Skin Deep database and search for each of these three products: Neutrogena Ultra Sheer Dry-Touch Sunscreen SPF 55, CeraVe Daily Moisturizing Lotion, and The Ordinary Niacinamide 10% + Zinc 1%. Record each product\'s EWG rating (1-10), the names of its primary risk ingredients (if the rating is >3), and the product\'s detail page link.\n\nFor products with a rating >3, extract the names of ingredients marked as \'High Concern\'. Then, go to PubMed and search for each high-concern ingredient combined with the keywords "pregnancy safety" or "fetal development". Filter for Systematic Reviews or Meta-analyses published since 2020. Record the literature title, publication year, main conclusions (specifically whether it indicates a clear contraindication during pregnancy), and the PubMed link.\n\nNext, go to Mayo Clinic and search for "pregnancy skin care" to find the pregnancy skincare guidelines page. Review the "Dos and Don\'ts" list to confirm if the previously identified high-concern ingredients are on Mayo Clinic\'s restricted list. Record Mayo Clinic\'s specific recommendations for these ingredients and the page link.\n\nFinally, return to EWG. For product categories that require replacement (e.g., sunscreen or moisturizer), filter within the same category for EWG VERIFIED™ certified products (rating 1-2). Find 2-3 alternative products.\n\n**Output:**\n\nFor each original product:\n*   Product Name\n*   EWG Rating\n*   EWG Detail Page Link\n*   List of Primary Risk Ingredients (if any)\n\nFor each high-concern ingredient:\n*   Ingredient Name\n*   Titles and Links of relevant PubMed literature (listed in descending order by publication year)\n*   Main Conclusions of the literature\n*   Whether Mayo Clinic lists it as contraindicated (Yes/No/Not mentioned)\n*   Mayo Clinic\'s specific recommendations\n\nFor each alternative product:\n*   Product Name\n*   EWG Rating\n*   EWG VERIFIED™ Certification Status\n*   Product Link'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    """Information extracted for each skincare product"""
    product_name: Optional[str] = None
    ewg_rating: Optional[str] = None
    ewg_detail_link: Optional[str] = None
    risk_ingredients: Optional[List[str]] = Field(default_factory=list)


class IngredientResearch(BaseModel):
    """Research information for a high-concern ingredient"""
    ingredient_name: Optional[str] = None
    pubmed_titles: Optional[List[str]] = Field(default_factory=list)
    pubmed_links: Optional[List[str]] = Field(default_factory=list)
    publication_years: Optional[List[int]] = Field(default_factory=list)
    main_conclusions: Optional[str] = None
    mayo_contraindicated: Optional[str] = None
    mayo_recommendations: Optional[str] = None


class AlternativeProduct(BaseModel):
    """Alternative product information"""
    product_name: Optional[str] = None
    ewg_rating: Optional[str] = None
    ewg_verified: Optional[bool] = None
    product_link: Optional[str] = None


class ExtractedData(BaseModel):
    """All extracted information from the answer"""
    products: Optional[List[ProductInfo]] = Field(default_factory=list)
    ingredient_research: Optional[List[IngredientResearch]] = Field(default_factory=list)
    mayo_page_link: Optional[str] = None
    alternative_products: Optional[List[AlternativeProduct]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_data() -> str:
    return """
Extract all skincare product safety information from the answer.

For products section, extract:
- product_name: name of each of the 3 original products
- ewg_rating: the EWG rating (1-10 scale)
- ewg_detail_link: link to the product detail page on EWG
- risk_ingredients: list of primary risk ingredients (if rating >3)

For ingredient_research section, extract for each high-concern ingredient:
- ingredient_name: name of the ingredient
- pubmed_titles: list of literature titles
- pubmed_links: list of PubMed article links
- publication_years: list of publication years
- main_conclusions: main conclusions about pregnancy safety
- mayo_contraindicated: "Yes", "No", or "Not mentioned"
- mayo_recommendations: Mayo Clinic's specific recommendations

For Mayo Clinic:
- mayo_page_link: link to the Mayo Clinic pregnancy skincare page

For alternative_products section, extract:
- product_name: name of alternative product
- ewg_rating: rating (should be 1-2)
- ewg_verified: whether it has EWG VERIFIED certification
- product_link: link to the product

If any field is missing, set it to null or empty list as appropriate.
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


def extract_rating(rating_text: Optional[str]) -> Optional[float]:
    if not rating_text:
        return None
    m = re.search(r'(\d+(\.\d+)?)', str(rating_text))
    if not m:
        return None
    try:
        val = float(m.group(1))
        if 1 <= val <= 10:
            return val
    except Exception:
        pass
    return None


def is_valid_ewg_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'ewg.org')


def is_valid_pubmed_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'pubmed') or ci_contains(link, 'ncbi.nlm.nih.gov')


def is_valid_mayo_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return ci_contains(link, 'mayoclinic.org')


def extract_year(year_val: Any) -> Optional[int]:
    if year_val is None:
        return None
    try:
        year = int(year_val)
        if 2020 <= year <= 2026:
            return year
    except Exception:
        pass
    return None


def check_product_names_present(products: List[ProductInfo]) -> bool:
    if not products or len(products) < 3:
        return False
    expected_keywords = [
        ['neutrogena', 'sunscreen'],
        ['cerave', 'moisturizing'],
        ['ordinary', 'niacinamide']
    ]
    found_count = 0
    for product in products:
        if not product.product_name:
            continue
        name_lower = product.product_name.lower()
        for keywords in expected_keywords:
            if all(kw in name_lower for kw in keywords):
                found_count += 1
                break
    return found_count >= 2


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
        extraction_name="skincare_safety_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 EWG Product Search Section
    ewg_products_node = evaluator.add_sequential(
        id="ewg_products_section",
        desc="EWG Skin Deep database - Product search and rating extraction",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Product search
    products_searched = check_product_names_present(data.products) if data.products else False
    evaluator.add_custom_node(
        result=bool(products_searched),
        id="ewg_product_search",
        desc="[Action Node] ewg.org:F1:A1 - Search for the three specified products in EWG Skin Deep database",
        parent=ewg_products_node,
        critical=False
    )

    # [Perception Node] ewg.org:F1:P7 - Understand EWG rating system
    ratings_present = False
    if data.products:
        valid_ratings = [extract_rating(p.ewg_rating) for p in data.products if extract_rating(p.ewg_rating) is not None]
        ratings_present = len(valid_ratings) >= 2

    evaluator.add_custom_node(
        result=bool(ratings_present),
        id="ewg_rating_understanding",
        desc="[Perception Node] ewg.org:F1:P7 - Extract and understand EWG ratings (1-10 scale) for products",
        parent=ewg_products_node,
        critical=False
    )

    # Product detail links present
    links_present = False
    if data.products:
        valid_links = [p for p in data.products if is_valid_ewg_link(p.ewg_detail_link)]
        links_present = len(valid_links) >= 2

    evaluator.add_custom_node(
        result=bool(links_present),
        id="ewg_product_links",
        desc="Record EWG product detail page links",
        parent=ewg_products_node,
        critical=False
    )

    # [Action Node] ewg.org:F3:A1 - Ingredient query for high-risk products
    ingredients_extracted = False
    if data.products:
        products_with_ingredients = [p for p in data.products if p.risk_ingredients and len(p.risk_ingredients) > 0]
        ingredients_extracted = len(products_with_ingredients) > 0

    evaluator.add_custom_node(
        result=bool(ingredients_extracted),
        id="ewg_ingredient_extraction",
        desc="[Action Node] ewg.org:F3:A1 - Extract high-concern ingredients from products with rating >3",
        parent=ewg_products_node,
        critical=False
    )

    # 3.2 PubMed Literature Search Section
    pubmed_node = evaluator.add_sequential(
        id="pubmed_section",
        desc="PubMed literature search for high-concern ingredients",
        parent=root,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Keyword search
    pubmed_search_ok = False
    if data.ingredient_research:
        ingredients_researched = [ir for ir in data.ingredient_research if ir.ingredient_name and (ir.pubmed_titles or ir.pubmed_links)]
        pubmed_search_ok = len(ingredients_researched) > 0

    evaluator.add_custom_node(
        result=bool(pubmed_search_ok),
        id="pubmed_keyword_search",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A1 - Search PubMed with ingredient names + pregnancy safety keywords",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Date range filter
    years_filtered = False
    if data.ingredient_research:
        for ir in data.ingredient_research:
            if ir.publication_years:
                valid_years = [extract_year(y) for y in ir.publication_years if extract_year(y) is not None]
                if valid_years and all(y >= 2020 for y in valid_years):
                    years_filtered = True
                    break

    evaluator.add_custom_node(
        result=bool(years_filtered),
        id="pubmed_date_filter",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F2:A3 - Filter literature published since 2020",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Article type filter
    article_type_filtered = has_any_ci(answer, ['systematic review', 'meta-analysis', 'meta analysis'])
    evaluator.add_custom_node(
        result=bool(article_type_filtered),
        id="pubmed_article_type_filter",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F6:A4 - Filter for Systematic Reviews or Meta-analyses",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F6:P1 - Identify article type labels
    article_labels_identified = article_type_filtered
    evaluator.add_custom_node(
        result=bool(article_labels_identified),
        id="pubmed_article_type_recognition",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F6:P1 - Recognize article type labels in search results",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F1:A6 - Sort by publication date
    sorting_mentioned = has_any_ci(answer, ['descending', '降序', 'publication year', 'publication date', 'recent'])
    evaluator.add_custom_node(
        result=bool(sorting_mentioned),
        id="pubmed_date_sorting",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F1:A6 - Sort results by publication date (descending)",
        parent=pubmed_node,
        critical=False
    )

    # [Action Node] pubmed.ncbi.nlm.nih.gov:F3:A9 - Access article details
    pubmed_links_present = False
    if data.ingredient_research:
        valid_pubmed_links = [ir for ir in data.ingredient_research if ir.pubmed_links and any(is_valid_pubmed_link(link) for link in ir.pubmed_links)]
        pubmed_links_present = len(valid_pubmed_links) > 0

    evaluator.add_custom_node(
        result=bool(pubmed_links_present),
        id="pubmed_article_details",
        desc="[Action Node] pubmed.ncbi.nlm.nih.gov:F3:A9 - Access article detail pages to extract information",
        parent=pubmed_node,
        critical=False
    )

    # [Perception Node] pubmed.ncbi.nlm.nih.gov:F3:P4 - Extract abstract conclusions
    conclusions_extracted = False
    if data.ingredient_research:
        with_conclusions = [ir for ir in data.ingredient_research if ir.main_conclusions and len(ir.main_conclusions.strip()) > 20]
        conclusions_extracted = len(with_conclusions) > 0

    evaluator.add_custom_node(
        result=bool(conclusions_extracted),
        id="pubmed_abstract_understanding",
        desc="[Perception Node] pubmed.ncbi.nlm.nih.gov:F3:P4 - Extract main conclusions about pregnancy safety from abstracts",
        parent=pubmed_node,
        critical=False
    )

    # 3.3 Mayo Clinic Guidelines Section
    mayo_node = evaluator.add_sequential(
        id="mayo_clinic_section",
        desc="Mayo Clinic pregnancy skincare guidelines verification",
        parent=root,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A2 - Navigate to health information
    mayo_link_ok = is_valid_mayo_link(data.mayo_page_link)
    mayo_mentioned = has_any_ci(answer, ['mayo clinic'])

    evaluator.add_custom_node(
        result=bool(mayo_link_ok or mayo_mentioned),
        id="mayo_navigation",
        desc="[Action Node] mayoclinic.org:F2:A2 - Navigate to Mayo Clinic and search for pregnancy skin care guidelines",
        parent=mayo_node,
        critical=False
    )

    # [Perception Node] mayoclinic.org:F2:P2 - Understand contraindication list
    contraindication_checked = False
    if data.ingredient_research:
        checked_ingredients = [ir for ir in data.ingredient_research if ir.mayo_contraindicated and ir.mayo_contraindicated.lower() in ['yes', 'no', 'not mentioned', '是', '否', '未提及']]
        contraindication_checked = len(checked_ingredients) > 0

    evaluator.add_custom_node(
        result=bool(contraindication_checked),
        id="mayo_contraindication_understanding",
        desc="[Perception Node] mayoclinic.org:F2:P2 - Check if high-concern ingredients are on Mayo Clinic's restricted list",
        parent=mayo_node,
        critical=False
    )

    # [Action Node] mayoclinic.org:F2:A4 - Navigate between content sections
    mayo_recommendations_present = False
    if data.ingredient_research:
        with_recommendations = [ir for ir in data.ingredient_research if ir.mayo_recommendations and len(ir.mayo_recommendations.strip()) > 10]
        mayo_recommendations_present = len(with_recommendations) > 0

    evaluator.add_custom_node(
        result=bool(mayo_recommendations_present),
        id="mayo_recommendations_extraction",
        desc="[Action Node] mayoclinic.org:F2:A4 - Extract Mayo Clinic's specific recommendations for ingredients",
        parent=mayo_node,
        critical=False
    )

    # 3.4 EWG Alternative Products Section
    ewg_alternatives_node = evaluator.add_sequential(
        id="ewg_alternatives_section",
        desc="EWG alternative product recommendations",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F2:A3 - Product category navigation
    category_mentioned = has_any_ci(answer, ['category', 'sunscreen', 'moisturizer', 'same category', '类别'])
    evaluator.add_custom_node(
        result=bool(category_mentioned),
        id="ewg_category_navigation",
        desc="[Action Node] ewg.org:F2:A3 - Navigate to product categories for finding alternatives",
        parent=ewg_alternatives_node,
        critical=False
    )

    # [Action Node] ewg.org:F4:A6 - EWG VERIFIED filter
    alternatives_ok = False
    verified_count = 0
    if data.alternative_products:
        alternatives_ok = len(data.alternative_products) >= 2
        verified_count = sum(1 for alt in data.alternative_products if alt.ewg_verified is True)

    evaluator.add_custom_node(
        result=bool(verified_count > 0),
        id="ewg_verified_filter",
        desc="[Action Node] ewg.org:F4:A6 - Filter for EWG VERIFIED certified products (rating 1-2)",
        parent=ewg_alternatives_node,
        critical=False
    )

    # Alternative products extraction
    evaluator.add_custom_node(
        result=bool(alternatives_ok),
        id="alternative_products_found",
        desc="Find 2-3 alternative products with EWG VERIFIED certification",
        parent=ewg_alternatives_node,
        critical=False
    )

    # Alternative product ratings check
    alt_ratings_ok = False
    if data.alternative_products:
        low_rated = [alt for alt in data.alternative_products if extract_rating(alt.ewg_rating) and extract_rating(alt.ewg_rating) <= 2]
        alt_ratings_ok = len(low_rated) >= 2

    evaluator.add_custom_node(
        result=bool(alt_ratings_ok),
        id="alternative_ratings_verified",
        desc="Verify alternative products have low ratings (1-2)",
        parent=ewg_alternatives_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
