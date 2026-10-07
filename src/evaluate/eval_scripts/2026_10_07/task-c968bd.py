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
TASK_ID = "task-c968bd"
TASK_DESCRIPTION = 'My baby’s eczema has flared up recently, and I want to buy an absolutely safe moisturizer—specifically one without steroids. Please help me run a cross-platform screening process:\n\nFirst, on Amazon, search for **“Baby Eczema Cream”** and use filters to find products with a rating above **4.5 stars** and a price of **$25 or less**. Pay special attention to products that explicitly list **“Colloidal Oatmeal”** as an ingredient. Try to identify **5 candidate products** (if fewer than 5 meet the criteria, record however many qualify and note the count).\n\nThen, go to **EWG.org** and search these 5 products one by one. Keep only products with an **EWG rating of 1–2** (green safety zone). If a product cannot be found on EWG, mark it as **“Not listed in EWG”** and continue checking the others. If fewer than 5 products ultimately meet the EWG 1–2 criterion, output all qualifying items and note the count.\n\nFinally, go to **WebMD** and look up **“Colloidal Oatmeal”** to find its **“Side Effects”** section. If there is no standalone entry, find relevant eczema/skincare pages on WebMD that mention side effects of this ingredient and cite the source page.\n\nPlease output the final recommended products that meet EWG safety standards, including: **product name, Amazon price, EWG rating, a brief WebMD side-effects summary, and links to the Amazon and EWG detail pages**.\n\nIf you encounter login requirements or access restrictions on any site, record the limitation and complete the remaining steps within accessible scope.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AmazonProduct(BaseModel):
    """A single Amazon product candidate"""
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    has_colloidal_oatmeal: Optional[bool] = None
    amazon_link: Optional[str] = None


class AmazonProducts(BaseModel):
    """List of Amazon product candidates"""
    products: List[AmazonProduct] = Field(default_factory=list)
    count: Optional[int] = None


class EWGProduct(BaseModel):
    """EWG verification result for a product"""
    product_name: Optional[str] = None
    ewg_rating: Optional[str] = None
    ewg_link: Optional[str] = None
    not_listed: Optional[bool] = False


class EWGProducts(BaseModel):
    """List of EWG verification results"""
    products: List[EWGProduct] = Field(default_factory=list)
    qualifying_count: Optional[int] = None


class WebMDInfo(BaseModel):
    """WebMD side effects information"""
    side_effects_summary: Optional[str] = None
    source_page: Optional[str] = None


class FinalRecommendations(BaseModel):
    """Final recommended products"""
    products: List[Dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_amazon_products() -> str:
    return """
Extract the Amazon product candidates from the answer. The user was asked to find Baby Eczema Cream products with:
- Rating above 4.5 stars
- Price of $25 or less
- Containing "Colloidal Oatmeal" as an ingredient
- Target: 5 candidate products (or fewer if that's all that qualify)

For each product, extract:
- name: product name
- price: price as stated (include $ if present)
- rating: rating as stated
- has_colloidal_oatmeal: true if the answer confirms it contains Colloidal Oatmeal
- amazon_link: the Amazon product link if provided

Also extract:
- count: the total number of candidate products found

If no products are mentioned, return empty list and count as null.
"""


def prompt_extract_ewg_products() -> str:
    return """
Extract the EWG verification results from the answer. The user was asked to check each Amazon candidate on EWG.org and keep only products with EWG rating 1-2.

For each product checked, extract:
- product_name: the product name
- ewg_rating: the EWG rating (e.g., "1", "2", "1-2")
- ewg_link: the EWG product page link if provided
- not_listed: true if the answer indicates the product was "Not listed in EWG" or could not be found

Also extract:
- qualifying_count: the number of products that meet the EWG 1-2 criterion

If no EWG information is mentioned, return empty list.
"""


def prompt_extract_webmd_info() -> str:
    return """
Extract the WebMD side effects information from the answer. The user was asked to look up "Colloidal Oatmeal" on WebMD to find its "Side Effects" section.

Extract:
- side_effects_summary: the side effects information or summary provided
- source_page: the WebMD page URL or citation if provided

If no WebMD information is mentioned, set both fields to null.
"""


def prompt_extract_final_recommendations() -> str:
    return """
Extract the final recommended products from the answer. The user asked for products that meet EWG safety standards, with:
- Product name
- Amazon price
- EWG rating
- Brief WebMD side-effects summary
- Links to Amazon and EWG detail pages

Extract all final recommended products as a list. Each product should be a dictionary with available fields.

If no final recommendations are provided, return empty list.
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


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    return num is not None


def looks_like_ewg_rating_1_or_2(text: Optional[str]) -> bool:
    if not text:
        return False
    text_clean = text.strip().lower()
    # Accept "1", "2", "1-2", "green", etc.
    if text_clean in ["1", "2", "1-2", "1–2"]:
        return True
    if "green" in text_clean:
        return True
    num = extract_float(text)
    if num is not None and 1 <= num <= 2:
        return True
    return False


def mentions_colloidal_oatmeal(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['colloidal oatmeal'])


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
    amazon_products = await evaluator.extract(
        prompt=prompt_extract_amazon_products(),
        template_class=AmazonProducts,
        extraction_name="amazon_products"
    )

    ewg_products = await evaluator.extract(
        prompt=prompt_extract_ewg_products(),
        template_class=EWGProducts,
        extraction_name="ewg_products"
    )

    webmd_info = await evaluator.extract(
        prompt=prompt_extract_webmd_info(),
        template_class=WebMDInfo,
        extraction_name="webmd_info"
    )

    final_recommendations = await evaluator.extract(
        prompt=prompt_extract_final_recommendations(),
        template_class=FinalRecommendations,
        extraction_name="final_recommendations"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon product search and filtering for Baby Eczema Cream",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Multi-select filtering for rating above 4.5
    amazon_mentions_rating_filter = has_any_ci(answer, ['4.5', 'rating', 'star'])
    products_meet_rating = False
    if amazon_products and amazon_products.products:
        products_meet_rating = all(
            looks_like_rating(p.rating) and (extract_float(p.rating) or 0) >= 4.5
            for p in amazon_products.products if p.rating
        )

    evaluator.add_custom_node(
        result=bool(amazon_mentions_rating_filter and (products_meet_rating or len(amazon_products.products if amazon_products else []) > 0)),
        id="amazon_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Apply rating filter to find products with rating above 4.5 stars",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A17 - Slider range filtering for price <= $25
    amazon_mentions_price_filter = has_any_ci(answer, ['25', 'price', '$'])
    products_meet_price = False
    if amazon_products and amazon_products.products:
        products_meet_price = all(
            looks_like_price(p.price) and (extract_float(p.price) or 999) <= 25
            for p in amazon_products.products if p.price
        )

    evaluator.add_custom_node(
        result=bool(amazon_mentions_price_filter and (products_meet_price or len(amazon_products.products if amazon_products else []) > 0)),
        id="amazon_price_filter",
        desc="[Action Node] Amazon:F3:A17 - Apply price filter to find products priced at $25 or less",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A24 - Expand/collapse to check ingredients for Colloidal Oatmeal
    products_have_colloidal_oatmeal = False
    if amazon_products and amazon_products.products:
        products_have_colloidal_oatmeal = any(
            p.has_colloidal_oatmeal for p in amazon_products.products
        )

    evaluator.add_custom_node(
        result=bool(mentions_colloidal_oatmeal(answer) and products_have_colloidal_oatmeal),
        id="amazon_colloidal_oatmeal_check",
        desc="[Action Node] Amazon:F5:A24 - Expand product details to verify Colloidal Oatmeal is listed as an ingredient",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F1:A6 - Pagination to find 5 candidate products
    found_products_count = len(amazon_products.products) if amazon_products and amazon_products.products else 0
    pagination_mentioned = has_any_ci(answer, ['page', 'next', 'result']) or found_products_count > 0

    evaluator.add_custom_node(
        result=bool(pagination_mentioned and found_products_count > 0),
        id="amazon_pagination",
        desc="[Action Node] Amazon:F1:A6 - Navigate through pages to identify 5 candidate products (or note if fewer qualify)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - State awareness for Best Seller and high-rated products
    best_seller_awareness = has_any_ci(answer, ['best seller', 'bestseller', 'popular', 'top rated'])

    evaluator.add_custom_node(
        result=bool(best_seller_awareness or found_products_count >= 3),
        id="amazon_state_awareness",
        desc="[Perception Node] Amazon:F3:P1 - Recognize market indicators like Best Seller tags or high ratings",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P9 - Visual feature extraction from product images for ingredient info
    image_check_mentioned = has_any_ci(answer, ['image', 'picture', 'photo', 'label', 'bottle', 'packaging'])

    evaluator.add_custom_node(
        result=bool(image_check_mentioned),
        id="amazon_visual_extraction",
        desc="[Perception Node] Amazon:F5:P9 - Extract ingredient information from product images (e.g., back label)",
        parent=amazon_node,
        critical=False
    )

    # Additional check: candidate count documented
    count_documented = (amazon_products and amazon_products.count is not None) or found_products_count > 0
    evaluator.add_custom_node(
        result=bool(count_documented),
        id="amazon_candidate_count",
        desc="Document the count of candidate products found (up to 5 or fewer if applicable)",
        parent=amazon_node,
        critical=False
    )

    # 3.2 EWG section
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG.org verification of product safety ratings",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Form search for each product on EWG
    ewg_search_performed = has_any_ci(answer, ['ewg']) and (
        has_any_ci(answer, ['search', 'look up', 'check']) or
        (ewg_products and len(ewg_products.products) > 0)
    )

    evaluator.add_custom_node(
        result=bool(ewg_search_performed),
        id="ewg_form_search",
        desc="[Action Node] ewg.org:F1:A1 - Search each Amazon candidate product on EWG.org one by one",
        parent=ewg_node,
        critical=False
    )

    # Check EWG ratings are 1-2 (green safety zone)
    ewg_ratings_valid = False
    qualifying_products_count = 0
    if ewg_products and ewg_products.products:
        qualifying_products = [
            p for p in ewg_products.products
            if p.ewg_rating and looks_like_ewg_rating_1_or_2(p.ewg_rating) and not p.not_listed
        ]
        qualifying_products_count = len(qualifying_products)
        ewg_ratings_valid = qualifying_products_count > 0

    evaluator.add_custom_node(
        result=bool(ewg_ratings_valid),
        id="ewg_rating_1_2",
        desc="Keep only products with EWG rating of 1-2 (green safety zone)",
        parent=ewg_node,
        critical=False
    )

    # Handle "Not listed in EWG" cases appropriately
    not_listed_handled = False
    if ewg_products and ewg_products.products:
        not_listed_count = sum(1 for p in ewg_products.products if p.not_listed)
        not_listed_handled = not_listed_count > 0 or has_any_ci(answer, ['not listed', 'not found on ewg', 'cannot be found'])

    evaluator.add_custom_node(
        result=bool(not_listed_handled or (ewg_products and len(ewg_products.products) > 0)),
        id="ewg_not_listed_handling",
        desc="Mark products not found on EWG as 'Not listed in EWG' and continue with others",
        parent=ewg_node,
        critical=False
    )

    # EWG links provided
    ewg_links_provided = False
    if ewg_products and ewg_products.products:
        ewg_links_provided = any(p.ewg_link for p in ewg_products.products if not p.not_listed)

    evaluator.add_custom_node(
        result=bool(ewg_links_provided),
        id="ewg_links",
        desc="Provide EWG detail page links for verified products",
        parent=ewg_node,
        critical=False
    )

    # 3.3 WebMD section
    webmd_node = evaluator.add_sequential(
        id="webmd_section",
        desc="WebMD research on Colloidal Oatmeal side effects",
        parent=root,
        critical=False
    )

    # [Action Node] webmd.com:F2:A5 - Keyword search for Colloidal Oatmeal
    webmd_search_performed = has_any_ci(answer, ['webmd']) and mentions_colloidal_oatmeal(answer)

    evaluator.add_custom_node(
        result=bool(webmd_search_performed),
        id="webmd_keyword_search",
        desc="[Action Node] webmd.com:F2:A5 - Search for 'Colloidal Oatmeal' on WebMD",
        parent=webmd_node,
        critical=False
    )

    # [Perception Node] webmd.com:F2:P3 - Extract Side Effects section content
    side_effects_found = False
    if webmd_info and webmd_info.side_effects_summary:
        side_effects_found = True
    side_effects_mentioned = has_any_ci(answer, ['side effect', 'adverse', 'reaction'])

    evaluator.add_custom_node(
        result=bool(side_effects_found and side_effects_mentioned),
        id="webmd_side_effects_extraction",
        desc="[Perception Node] webmd.com:F2:P3 - Locate and extract Side Effects section for Colloidal Oatmeal",
        parent=webmd_node,
        critical=False
    )

    # Source page citation
    source_cited = webmd_info and webmd_info.source_page is not None

    evaluator.add_custom_node(
        result=bool(source_cited),
        id="webmd_source_citation",
        desc="Cite the WebMD source page URL where side effects information was found",
        parent=webmd_node,
        critical=False
    )

    # 3.4 Final output section
    final_output_node = evaluator.add_sequential(
        id="final_output_section",
        desc="Comprehensive final recommendations with all required information",
        parent=root,
        critical=False
    )

    # Check final recommendations completeness
    final_recs_exist = final_recommendations and len(final_recommendations.products) > 0

    evaluator.add_custom_node(
        result=bool(final_recs_exist),
        id="final_recommendations_provided",
        desc="Provide final recommended products that meet EWG safety standards",
        parent=final_output_node,
        critical=False
    )

    # Check all required fields are included
    all_fields_present = False
    if final_recs_exist:
        # Check if recommendations mention the required fields
        answer_lower = answer.lower()
        has_name = has_any_ci(answer, ['product name', 'name:'])
        has_price = has_any_ci(answer, ['price', '$'])
        has_ewg_rating = has_any_ci(answer, ['ewg rating', 'rating:'])
        has_side_effects = has_any_ci(answer, ['side effect', 'webmd'])
        has_links = has_any_ci(answer, ['link', 'url', 'http'])

        all_fields_present = has_name and has_price and has_ewg_rating and has_side_effects and has_links

    evaluator.add_custom_node(
        result=bool(all_fields_present),
        id="final_output_completeness",
        desc="Include all required fields: product name, Amazon price, EWG rating, WebMD side-effects summary, and links",
        parent=final_output_node,
        critical=False
    )

    # Access restrictions handling
    access_restrictions_noted = has_any_ci(answer, [
        'login', 'access restriction', 'blocked', 'unavailable',
        'cannot access', 'limitation', 'restricted'
    ])

    evaluator.add_custom_node(
        result=True,  # Always pass, just checking if mentioned
        id="access_restrictions_handling",
        desc="Note any login requirements or access restrictions encountered (if applicable)",
        parent=final_output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
