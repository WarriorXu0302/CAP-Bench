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
TASK_ID = "task-747160"
TASK_DESCRIPTION = 'I’m looking for safe facial moisturizers for sensitive skin during pregnancy and want to verify that the ingredients are truly safe and reliable.\n\nFirst, go to the EWG website. Under the **Face & Body** section, find the **Facial Moisturizer** category, then filter for products with an **EWG score of 1–2** (green safety range) and an **EWG VERIFIED** badge. Try to find and record up to **5 products**.\n\nNext, search for these same products on **Sephora** and collect, for each one: user rating, price, whether it has the **Clean at Sephora** label, and especially whether reviews mention feedback from users with sensitive skin or who are pregnant.\n\nThen search for the same products on **Target**, compare prices and **Target Circle** member offers, and determine which platform is more cost-effective.\n\nFinally, visit each product’s official brand website and locate either the certification page or product detail page to confirm whether the site displays certifications such as **EWG VERIFIED**, **USDA Organic**, or **Cruelty-Free**.\n\nFor each product, output:\n- Full product name  \n- EWG score (specific number: 1 or 2)  \n- EWG VERIFIED certification (Yes/No)  \n- Sephora rating and price  \n- Clean at Sephora label (Yes/No)  \n- Target price and Target Circle offer details  \n- Certification labels shown on the brand’s official website  \n- Detail page links from each platform (**one each from EWG / Sephora / Target / brand official site**)\n\nIf fewer than 5 products meet the EWG criteria, proceed with the actual number found. If a product cannot be found on Sephora or Target, note that and continue collecting the remaining information for other products.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    """Information for a single product extracted from the answer"""
    product_name: Optional[str] = None
    ewg_score: Optional[str] = None
    ewg_verified: Optional[str] = None
    sephora_rating: Optional[str] = None
    sephora_price: Optional[str] = None
    clean_at_sephora: Optional[str] = None
    sensitive_skin_reviews: Optional[str] = None
    target_price: Optional[str] = None
    target_circle_offer: Optional[str] = None
    brand_certifications: Optional[str] = None
    ewg_link: Optional[str] = None
    sephora_link: Optional[str] = None
    target_link: Optional[str] = None
    brand_link: Optional[str] = None


class AllProductsData(BaseModel):
    """Container for all products found"""
    products: List[ProductInfo] = Field(default_factory=list)
    total_count: Optional[int] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_products_from_answer() -> str:
    return """
Extract all facial moisturizer products mentioned in the answer along with their details.

For each product, extract:
- product_name: the full product name
- ewg_score: the EWG score (should be 1 or 2)
- ewg_verified: whether it has EWG VERIFIED certification (Yes/No or similar)
- sephora_rating: the user rating on Sephora
- sephora_price: the price on Sephora
- clean_at_sephora: whether it has Clean at Sephora label (Yes/No or similar)
- sensitive_skin_reviews: any mention of sensitive skin or pregnancy reviews
- target_price: the price on Target
- target_circle_offer: Target Circle member offers or deals
- brand_certifications: certifications displayed on brand website (EWG VERIFIED, USDA Organic, Cruelty-Free, etc.)
- ewg_link: link to EWG product page
- sephora_link: link to Sephora product page
- target_link: link to Target product page
- brand_link: link to brand official website product page

Also extract:
- total_count: total number of products found

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


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0])
    except Exception:
        return None


def is_yes_like(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['yes', 'true', '✓', '✔'])


def is_no_like(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['no', 'false', 'not found', 'n/a', 'unavailable'])


def looks_like_facial_moisturizer(name: Optional[str]) -> bool:
    if not name:
        return False
    return has_any_ci(name, ['moisturizer', 'cream', 'lotion', 'hydrat', 'facial'])


def looks_like_ewg_score_1_or_2(score: Optional[str]) -> bool:
    if not score:
        return False
    num = extract_number(score)
    return num is not None and (num == 1 or num == 2)


def looks_like_rating(rating: Optional[str]) -> bool:
    if not rating:
        return False
    num = extract_number(rating)
    return num is not None and 0 <= num <= 5


def looks_like_price(price: Optional[str]) -> bool:
    if not price:
        return False
    return has_any_ci(price, ['$', 'usd', 'dollar']) or contains_digits(price)


def looks_like_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return has_any_ci(url, ['http://', 'https://', 'www.', '.com', '.org'])


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
    products_data = await evaluator.extract(
        prompt=prompt_extract_products_from_answer(),
        template_class=AllProductsData,
        extraction_name="all_products_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 EWG section
    ewg_node = evaluator.add_sequential(
        id="ewg_section",
        desc="EWG website - Find facial moisturizers with score 1-2 and EWG VERIFIED badge",
        parent=root,
        critical=False
    )

    # [Action Node] ewg.org:F2:A3 - Navigate to Facial Moisturizer category
    ewg_category_nav_ok = (has_any_ci(answer, ['ewg']) and
                           has_any_ci(answer, ['face', 'facial']) and
                           has_any_ci(answer, ['moisturizer']))
    evaluator.add_custom_node(
        result=bool(ewg_category_nav_ok),
        id="ewg_category_navigation",
        desc="[Action Node] ewg.org:F2:A3 - Navigate to Facial Moisturizer category under Face & Body",
        parent=ewg_node,
        critical=False
    )

    # [Perception Node] ewg.org:F2:P3 - Understand category structure
    ewg_category_structure_ok = (has_any_ci(answer, ['face & body', 'face and body']) or
                                 (has_any_ci(answer, ['category', 'section']) and has_any_ci(answer, ['facial moisturizer'])))
    evaluator.add_custom_node(
        result=bool(ewg_category_structure_ok),
        id="ewg_category_structure",
        desc="[Perception Node] ewg.org:F2:P3 - Understand Face & Body category structure",
        parent=ewg_node,
        critical=False
    )

    # [Action Node] ewg.org:F1:A1 - Search and filter products
    ewg_filter_ok = (has_any_ci(answer, ['score', 'rating', '1-2', '1–2']) and
                     has_any_ci(answer, ['ewg verified', 'verified']))
    evaluator.add_custom_node(
        result=bool(ewg_filter_ok),
        id="ewg_search_filter",
        desc="[Action Node] ewg.org:F1:A1 - Filter for products with EWG score 1-2 and EWG VERIFIED badge",
        parent=ewg_node,
        critical=False
    )

    # Check products found
    products = products_data.products if products_data and products_data.products else []
    has_products = len(products) > 0

    evaluator.add_custom_node(
        result=bool(has_products),
        id="ewg_products_found",
        desc="At least one product was found meeting EWG criteria",
        parent=ewg_node,
        critical=False
    )

    # Validate products section
    if has_products:
        products_validation_node = evaluator.add_parallel(
            id="products_validation",
            desc="Validate extracted product information",
            parent=root,
            critical=False
        )

        # Check product names are facial moisturizers
        product_names_ok = all(looks_like_facial_moisturizer(p.product_name) for p in products if p.product_name)
        evaluator.add_custom_node(
            result=bool(product_names_ok and len([p for p in products if p.product_name]) > 0),
            id="product_names_valid",
            desc="Product names appear to be facial moisturizers",
            parent=products_validation_node,
            critical=False
        )

        # Check EWG scores are 1 or 2
        ewg_scores_ok = all(looks_like_ewg_score_1_or_2(p.ewg_score) for p in products if p.ewg_score)
        evaluator.add_custom_node(
            result=bool(ewg_scores_ok and len([p for p in products if p.ewg_score]) > 0),
            id="ewg_scores_valid",
            desc="EWG scores are 1 or 2 as required",
            parent=products_validation_node,
            critical=False
        )

        # [Perception Node] ewg.org:F4:A6 - EWG VERIFIED certification recognition
        ewg_verified_ok = len([p for p in products if p.ewg_verified and is_yes_like(p.ewg_verified)]) > 0
        evaluator.add_custom_node(
            result=bool(ewg_verified_ok),
            id="ewg_verified_recognition",
            desc="[Action Node] ewg.org:F4:A6 - Recognize EWG VERIFIED certification on products",
            parent=products_validation_node,
            critical=False
        )

    # 3.2 Sephora section
    sephora_node = evaluator.add_sequential(
        id="sephora_section",
        desc="Sephora - Search products and collect ratings, prices, and Clean label",
        parent=root,
        critical=False
    )

    # [Action Node] sephora.com:F1:A1 - Search products on Sephora
    sephora_search_ok = has_any_ci(answer, ['sephora'])
    evaluator.add_custom_node(
        result=bool(sephora_search_ok),
        id="sephora_search",
        desc="[Action Node] sephora.com:F1:A1 - Search for products on Sephora",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F1:A3 - Click product cards to view details
    sephora_click_ok = len([p for p in products if p.sephora_link or p.sephora_rating or p.sephora_price]) > 0
    evaluator.add_custom_node(
        result=bool(sephora_click_ok),
        id="sephora_product_click",
        desc="[Action Node] sephora.com:F1:A3 - Access product detail pages on Sephora",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P2 - Extract ratings
    sephora_ratings_ok = len([p for p in products if p.sephora_rating and looks_like_rating(p.sephora_rating)]) > 0
    evaluator.add_custom_node(
        result=bool(sephora_ratings_ok),
        id="sephora_ratings_extraction",
        desc="[Perception Node] sephora.com:F2:P2 - Extract user ratings from Sephora",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F2:P7 - Extract prices
    sephora_prices_ok = len([p for p in products if p.sephora_price and looks_like_price(p.sephora_price)]) > 0
    evaluator.add_custom_node(
        result=bool(sephora_prices_ok),
        id="sephora_prices_extraction",
        desc="[Perception Node] sephora.com:F2:P7 - Extract price information from Sephora",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F1:P1 - Recognize Clean at Sephora label
    clean_label_ok = len([p for p in products if p.clean_at_sephora]) > 0
    evaluator.add_custom_node(
        result=bool(clean_label_ok),
        id="sephora_clean_label",
        desc="[Perception Node] sephora.com:F1:P1 - Recognize Clean at Sephora label",
        parent=sephora_node,
        critical=False
    )

    # [Action Node] sephora.com:F9:A7 - Filter reviews by skin type
    reviews_filter_ok = len([p for p in products if p.sensitive_skin_reviews]) > 0
    evaluator.add_custom_node(
        result=bool(reviews_filter_ok),
        id="sephora_reviews_filter",
        desc="[Action Node] sephora.com:F9:A7 - Filter or identify reviews from sensitive skin or pregnant users",
        parent=sephora_node,
        critical=False
    )

    # [Perception Node] sephora.com:F9:P5 - Understand review content
    reviews_content_ok = len([p for p in products if p.sensitive_skin_reviews and
                              (has_any_ci(p.sensitive_skin_reviews, ['sensitive', 'pregnancy', 'pregnant']))])  > 0
    evaluator.add_custom_node(
        result=bool(reviews_content_ok),
        id="sephora_reviews_understanding",
        desc="[Perception Node] sephora.com:F9:P5 - Extract relevant feedback from sensitive skin or pregnant users",
        parent=sephora_node,
        critical=False
    )

    # 3.3 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target - Search products, compare prices and Target Circle offers",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Search products on Target
    target_search_ok = has_any_ci(answer, ['target'])
    evaluator.add_custom_node(
        result=bool(target_search_ok),
        id="target_search",
        desc="[Action Node] target.com:F1:A1 - Search for products on Target",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product cards
    target_click_ok = len([p for p in products if p.target_link or p.target_price]) > 0
    evaluator.add_custom_node(
        result=bool(target_click_ok),
        id="target_product_click",
        desc="[Action Node] target.com:F1:A21 - Access product detail pages on Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P6 - Extract prices
    target_prices_ok = len([p for p in products if p.target_price and looks_like_price(p.target_price)]) > 0
    evaluator.add_custom_node(
        result=bool(target_prices_ok),
        id="target_prices_extraction",
        desc="[Perception Node] target.com:F2:P6 - Extract price information from Target",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P5 - Recognize Target Circle offers
    circle_offers_ok = len([p for p in products if p.target_circle_offer]) > 0
    evaluator.add_custom_node(
        result=bool(circle_offers_ok),
        id="target_circle_offers",
        desc="[Perception Node] target.com:F2:P5 - Recognize Target Circle member offers",
        parent=target_node,
        critical=False
    )

    # Price comparison mentioned
    price_comparison_ok = has_any_ci(answer, ['compare', 'comparison', 'cheaper', 'cost-effective', 'better deal'])
    evaluator.add_custom_node(
        result=bool(price_comparison_ok),
        id="price_comparison",
        desc="Compare prices between Sephora and Target to determine cost-effectiveness",
        parent=target_node,
        critical=False
    )

    # 3.4 Brand website section
    brand_node = evaluator.add_sequential(
        id="brand_website_section",
        desc="Brand official websites - Verify certifications",
        parent=root,
        critical=False
    )

    brand_visit_ok = len([p for p in products if p.brand_link]) > 0
    evaluator.add_custom_node(
        result=bool(brand_visit_ok),
        id="brand_website_visit",
        desc="Visit brand official websites for certification verification",
        parent=brand_node,
        critical=False
    )

    brand_certs_ok = len([p for p in products if p.brand_certifications and
                          has_any_ci(p.brand_certifications, ['ewg', 'usda', 'organic', 'cruelty-free', 'certified'])]) > 0
    evaluator.add_custom_node(
        result=bool(brand_certs_ok),
        id="brand_certifications_found",
        desc="Identify certifications (EWG VERIFIED, USDA Organic, Cruelty-Free) on brand websites",
        parent=brand_node,
        critical=False
    )

    # 3.5 Output completeness section
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Check completeness of output information",
        parent=root,
        critical=False
    )

    # Links provided
    ewg_links_ok = len([p for p in products if p.ewg_link and looks_like_url(p.ewg_link)]) > 0
    evaluator.add_custom_node(
        result=bool(ewg_links_ok),
        id="ewg_links_provided",
        desc="EWG product detail page links provided",
        parent=output_node,
        critical=False
    )

    sephora_links_ok = len([p for p in products if p.sephora_link and looks_like_url(p.sephora_link)]) > 0
    evaluator.add_custom_node(
        result=bool(sephora_links_ok),
        id="sephora_links_provided",
        desc="Sephora product detail page links provided",
        parent=output_node,
        critical=False
    )

    target_links_ok = len([p for p in products if p.target_link and looks_like_url(p.target_link)]) > 0
    evaluator.add_custom_node(
        result=bool(target_links_ok),
        id="target_links_provided",
        desc="Target product detail page links provided",
        parent=output_node,
        critical=False
    )

    brand_links_ok = len([p for p in products if p.brand_link and looks_like_url(p.brand_link)]) > 0
    evaluator.add_custom_node(
        result=bool(brand_links_ok),
        id="brand_links_provided",
        desc="Brand official website links provided",
        parent=output_node,
        critical=False
    )

    # Structured output
    structured_output_ok = (has_any_ci(answer, ['product name', 'ewg score']) and
                            has_any_ci(answer, ['sephora', 'rating', 'price']) and
                            has_any_ci(answer, ['target']))
    evaluator.add_custom_node(
        result=bool(structured_output_ok),
        id="structured_output_format",
        desc="Output follows requested format with all required fields",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
