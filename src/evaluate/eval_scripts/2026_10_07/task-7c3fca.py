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
TASK_ID = "task-7c3fca"
TASK_DESCRIPTION = 'I want to buy three types of upgrades for my 2025 Ford Bronco: a roof rack, a winch, and all-terrain tires. First, go to Car and Driver and find the review article for the 2025 Ford Bronco to see which upgrade brands and specific models they recommend. If that review does not provide specific brands/models for all three upgrade categories, continue searching within Car and Driver for Bronco upgrade guides or related articles to fill in the missing recommendations.\n\nThen take those recommended models and search for them on Amazon, eBay, Target, and Best Buy to compare prices. Apply these filters: rating above 4 stars, in stock, and standard shipping or faster delivery available. For each upgrade category, identify the lowest-priced option across the four platforms; however, if the lowest-priced option has a rating below 4.5 stars or fewer than 50 reviews, choose the second-lowest-priced option instead.\n\nFinally, output for each upgrade category: recommended brand and model, the most cost-effective platform to buy from, that platform’s price, rating, number of reviews, stock status, shipping method, and product detail page link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UpgradeRecommendations(BaseModel):
    """Recommended upgrade brands and models from Car and Driver"""
    roof_rack_brand_model: Optional[str] = None
    winch_brand_model: Optional[str] = None
    all_terrain_tires_brand_model: Optional[str] = None


class UpgradePurchaseInfo(BaseModel):
    """Purchase information for one upgrade category"""
    category: Optional[str] = None
    brand_model: Optional[str] = None
    platform: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    num_reviews: Optional[str] = None
    stock_status: Optional[str] = None
    shipping_method: Optional[str] = None
    product_link: Optional[str] = None


class AllUpgrades(BaseModel):
    """All three upgrade purchase details"""
    roof_rack: Optional[UpgradePurchaseInfo] = None
    winch: Optional[UpgradePurchaseInfo] = None
    all_terrain_tires: Optional[UpgradePurchaseInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_recommendations() -> str:
    return """
From the answer, extract the recommended brands and specific models for the three upgrade categories from Car and Driver:

- roof_rack_brand_model: the recommended roof rack brand and model
- winch_brand_model: the recommended winch brand and model
- all_terrain_tires_brand_model: the recommended all-terrain tires brand and model

If any field is missing, set it to null.
"""


def prompt_extract_purchase_info() -> str:
    return """
From the answer, extract the final purchase recommendation for each of the three upgrade categories (roof rack, winch, all-terrain tires).

For each category, extract:
- category: the upgrade type
- brand_model: the recommended brand and model
- platform: the e-commerce platform (Amazon, eBay, Target, or Best Buy)
- price: the price as stated
- rating: the product rating
- num_reviews: the number of reviews
- stock_status: the stock/availability status
- shipping_method: the shipping method
- product_link: the product detail page URL

If any field is missing for a category, set it to null.
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


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    return contains_digits(text) and (ci_contains(text, '$') or ci_contains(text, 'usd'))


def looks_like_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return text.startswith('http://') or text.startswith('https://')


def has_platform_name(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['amazon', 'ebay', 'target', 'best buy'])


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
    recommendations = await evaluator.extract(
        prompt=prompt_extract_recommendations(),
        template_class=UpgradeRecommendations,
        extraction_name="upgrade_recommendations"
    )

    purchase_info = await evaluator.extract(
        prompt=prompt_extract_purchase_info(),
        template_class=AllUpgrades,
        extraction_name="all_upgrades_purchase"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Car and Driver section
    caranddriver_node = evaluator.add_sequential(
        id="caranddriver_section",
        desc="Car and Driver review and upgrade recommendations for 2025 Ford Bronco",
        parent=root,
        critical=False
    )

    # [Action Node] caranddriver.com:F1:A11 - Click review article card
    caranddriver_action_click = has_any_ci(answer, ['car and driver']) and has_any_ci(answer, ['2025 ford bronco', 'ford bronco 2025', 'bronco'])
    evaluator.add_custom_node(
        result=bool(caranddriver_action_click),
        id="caranddriver_action_click_article",
        desc="[Action Node] caranddriver.com:F1:A11 - Navigate to and click the 2025 Ford Bronco review article on Car and Driver",
        parent=caranddriver_node,
        critical=False
    )

    # [Perception Node] caranddriver.com:F1:P2 - Understand article content and match target vehicle
    article_content_ok = has_any_ci(answer, ['2025 ford bronco', 'ford bronco']) and (
        has_any_ci(answer, ['review', 'article', 'evaluation']) or
        has_any_ci(answer, ['roof rack', 'winch', 'tire', 'upgrade', 'accessory'])
    )
    evaluator.add_custom_node(
        result=bool(article_content_ok),
        id="caranddriver_perception_article_content",
        desc="[Perception Node] caranddriver.com:F1:P2 - Confirm the article is indeed a 2025 Ford Bronco review with upgrade recommendations",
        parent=caranddriver_node,
        critical=False
    )

    # [Action Node] caranddriver.com:F2:A1 - Search box input for finding articles
    caranddriver_search_ok = has_any_ci(answer, ['car and driver']) and has_any_ci(answer, ['search', 'find', 'look'])
    evaluator.add_custom_node(
        result=bool(caranddriver_search_ok),
        id="caranddriver_action_search",
        desc="[Action Node] caranddriver.com:F2:A1 - Use search functionality to find 2025 Ford Bronco related articles",
        parent=caranddriver_node,
        critical=False
    )

    # Check that all three upgrade types have recommendations
    has_roof_rack = bool(recommendations.roof_rack_brand_model and recommendations.roof_rack_brand_model.strip())
    has_winch = bool(recommendations.winch_brand_model and recommendations.winch_brand_model.strip())
    has_tires = bool(recommendations.all_terrain_tires_brand_model and recommendations.all_terrain_tires_brand_model.strip())

    evaluator.add_custom_node(
        result=bool(has_roof_rack and has_winch and has_tires),
        id="caranddriver_all_recommendations_present",
        desc="All three upgrade categories (roof rack, winch, all-terrain tires) have specific brand and model recommendations",
        parent=caranddriver_node,
        critical=False
    )

    # 3.2 E-commerce platforms section (parallel searches)
    ecommerce_node = evaluator.add_parallel(
        id="ecommerce_section",
        desc="Search and compare prices across Amazon, eBay, Target, and Best Buy",
        parent=root,
        critical=False
    )

    # 3.2.1 Amazon
    amazon_node = evaluator.add_sequential(
        id="amazon_subsection",
        desc="Amazon search and filtering",
        parent=ecommerce_node,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search box input with auto-complete
    amazon_search_ok = has_any_ci(answer, ['amazon'])
    evaluator.add_custom_node(
        result=bool(amazon_search_ok),
        id="amazon_action_search",
        desc="[Action Node] Amazon:F1:A13 - Search for recommended upgrade models on Amazon",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A14 - Brand multi-select filter
    amazon_brand_filter_ok = has_any_ci(answer, ['amazon']) and has_any_ci(answer, ['brand', 'filter'])
    evaluator.add_custom_node(
        result=bool(amazon_brand_filter_ok),
        id="amazon_action_brand_filter",
        desc="[Action Node] Amazon:F3:A14 - Apply brand filter to match recommended brands",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Rating filter checkbox (4 stars and above)
    amazon_rating_filter_ok = has_any_ci(answer, ['amazon']) and has_any_ci(answer, ['4 star', 'rating'])
    evaluator.add_custom_node(
        result=bool(amazon_rating_filter_ok),
        id="amazon_action_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Apply rating filter (4 stars and above)",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Recognize promotional tags
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['price', 'deal', 'discount', 'save'])),
        id="amazon_perception_promo_tags",
        desc="[Perception Node] Amazon:F3:P1 - Identify promotional tags to find better prices",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P2 - Extract price information
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['amazon']) and contains_digits(answer)),
        id="amazon_perception_price",
        desc="[Perception Node] Amazon:F3:P2 - Extract price information from product listings",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click product card to view details
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['amazon.com', 'amazon'])),
        id="amazon_action_click_product",
        desc="[Action Node] Amazon:F5:A20 - Click product card to enter detail page",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P3 - Recognize product image content
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['amazon']) and (has_any_ci(answer, ['roof rack', 'winch', 'tire']))),
        id="amazon_perception_product_image",
        desc="[Perception Node] Amazon:F5:P3 - Verify product images match the target upgrade type",
        parent=amazon_node,
        critical=False
    )

    # 3.2.2 eBay
    ebay_node = evaluator.add_sequential(
        id="ebay_subsection",
        desc="eBay search and filtering",
        parent=ecommerce_node,
        critical=False
    )

    # [Action Node] eBay:F1:A5 - Category selection dropdown
    ebay_category_ok = has_any_ci(answer, ['ebay'])
    evaluator.add_custom_node(
        result=bool(ebay_category_ok),
        id="ebay_action_category_select",
        desc="[Action Node] eBay:F1:A5 - Select appropriate category (eBay Motors or Parts & Accessories)",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A11 - Product condition multi-select (New/In stock)
    ebay_condition_ok = has_any_ci(answer, ['ebay']) and has_any_ci(answer, ['stock', 'in stock', 'available'])
    evaluator.add_custom_node(
        result=bool(ebay_condition_ok),
        id="ebay_action_condition_filter",
        desc="[Action Node] eBay:F3:A11 - Filter by condition (New) and stock status",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F3:A18 - Price range slider
    ebay_price_slider_ok = has_any_ci(answer, ['ebay']) and has_any_ci(answer, ['price', 'lowest'])
    evaluator.add_custom_node(
        result=bool(ebay_price_slider_ok),
        id="ebay_action_price_slider",
        desc="[Action Node] eBay:F3:A18 - Use price range slider to narrow down to lowest prices",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F3:P5 - Recognize shipping tags
    ebay_shipping_ok = has_any_ci(answer, ['ebay']) and has_any_ci(answer, ['shipping', 'delivery', 'standard'])
    evaluator.add_custom_node(
        result=bool(ebay_shipping_ok),
        id="ebay_perception_shipping_tags",
        desc="[Perception Node] eBay:F3:P5 - Identify shipping tags (standard or faster)",
        parent=ebay_node,
        critical=False
    )

    # [Action Node] eBay:F5:A27 - Click product card
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['ebay.com', 'ebay'])),
        id="ebay_action_click_product",
        desc="[Action Node] eBay:F5:A27 - Click product card to view details",
        parent=ebay_node,
        critical=False
    )

    # [Perception Node] eBay:F5:P9 - Recognize product type
    ebay_product_type_ok = has_any_ci(answer, ['ebay']) and (has_any_ci(answer, ['roof rack', 'winch', 'tire']))
    evaluator.add_custom_node(
        result=bool(ebay_product_type_ok),
        id="ebay_perception_product_type",
        desc="[Perception Node] eBay:F5:P9 - Verify product type matches target upgrade",
        parent=ebay_node,
        critical=False
    )

    # 3.2.3 Target
    target_node = evaluator.add_sequential(
        id="target_subsection",
        desc="Target search and filtering",
        parent=ecommerce_node,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filter panel
    target_filter_ok = has_any_ci(answer, ['target']) and has_any_ci(answer, ['rating', 'stock', 'shipping'])
    evaluator.add_custom_node(
        result=bool(target_filter_ok),
        id="target_action_multi_filter",
        desc="[Action Node] target.com:F1:A1 - Apply multiple filters (rating 4+ stars, in stock, standard shipping or faster)",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A2 - Sort selection (price ascending)
    target_sort_ok = has_any_ci(answer, ['target']) and has_any_ci(answer, ['price', 'lowest', 'sort'])
    evaluator.add_custom_node(
        result=bool(target_sort_ok),
        id="target_action_sort",
        desc="[Action Node] target.com:F1:A2 - Sort by price (lowest first)",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Recognize product list information
    target_list_info_ok = has_any_ci(answer, ['target']) and contains_digits(answer)
    evaluator.add_custom_node(
        result=bool(target_list_info_ok),
        id="target_perception_list_info",
        desc="[Perception Node] target.com:F1:P1 - Extract price, rating, and promotional info from product list",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F2:A21 - Click product card
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['target.com', 'target'])),
        id="target_action_click_product",
        desc="[Action Node] target.com:F2:A21 - Click product card to view details",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P5 - Recognize promotional tags
    target_promo_ok = has_any_ci(answer, ['target']) and has_any_ci(answer, ['price', 'deal', 'save'])
    evaluator.add_custom_node(
        result=bool(target_promo_ok),
        id="target_perception_promo_tags",
        desc="[Perception Node] target.com:F2:P5 - Identify promotional tags to find better prices",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P7 - Recognize stock status
    target_stock_ok = has_any_ci(answer, ['target']) and has_any_ci(answer, ['stock', 'available', 'in stock'])
    evaluator.add_custom_node(
        result=bool(target_stock_ok),
        id="target_perception_stock_status",
        desc="[Perception Node] target.com:F2:P7 - Identify stock status",
        parent=target_node,
        critical=False
    )

    # 3.2.4 Best Buy
    bestbuy_node = evaluator.add_sequential(
        id="bestbuy_subsection",
        desc="Best Buy search and filtering",
        parent=ecommerce_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A1 - Search box input
    bestbuy_search_ok = has_any_ci(answer, ['best buy', 'bestbuy'])
    evaluator.add_custom_node(
        result=bool(bestbuy_search_ok),
        id="bestbuy_action_search",
        desc="[Action Node] bestbuy.com:F1:A1 - Search for upgrade models on Best Buy",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A2 - Multi-condition filter panel
    bestbuy_filter_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and has_any_ci(answer, ['rating', 'stock'])
    evaluator.add_custom_node(
        result=bool(bestbuy_filter_ok),
        id="bestbuy_action_multi_filter",
        desc="[Action Node] bestbuy.com:F1:A2 - Apply filters (rating 4+ stars, in stock)",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A3 - Sort dropdown (price ascending)
    bestbuy_sort_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and has_any_ci(answer, ['price', 'lowest', 'sort'])
    evaluator.add_custom_node(
        result=bool(bestbuy_sort_ok),
        id="bestbuy_action_sort",
        desc="[Action Node] bestbuy.com:F1:A3 - Sort by price (lowest first)",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F1:P1 - Recognize promotional tags
    bestbuy_promo_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and has_any_ci(answer, ['save', 'deal', 'discount'])
    evaluator.add_custom_node(
        result=bool(bestbuy_promo_ok),
        id="bestbuy_perception_promo_tags",
        desc="[Perception Node] bestbuy.com:F1:P1 - Identify promotional tags (Save, Deal, etc.)",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F1:P2 - Extract price information
    bestbuy_price_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and contains_digits(answer)
    evaluator.add_custom_node(
        result=bool(bestbuy_price_ok),
        id="bestbuy_perception_price",
        desc="[Perception Node] bestbuy.com:F1:P2 - Extract price information from product cards",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F2:A5 - Click product card to view details
    evaluator.add_custom_node(
        result=bool(has_any_ci(answer, ['bestbuy.com', 'best buy'])),
        id="bestbuy_action_click_product",
        desc="[Action Node] bestbuy.com:F2:A5 - Click product card to enter detail page",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F2:P5 - Extract stock and shipping information
    bestbuy_stock_shipping_ok = has_any_ci(answer, ['best buy', 'bestbuy']) and (
        has_any_ci(answer, ['stock', 'available']) or has_any_ci(answer, ['shipping', 'delivery'])
    )
    evaluator.add_custom_node(
        result=bool(bestbuy_stock_shipping_ok),
        id="bestbuy_perception_stock_shipping",
        desc="[Perception Node] bestbuy.com:F2:P5 - Extract stock status and shipping information",
        parent=bestbuy_node,
        critical=False
    )

    # 3.3 Price comparison and decision logic
    comparison_node = evaluator.add_sequential(
        id="comparison_section",
        desc="Compare prices across platforms and apply decision rules",
        parent=root,
        critical=False
    )

    # Check if answer contains price comparison
    has_price_comparison = has_any_ci(answer, ['lowest', 'cheapest', 'compare', 'price'])
    evaluator.add_custom_node(
        result=bool(has_price_comparison),
        id="comparison_price_analysis",
        desc="Perform price comparison across all four platforms",
        parent=comparison_node,
        critical=False
    )

    # Check if rating/review rules are applied (4.5 stars, 50 reviews)
    applies_rating_rules = has_any_ci(answer, ['4.5', '50 reviews', 'second lowest'])
    evaluator.add_custom_node(
        result=bool(applies_rating_rules),
        id="comparison_rating_rules",
        desc="Apply decision rules: if lowest price has rating < 4.5 stars or < 50 reviews, choose second-lowest",
        parent=comparison_node,
        critical=False
    )

    # 3.4 Final output verification
    output_node = evaluator.add_sequential(
        id="output_section",
        desc="Verify final output completeness and correctness",
        parent=root,
        critical=False
    )

    # Check roof rack output
    roof_rack_info = purchase_info.roof_rack if purchase_info else None
    roof_rack_complete = (
        roof_rack_info and
        bool(roof_rack_info.brand_model) and
        has_platform_name(roof_rack_info.platform) and
        looks_like_price(roof_rack_info.price) and
        looks_like_rating(roof_rack_info.rating) and
        contains_digits(roof_rack_info.num_reviews) and
        bool(roof_rack_info.stock_status) and
        bool(roof_rack_info.shipping_method) and
        looks_like_url(roof_rack_info.product_link)
    )
    evaluator.add_custom_node(
        result=bool(roof_rack_complete),
        id="output_roof_rack_complete",
        desc="Roof rack output includes: brand/model, platform, price, rating, reviews, stock, shipping, link",
        parent=output_node,
        critical=False
    )

    # Check winch output
    winch_info = purchase_info.winch if purchase_info else None
    winch_complete = (
        winch_info and
        bool(winch_info.brand_model) and
        has_platform_name(winch_info.platform) and
        looks_like_price(winch_info.price) and
        looks_like_rating(winch_info.rating) and
        contains_digits(winch_info.num_reviews) and
        bool(winch_info.stock_status) and
        bool(winch_info.shipping_method) and
        looks_like_url(winch_info.product_link)
    )
    evaluator.add_custom_node(
        result=bool(winch_complete),
        id="output_winch_complete",
        desc="Winch output includes: brand/model, platform, price, rating, reviews, stock, shipping, link",
        parent=output_node,
        critical=False
    )

    # Check all-terrain tires output
    tires_info = purchase_info.all_terrain_tires if purchase_info else None
    tires_complete = (
        tires_info and
        bool(tires_info.brand_model) and
        has_platform_name(tires_info.platform) and
        looks_like_price(tires_info.price) and
        looks_like_rating(tires_info.rating) and
        contains_digits(tires_info.num_reviews) and
        bool(tires_info.stock_status) and
        bool(tires_info.shipping_method) and
        looks_like_url(tires_info.product_link)
    )
    evaluator.add_custom_node(
        result=bool(tires_complete),
        id="output_tires_complete",
        desc="All-terrain tires output includes: brand/model, platform, price, rating, reviews, stock, shipping, link",
        parent=output_node,
        critical=False
    )

    # Verify all three categories are present in output
    evaluator.add_custom_node(
        result=bool(roof_rack_complete and winch_complete and tires_complete),
        id="output_all_three_categories",
        desc="All three upgrade categories have complete purchase information in the output",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
