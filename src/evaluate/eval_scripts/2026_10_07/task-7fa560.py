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
TASK_ID = "task-7fa560"
TASK_DESCRIPTION = 'I’m considering buying clothing from sustainable fashion brands and want to verify whether their environmental claims are credible.\n\nFirst, please search for sustainability-related women’s clothing items on Target (prioritize the keyword **“sustainable”**; if there is no explicit “sustainable” label on the site, use related keywords/tags such as **“organic cotton”** or **“recycled materials”**). Try to find products from **5 different brands**, each with a rating of **4 stars or above**. Record the following for each item: **brand name, product name, price, rating, and the environmental claim mentioned on the product page** (e.g., “made with organic cotton,” “reduced carbon emissions,” etc.). If fewer than 5 qualifying items are found, record the actual number found and specify which criteria were not fully met.\n\nThen, go to **Our World in Data** to look up data on **carbon emissions** and **water consumption** in the textile industry. Identify the **global average benchmark values** for textile carbon emissions intensity and water consumption, review the trends shown in the charts, and record key figures.\n\nNext, search Etsy for **“sustainable handmade clothing women”** and try to find **3 independent designer shops** with ratings of **4.5 stars or above**. Check how their product descriptions and shop About pages describe their sustainability practices. Record: **shop name, representative product, price, and sustainability-practice description**. If fewer than 3 such shops are found, record the actual number and supplement with relevant shops rated **4.0 stars or above**, clearly noting the rating differences.\n\nFinally, output:\n- For each Target brand: **brand name, product name, price, rating, sustainability claim description, product link**\n- From Our World in Data: **textile industry carbon-emissions and water-consumption benchmark values, plus chart links**\n- For each Etsy shop: **shop name, representative product, price, rating, sustainability-practice description, product link**'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TargetProduct(BaseModel):
    """Single Target product information"""
    brand_name: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    sustainability_claim: Optional[str] = None
    product_link: Optional[str] = None


class TargetProducts(BaseModel):
    """Collection of Target products"""
    products: List[TargetProduct] = Field(default_factory=list)


class OurWorldInDataInfo(BaseModel):
    """Our World in Data textile industry data"""
    carbon_emissions_benchmark: Optional[str] = None
    water_consumption_benchmark: Optional[str] = None
    carbon_chart_link: Optional[str] = None
    water_chart_link: Optional[str] = None
    trend_description: Optional[str] = None


class EtsyShop(BaseModel):
    """Single Etsy shop information"""
    shop_name: Optional[str] = None
    representative_product: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    sustainability_practice: Optional[str] = None
    product_link: Optional[str] = None


class EtsyShops(BaseModel):
    """Collection of Etsy shops"""
    shops: List[EtsyShop] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_target_products() -> str:
    return """
Extract all Target products mentioned in the answer that are related to sustainable women's clothing.

For each product, extract:
- brand_name: the brand name
- product_name: the product name
- price: the price as stated (include currency symbols if present)
- rating: the rating value
- sustainability_claim: the environmental claim mentioned (e.g., "made with organic cotton", "recycled materials")
- product_link: the product URL if provided

Return all products found in a list. If any field is missing for a product, set it to null.
"""


def prompt_extract_ourworldindata() -> str:
    return """
Extract the Our World in Data information about the textile industry from the answer.

Extract:
- carbon_emissions_benchmark: the global average carbon emissions intensity value for textiles
- water_consumption_benchmark: the global average water consumption value for textiles
- carbon_chart_link: the URL to the carbon emissions chart if provided
- water_chart_link: the URL to the water consumption chart if provided
- trend_description: any description of trends mentioned

If any field is missing, set it to null.
"""


def prompt_extract_etsy_shops() -> str:
    return """
Extract all Etsy shops mentioned in the answer that sell sustainable handmade women's clothing.

For each shop, extract:
- shop_name: the name of the shop
- representative_product: a representative product mentioned
- price: the price as stated
- rating: the shop or product rating
- sustainability_practice: the sustainability practice description from product or About page
- product_link: the product or shop URL if provided

Return all shops found in a list. If any field is missing, set it to null.
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


def count_unique_brands(products: List[TargetProduct]) -> int:
    brands = set()
    for p in products:
        if p.brand_name and p.brand_name.strip():
            brands.add(p.brand_name.strip().lower())
    return len(brands)


def all_ratings_above(products: List[TargetProduct], threshold: float) -> bool:
    for p in products:
        if not p.rating:
            return False
        rating_val = extract_float(p.rating)
        if rating_val is None or rating_val < threshold:
            return False
    return True


def all_etsy_ratings_above(shops: List[EtsyShop], threshold: float) -> bool:
    for s in shops:
        if not s.rating:
            return False
        rating_val = extract_float(s.rating)
        if rating_val is None or rating_val < threshold:
            return False
    return True


def has_sustainability_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['sustainable', 'organic', 'recycled', 'eco', 'green', 'environmental', 'carbon', 'emission']
    return has_any_ci(text, keywords)


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
    target_products = await evaluator.extract(
        prompt=prompt_extract_target_products(),
        template_class=TargetProducts,
        extraction_name="target_products"
    )

    owid_info = await evaluator.extract(
        prompt=prompt_extract_ourworldindata(),
        template_class=OurWorldInDataInfo,
        extraction_name="ourworldindata_info"
    )

    etsy_shops = await evaluator.extract(
        prompt=prompt_extract_etsy_shops(),
        template_class=EtsyShops,
        extraction_name="etsy_shops"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target sustainable women's clothing products",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filtering (rating >= 4 stars)
    products_list = target_products.products if target_products else []
    has_products = len(products_list) > 0
    ratings_ok = all_ratings_above(products_list, 4.0) if has_products else False

    evaluator.add_custom_node(
        result=bool(has_products and ratings_ok),
        id="target_filtering_rating",
        desc="[Action Node] target.com:F1:A1 - Filter products with rating 4 stars or above",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Product list awareness (5 different brands)
    unique_brands = count_unique_brands(products_list)
    brands_ok = unique_brands >= 5

    evaluator.add_custom_node(
        result=bool(brands_ok),
        id="target_brand_diversity",
        desc="[Perception Node] target.com:F1:P1 - Identify and select products from 5 different brands",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product card to view details
    has_detailed_claims = any(p.sustainability_claim and len(p.sustainability_claim.strip()) > 10 for p in products_list)

    evaluator.add_custom_node(
        result=bool(has_detailed_claims),
        id="target_product_detail_click",
        desc="[Action Node] target.com:F1:A21 - Click into product detail pages to view sustainability claims",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P18 - Product description perception (sustainability claims)
    claims_with_keywords = sum(1 for p in products_list if has_sustainability_keywords(p.sustainability_claim))
    claims_ok = claims_with_keywords >= 3

    evaluator.add_custom_node(
        result=bool(claims_ok),
        id="target_sustainability_claims",
        desc="[Perception Node] target.com:F2:P18 - Extract sustainability claims from product descriptions",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P5 - Promotional tag perception (sustainable labels)
    mentions_sustainable_keywords = has_any_ci(answer, ['sustainable', 'organic cotton', 'recycled materials', 'eco-friendly'])

    evaluator.add_custom_node(
        result=bool(mentions_sustainable_keywords),
        id="target_sustainable_labels",
        desc="[Perception Node] target.com:F2:P5 - Identify sustainable or eco-friendly labels/tags on products",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P6 - Price information perception
    prices_present = sum(1 for p in products_list if p.price and contains_digits(p.price))
    prices_ok = prices_present >= 3

    evaluator.add_custom_node(
        result=bool(prices_ok),
        id="target_price_extraction",
        desc="[Perception Node] target.com:F2:P6 - Extract price information for products",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P4 - Rating data perception
    ratings_present = sum(1 for p in products_list if p.rating and contains_digits(p.rating))
    ratings_data_ok = ratings_present >= 3

    evaluator.add_custom_node(
        result=bool(ratings_data_ok),
        id="target_rating_perception",
        desc="[Perception Node] target.com:F2:P4 - Extract accurate rating values for products",
        parent=target_node,
        critical=False
    )

    # 3.2 Our World in Data section
    owid_node = evaluator.add_sequential(
        id="ourworldindata_section",
        desc="Our World in Data textile industry environmental data",
        parent=root,
        critical=False
    )

    # [Action Node] ourworldindata.org:F1:A8 - Hover over charts to view values
    has_specific_values = (owid_info and
                          owid_info.carbon_emissions_benchmark and
                          contains_digits(owid_info.carbon_emissions_benchmark) and
                          owid_info.water_consumption_benchmark and
                          contains_digits(owid_info.water_consumption_benchmark))

    evaluator.add_custom_node(
        result=bool(has_specific_values),
        id="owid_chart_hover",
        desc="[Action Node] ourworldindata.org:F1:A8 - Hover over charts to extract precise numerical values",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P9 - Value reading (benchmark values)
    carbon_value = extract_float(owid_info.carbon_emissions_benchmark) if owid_info else None
    water_value = extract_float(owid_info.water_consumption_benchmark) if owid_info else None
    values_ok = carbon_value is not None and water_value is not None

    evaluator.add_custom_node(
        result=bool(values_ok),
        id="owid_value_reading",
        desc="[Perception Node] ourworldindata.org:F1:P9 - Read accurate benchmark values from charts with units",
        parent=owid_node,
        critical=False
    )

    # [Perception Node] ourworldindata.org:F1:P1 - Trend judgment
    has_trend_info = owid_info and owid_info.trend_description and len(owid_info.trend_description.strip()) > 5

    evaluator.add_custom_node(
        result=bool(has_trend_info),
        id="owid_trend_analysis",
        desc="[Perception Node] ourworldindata.org:F1:P1 - Identify and describe trends from charts",
        parent=owid_node,
        critical=False
    )

    # 3.3 Etsy section
    etsy_node = evaluator.add_sequential(
        id="etsy_section",
        desc="Etsy sustainable handmade clothing shops",
        parent=root,
        critical=False
    )

    # [Action Node] etsy.com:F1:A1 - Multi-condition filtering (rating >= 4.5 stars)
    shops_list = etsy_shops.shops if etsy_shops else []
    has_shops = len(shops_list) > 0
    etsy_ratings_ok = all_etsy_ratings_above(shops_list, 4.5) if has_shops else False

    evaluator.add_custom_node(
        result=bool(has_shops and etsy_ratings_ok),
        id="etsy_filtering_rating",
        desc="[Action Node] etsy.com:F1:A1 - Filter shops with rating 4.5 stars or above",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] etsy.com:F1:P1 - Search result perception (3 independent designers)
    shop_count = len(shops_list)
    shop_count_ok = shop_count >= 3

    evaluator.add_custom_node(
        result=bool(shop_count_ok),
        id="etsy_shop_identification",
        desc="[Perception Node] etsy.com:F1:P1 - Identify and select 3 independent designer shops",
        parent=etsy_node,
        critical=False
    )

    # [Action Node] etsy.com:F3:A21 - Click product to view details
    etsy_detailed_practices = sum(1 for s in shops_list if s.sustainability_practice and len(s.sustainability_practice.strip()) > 15)
    etsy_details_ok = etsy_detailed_practices >= 2

    evaluator.add_custom_node(
        result=bool(etsy_details_ok),
        id="etsy_product_click",
        desc="[Action Node] etsy.com:F3:A21 - Click into product pages to view sustainability descriptions",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] etsy.com:F3:P2 - Product description understanding
    etsy_practices_with_keywords = sum(1 for s in shops_list if has_sustainability_keywords(s.sustainability_practice))
    etsy_practices_ok = etsy_practices_with_keywords >= 2

    evaluator.add_custom_node(
        result=bool(etsy_practices_ok),
        id="etsy_product_description",
        desc="[Perception Node] etsy.com:F3:P2 - Understand and extract sustainability practices from product descriptions",
        parent=etsy_node,
        critical=False
    )

    # [Action Node] etsy.com:F4:A10 - Visit shop page
    mentions_about_page = has_any_ci(answer, ['about', 'shop page', 'shop info', 'shop description'])

    evaluator.add_custom_node(
        result=bool(mentions_about_page),
        id="etsy_shop_page_visit",
        desc="[Action Node] etsy.com:F4:A10 - Navigate to shop About page",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] etsy.com:F4:P7 - Shop information understanding
    comprehensive_practices = sum(1 for s in shops_list if s.sustainability_practice and len(s.sustainability_practice.strip()) > 30)
    comprehensive_ok = comprehensive_practices >= 1

    evaluator.add_custom_node(
        result=bool(comprehensive_ok),
        id="etsy_shop_info_understanding",
        desc="[Perception Node] etsy.com:F4:P7 - Understand and synthesize sustainability practices from shop About page",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] etsy.com:F3:P4 - Rating perception
    etsy_ratings_present = sum(1 for s in shops_list if s.rating and contains_digits(s.rating))
    etsy_ratings_data_ok = etsy_ratings_present >= 2

    evaluator.add_custom_node(
        result=bool(etsy_ratings_data_ok),
        id="etsy_rating_perception",
        desc="[Perception Node] etsy.com:F3:P4 - Extract accurate shop or product ratings",
        parent=etsy_node,
        critical=False
    )

    # [Perception Node] etsy.com:F3:P3 - Price perception
    etsy_prices_present = sum(1 for s in shops_list if s.price and contains_digits(s.price))
    etsy_prices_ok = etsy_prices_present >= 2

    evaluator.add_custom_node(
        result=bool(etsy_prices_ok),
        id="etsy_price_perception",
        desc="[Perception Node] etsy.com:F3:P3 - Extract price information from Etsy products",
        parent=etsy_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
