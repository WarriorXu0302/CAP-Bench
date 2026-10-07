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
TASK_ID = "task-b623c0"
TASK_DESCRIPTION = "I just moved into a new apartment and need to purchase a living room set: a sofa, a coffee table, and a bookshelf. My total budget is under $800. I want to compare prices across IKEA, Target, and Amazon to find the most cost-effective combination.\n\nFirst, go to IKEA and search for these three furniture categories. For each category, find 3 items with a rating of 4 stars or higher, sorted by price from low to high. Record the product name, price, and rating. Also, check the stock status for my ZIP code (assume 10001).\n\nThen, go to Target and search for the same three categories. Again, for each category, find 3 items with a rating of 4 stars or higher, sorted by price in ascending order. Record the product name, price, rating, and delivery method (specifically note if free 'Order Pickup' is available).\n\nFinally, go to Amazon and search for these three categories. Filter for 'Prime Eligible' and a rating of 4 stars or higher, sorted by price in ascending order. Find 3 items for each category. Record the product name, price, rating, and whether it has the 'Prime free delivery' indicator.\n\nPlease organize the data from all three platforms into a table. Then, calculate: How many combinations can be formed by selecting one item from each of the three categories (sofa + coffee table + bookshelf) across all three platforms? Which combination has the lowest total price?\n\nIf Target items are chosen with free 'Order Pickup', Amazon items with 'Prime free delivery', and IKEA items are calculated with a $50 shipping fee, what is the most economical solution?\n\nOutput: For each platform and each category, the top 3 products (product name, price, rating, stock/delivery status, product detail page link). As well as the recommended optimal combination plan (which three specific items, total price, and how much it saves compared to the second-best option)."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class IKEAProduct(BaseModel):
    """IKEA product details"""
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    stock_status: Optional[str] = None
    link: Optional[str] = None


class IKEAData(BaseModel):
    """IKEA data for three categories"""
    sofas: List[IKEAProduct] = Field(default_factory=list)
    coffee_tables: List[IKEAProduct] = Field(default_factory=list)
    bookshelves: List[IKEAProduct] = Field(default_factory=list)


class TargetProduct(BaseModel):
    """Target product details"""
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    delivery_method: Optional[str] = None
    order_pickup_available: Optional[bool] = None
    link: Optional[str] = None


class TargetData(BaseModel):
    """Target data for three categories"""
    sofas: List[TargetProduct] = Field(default_factory=list)
    coffee_tables: List[TargetProduct] = Field(default_factory=list)
    bookshelves: List[TargetProduct] = Field(default_factory=list)


class AmazonProduct(BaseModel):
    """Amazon product details"""
    name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    prime_eligible: Optional[bool] = None
    prime_free_delivery: Optional[bool] = None
    link: Optional[str] = None


class AmazonData(BaseModel):
    """Amazon data for three categories"""
    sofas: List[AmazonProduct] = Field(default_factory=list)
    coffee_tables: List[AmazonProduct] = Field(default_factory=list)
    bookshelves: List[AmazonProduct] = Field(default_factory=list)


class OptimalCombination(BaseModel):
    """Optimal combination analysis"""
    total_combinations: Optional[int] = None
    lowest_price_combination: Optional[str] = None
    lowest_price_total: Optional[str] = None
    economical_with_shipping: Optional[str] = None
    economical_total: Optional[str] = None
    savings: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ikea_data() -> str:
    return """
Extract all IKEA product data from the answer for sofas, coffee tables, and bookshelves.
For each category, extract up to 3 products with their name, price, rating, stock status, and product detail page link.
If any field is missing, set it to null.
"""


def prompt_extract_target_data() -> str:
    return """
Extract all Target product data from the answer for sofas, coffee tables, and bookshelves.
For each category, extract up to 3 products with their name, price, rating, delivery method,
whether free Order Pickup is available, and product detail page link.
If any field is missing, set it to null.
"""


def prompt_extract_amazon_data() -> str:
    return """
Extract all Amazon product data from the answer for sofas, coffee tables, and bookshelves.
For each category, extract up to 3 products with their name, price, rating,
whether it's Prime eligible, whether it has Prime free delivery, and product detail page link.
If any field is missing, set it to null.
"""


def prompt_extract_optimal_combination() -> str:
    return """
Extract the optimal combination analysis from the answer:
- total_combinations: the total number of possible combinations mentioned
- lowest_price_combination: description of the lowest price combination
- lowest_price_total: the total price of the lowest price combination
- economical_with_shipping: description of the most economical solution considering shipping fees
- economical_total: the total price of the most economical solution
- savings: the amount saved compared to the second-best option

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


def extract_rating(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    rating = extract_float(text)
    return rating


def has_rating_4_or_higher(rating_text: Optional[str]) -> bool:
    rating = extract_rating(rating_text)
    if rating is None:
        return False
    return rating >= 4.0


def has_price_info(price_text: Optional[str]) -> bool:
    if not price_text:
        return False
    return bool(re.search(r'[\$\€\£]?\d+(?:\.\d+)?', price_text))


def prices_ascending(products: List) -> bool:
    if not products or len(products) < 2:
        return True
    prices = []
    for p in products:
        price_text = getattr(p, 'price', None)
        if not price_text:
            return False
        price_val = extract_float(price_text)
        if price_val is None:
            return False
        prices.append(price_val)
    for i in range(len(prices) - 1):
        if prices[i] > prices[i + 1]:
            return False
    return True


def count_products(products: List) -> int:
    return len([p for p in products if getattr(p, 'name', None)])


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
    ikea_data = await evaluator.extract(
        prompt=prompt_extract_ikea_data(),
        template_class=IKEAData,
        extraction_name="ikea_products"
    )

    target_data = await evaluator.extract(
        prompt=prompt_extract_target_data(),
        template_class=TargetData,
        extraction_name="target_products"
    )

    amazon_data = await evaluator.extract(
        prompt=prompt_extract_amazon_data(),
        template_class=AmazonData,
        extraction_name="amazon_products"
    )

    optimal_combination = await evaluator.extract(
        prompt=prompt_extract_optimal_combination(),
        template_class=OptimalCombination,
        extraction_name="optimal_combination"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 IKEA section
    ikea_node = evaluator.add_sequential(
        id="ikea_section",
        desc="IKEA furniture search and data collection",
        parent=root,
        critical=False
    )

    # [Action Node] ikea.com:F1:A3 - Multi-select filtering (rating 4+ stars)
    ikea_sofas_rating_ok = all(has_rating_4_or_higher(p.rating) for p in ikea_data.sofas if p.rating)
    ikea_tables_rating_ok = all(has_rating_4_or_higher(p.rating) for p in ikea_data.coffee_tables if p.rating)
    ikea_shelves_rating_ok = all(has_rating_4_or_higher(p.rating) for p in ikea_data.bookshelves if p.rating)
    ikea_rating_filter_ok = ikea_sofas_rating_ok and ikea_tables_rating_ok and ikea_shelves_rating_ok

    evaluator.add_custom_node(
        result=bool(ikea_rating_filter_ok and (ikea_data.sofas or ikea_data.coffee_tables or ikea_data.bookshelves)),
        id="ikea_rating_filter",
        desc="[Action Node] ikea.com:F1:A3 - Apply rating filter (4 stars or higher) for all categories",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F1:A4 - Sort by price (low to high)
    ikea_sofas_sorted = prices_ascending(ikea_data.sofas)
    ikea_tables_sorted = prices_ascending(ikea_data.coffee_tables)
    ikea_shelves_sorted = prices_ascending(ikea_data.bookshelves)
    ikea_sort_ok = ikea_sofas_sorted and ikea_tables_sorted and ikea_shelves_sorted

    evaluator.add_custom_node(
        result=bool(ikea_sort_ok),
        id="ikea_price_sort",
        desc="[Action Node] ikea.com:F1:A4 - Sort products by price in ascending order",
        parent=ikea_node,
        critical=False
    )

    # [Action Node] ikea.com:F1:A5 - Click into product details
    ikea_has_details = (count_products(ikea_data.sofas) > 0 or
                        count_products(ikea_data.coffee_tables) > 0 or
                        count_products(ikea_data.bookshelves) > 0)
    ikea_has_prices = any(has_price_info(p.price) for p in ikea_data.sofas + ikea_data.coffee_tables + ikea_data.bookshelves)
    ikea_has_ratings = any(p.rating for p in ikea_data.sofas + ikea_data.coffee_tables + ikea_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(ikea_has_details and ikea_has_prices and ikea_has_ratings),
        id="ikea_detail_page_access",
        desc="[Action Node] ikea.com:F1:A5 - Access product detail pages to collect complete information",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F1:P7 - Promotional label awareness
    ikea_mentions_promo = has_any_ci(answer, ['new', 'ikea family', 'sale', 'discount', 'promo'])

    evaluator.add_custom_node(
        result=bool(ikea_mentions_promo or ikea_has_details),
        id="ikea_promo_label_awareness",
        desc="[Perception Node] ikea.com:F1:P7 - Recognize promotional labels (New, IKEA Family price, etc.)",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P6 - Stock status awareness
    ikea_has_stock = any(p.stock_status for p in ikea_data.sofas + ikea_data.coffee_tables + ikea_data.bookshelves)
    ikea_mentions_zip = has_any_ci(answer, ['10001', 'zip', 'stock'])

    evaluator.add_custom_node(
        result=bool(ikea_has_stock and ikea_mentions_zip),
        id="ikea_stock_status_awareness",
        desc="[Perception Node] ikea.com:F2:P6 - Check stock status for ZIP code 10001",
        parent=ikea_node,
        critical=False
    )

    # [Perception Node] ikea.com:F2:P8 - Price information awareness
    ikea_price_complete = all(has_price_info(p.price) for p in ikea_data.sofas + ikea_data.coffee_tables + ikea_data.bookshelves if p.name)

    evaluator.add_custom_node(
        result=bool(ikea_price_complete),
        id="ikea_price_awareness",
        desc="[Perception Node] ikea.com:F2:P8 - Extract accurate price information for all products",
        parent=ikea_node,
        critical=False
    )

    # Check for 3 items per category
    ikea_three_per_category = (count_products(ikea_data.sofas) >= 3 and
                                count_products(ikea_data.coffee_tables) >= 3 and
                                count_products(ikea_data.bookshelves) >= 3)

    evaluator.add_custom_node(
        result=bool(ikea_three_per_category),
        id="ikea_three_items_per_category",
        desc="Collect 3 items for each of the three furniture categories",
        parent=ikea_node,
        critical=False
    )

    # 3.2 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target furniture search and data collection",
        parent=root,
        critical=False
    )

    # [Action Node] target.com:F1:A1 - Multi-condition filtering (rating 4+ stars)
    target_sofas_rating_ok = all(has_rating_4_or_higher(p.rating) for p in target_data.sofas if p.rating)
    target_tables_rating_ok = all(has_rating_4_or_higher(p.rating) for p in target_data.coffee_tables if p.rating)
    target_shelves_rating_ok = all(has_rating_4_or_higher(p.rating) for p in target_data.bookshelves if p.rating)
    target_rating_filter_ok = target_sofas_rating_ok and target_tables_rating_ok and target_shelves_rating_ok

    evaluator.add_custom_node(
        result=bool(target_rating_filter_ok and (target_data.sofas or target_data.coffee_tables or target_data.bookshelves)),
        id="target_rating_filter",
        desc="[Action Node] target.com:F1:A1 - Apply rating filter (4 stars or higher) in left sidebar",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A2 - Sort by price (low to high)
    target_sofas_sorted = prices_ascending(target_data.sofas)
    target_tables_sorted = prices_ascending(target_data.coffee_tables)
    target_shelves_sorted = prices_ascending(target_data.bookshelves)
    target_sort_ok = target_sofas_sorted and target_tables_sorted and target_shelves_sorted

    evaluator.add_custom_node(
        result=bool(target_sort_ok),
        id="target_price_sort",
        desc="[Action Node] target.com:F1:A2 - Select 'Price low to high' in Sort by dropdown",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A21 - Click product card to access details
    target_has_details = (count_products(target_data.sofas) > 0 or
                          count_products(target_data.coffee_tables) > 0 or
                          count_products(target_data.bookshelves) > 0)
    target_has_prices = any(has_price_info(p.price) for p in target_data.sofas + target_data.coffee_tables + target_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(target_has_details and target_has_prices),
        id="target_product_card_click",
        desc="[Action Node] target.com:F1:A21 - Click product cards to access detail pages",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Product list data awareness
    target_has_names = count_products(target_data.sofas + target_data.coffee_tables + target_data.bookshelves) > 0
    target_has_ratings = any(p.rating for p in target_data.sofas + target_data.coffee_tables + target_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(target_has_names and target_has_prices and target_has_ratings),
        id="target_list_data_awareness",
        desc="[Perception Node] target.com:F1:P1 - Extract product name, price, rating from search results",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F2:P5 - Promotional label awareness
    target_mentions_promo = has_any_ci(answer, ['sale', 'target circle', 'discount', 'deal'])

    evaluator.add_custom_node(
        result=bool(target_mentions_promo or target_has_details),
        id="target_promo_label_awareness",
        desc="[Perception Node] target.com:F2:P5 - Recognize promotional labels (Sale, Target Circle, etc.)",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F7:P17 - Fulfillment option awareness (Order Pickup)
    target_has_delivery = any(p.delivery_method for p in target_data.sofas + target_data.coffee_tables + target_data.bookshelves)
    target_mentions_pickup = has_any_ci(answer, ['order pickup', 'pickup', 'free pickup'])

    evaluator.add_custom_node(
        result=bool(target_has_delivery or target_mentions_pickup),
        id="target_fulfillment_awareness",
        desc="[Perception Node] target.com:F7:P17 - Identify free Order Pickup availability in fulfillment options",
        parent=target_node,
        critical=False
    )

    # Check for 3 items per category
    target_three_per_category = (count_products(target_data.sofas) >= 3 and
                                  count_products(target_data.coffee_tables) >= 3 and
                                  count_products(target_data.bookshelves) >= 3)

    evaluator.add_custom_node(
        result=bool(target_three_per_category),
        id="target_three_items_per_category",
        desc="Collect 3 items for each of the three furniture categories",
        parent=target_node,
        critical=False
    )

    # 3.3 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon furniture search and data collection",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A13 - Search with autocomplete
    amazon_has_data = (count_products(amazon_data.sofas) > 0 or
                       count_products(amazon_data.coffee_tables) > 0 or
                       count_products(amazon_data.bookshelves) > 0)
    amazon_mentions_search = has_any_ci(answer, ['amazon', 'search', 'sofa', 'coffee table', 'bookshelf'])

    evaluator.add_custom_node(
        result=bool(amazon_has_data and amazon_mentions_search),
        id="amazon_search",
        desc="[Action Node] Amazon:F1:A13 - Search for three furniture categories using search box",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A14 - Check Prime checkbox
    amazon_prime_filter = all(getattr(p, 'prime_eligible', False) for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves if p.name)

    evaluator.add_custom_node(
        result=bool(amazon_prime_filter or has_any_ci(answer, ['prime eligible', 'prime'])),
        id="amazon_prime_filter",
        desc="[Action Node] Amazon:F3:A14 - Check Prime checkbox in left sidebar filters",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Rating filter (4 stars & up)
    amazon_sofas_rating_ok = all(has_rating_4_or_higher(p.rating) for p in amazon_data.sofas if p.rating)
    amazon_tables_rating_ok = all(has_rating_4_or_higher(p.rating) for p in amazon_data.coffee_tables if p.rating)
    amazon_shelves_rating_ok = all(has_rating_4_or_higher(p.rating) for p in amazon_data.bookshelves if p.rating)
    amazon_rating_filter_ok = amazon_sofas_rating_ok and amazon_tables_rating_ok and amazon_shelves_rating_ok

    evaluator.add_custom_node(
        result=bool(amazon_rating_filter_ok and amazon_has_data),
        id="amazon_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Select '4 Stars & Up' in Customer Review filter",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F4:A4 - Sort by price (low to high)
    amazon_sofas_sorted = prices_ascending(amazon_data.sofas)
    amazon_tables_sorted = prices_ascending(amazon_data.coffee_tables)
    amazon_shelves_sorted = prices_ascending(amazon_data.bookshelves)
    amazon_sort_ok = amazon_sofas_sorted and amazon_tables_sorted and amazon_shelves_sorted

    evaluator.add_custom_node(
        result=bool(amazon_sort_ok),
        id="amazon_price_sort",
        desc="[Action Node] Amazon:F4:A4 - Select 'Price: Low to High' in Sort by dropdown",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click into product details
    amazon_has_prices = any(has_price_info(p.price) for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves)
    amazon_has_ratings = any(p.rating for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(amazon_has_data and amazon_has_prices and amazon_has_ratings),
        id="amazon_detail_page_access",
        desc="[Action Node] Amazon:F5:A20 - Click product cards to access detail pages",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Prime badge awareness
    amazon_prime_awareness = any(getattr(p, 'prime_eligible', False) for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(amazon_prime_awareness or has_any_ci(answer, ['prime'])),
        id="amazon_prime_badge_awareness",
        desc="[Perception Node] Amazon:F3:P1 - Recognize Prime badge on product cards",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F5:P19 - Prime free delivery awareness
    amazon_free_delivery = any(getattr(p, 'prime_free_delivery', False) for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves)
    amazon_mentions_free_delivery = has_any_ci(answer, ['prime free delivery', 'free delivery', 'free shipping'])

    evaluator.add_custom_node(
        result=bool(amazon_free_delivery or amazon_mentions_free_delivery),
        id="amazon_free_delivery_awareness",
        desc="[Perception Node] Amazon:F5:P19 - Identify Prime free delivery indicator by hovering or reading details",
        parent=amazon_node,
        critical=False
    )

    # Check for 3 items per category
    amazon_three_per_category = (count_products(amazon_data.sofas) >= 3 and
                                  count_products(amazon_data.coffee_tables) >= 3 and
                                  count_products(amazon_data.bookshelves) >= 3)

    evaluator.add_custom_node(
        result=bool(amazon_three_per_category),
        id="amazon_three_items_per_category",
        desc="Collect 3 items for each of the three furniture categories",
        parent=amazon_node,
        critical=False
    )

    # 3.4 Analysis section
    analysis_node = evaluator.add_sequential(
        id="analysis_section",
        desc="Cross-platform price comparison and optimal combination calculation",
        parent=root,
        critical=False
    )

    # Check for table/organized data presentation
    has_table = has_any_ci(answer, ['table', '|', 'platform', 'category'])

    evaluator.add_custom_node(
        result=bool(has_table),
        id="data_organization",
        desc="Organize collected data from all platforms into a table format",
        parent=analysis_node,
        critical=False
    )

    # Check combinations calculation (27 = 3×3×3 per platform × 3 platforms)
    mentions_combinations = has_any_ci(answer, ['combination', 'combinations'])
    has_combination_count = optimal_combination.total_combinations is not None

    evaluator.add_custom_node(
        result=bool(mentions_combinations and has_combination_count),
        id="combination_count",
        desc="Calculate total number of possible combinations (27 = 3×3×3 across categories)",
        parent=analysis_node,
        critical=False
    )

    # Check lowest price combination
    has_lowest_combo = (optimal_combination.lowest_price_combination is not None and
                        optimal_combination.lowest_price_total is not None)

    evaluator.add_custom_node(
        result=bool(has_lowest_combo),
        id="lowest_price_combination",
        desc="Identify combination with the lowest total price",
        parent=analysis_node,
        critical=False
    )

    # Check shipping-adjusted economical solution
    mentions_shipping = has_any_ci(answer, ['shipping', 'delivery', '$50', 'order pickup', 'prime'])
    has_economical_solution = (optimal_combination.economical_with_shipping is not None and
                                optimal_combination.economical_total is not None)

    evaluator.add_custom_node(
        result=bool(mentions_shipping and has_economical_solution),
        id="shipping_adjusted_solution",
        desc="Calculate most economical solution considering shipping fees (IKEA +$50, Target Order Pickup free, Amazon Prime free)",
        parent=analysis_node,
        critical=False
    )

    # Check savings calculation
    has_savings = optimal_combination.savings is not None
    mentions_savings = has_any_ci(answer, ['save', 'saving', 'cheaper', 'second'])

    evaluator.add_custom_node(
        result=bool(has_savings or mentions_savings),
        id="savings_comparison",
        desc="Calculate savings compared to the second-best option",
        parent=analysis_node,
        critical=False
    )

    # 3.5 Output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Verify all required output elements are present",
        parent=root,
        critical=False
    )

    # Product links provided
    ikea_links = any(p.link for p in ikea_data.sofas + ikea_data.coffee_tables + ikea_data.bookshelves)
    target_links = any(p.link for p in target_data.sofas + target_data.coffee_tables + target_data.bookshelves)
    amazon_links = any(p.link for p in amazon_data.sofas + amazon_data.coffee_tables + amazon_data.bookshelves)

    evaluator.add_custom_node(
        result=bool(ikea_links or target_links or amazon_links),
        id="product_links",
        desc="Include product detail page links for collected items",
        parent=output_node,
        critical=False
    )

    # Recommendation provided
    has_recommendation = has_any_ci(answer, ['recommend', 'optimal', 'best', 'suggestion'])

    evaluator.add_custom_node(
        result=bool(has_recommendation and has_lowest_combo),
        id="recommendation",
        desc="Provide clear recommendation with specific product names and total price",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
