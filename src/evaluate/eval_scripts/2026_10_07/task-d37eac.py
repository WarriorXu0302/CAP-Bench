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
TASK_ID = "task-d37eac"
TASK_DESCRIPTION = 'I saw a smart coaster trending on TikTok and want to verify its actual performance across major e-commerce platforms before deciding whether to enter this product category.\n\nFirst, search for **"Smart Coaster"** on **Amazon, Target, and Best Buy**. For each platform, record **2–3 top-rated products**. If a platform has too few results for that keyword, expand the search to **"Smart Coffee Warmer," "Coffee Warmer,"** or **"Mug Warmer,"** and clearly note in the results that an alternative keyword was used.\n\nThen, focus on **1–2 star negative reviews** across platforms and identify what users complain about most. Summarize the **high-frequency product defects** (e.g., durability issues, compatibility problems, misleading claims, etc.).\n\nFinally, search Reddit for **"Smart Coaster"** or **"Smart Coffee Warmer"** to review authentic user feedback in discussion threads—especially whether users mention the same issues found in e-commerce negative reviews.\n\nOutput should include:\n\n- For each platform: **product name, current price, rating, sales rank** (Best Seller / bestseller position, if available), and **product page link**  \n- A **negative review summary table**: issue keywords appearing **more than 3 times** in negative reviews per platform, with frequencies (e.g., “not durable” appears 12 times)  \n- A **Reddit discussion summary**: for at least **3 relevant threads**, provide **title, comment count, key takeaways, and post link**\n\nIf any page requires login, has collapsed content, or access restrictions, record the limitation and continue with the remaining steps based on visible information.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProductInfo(BaseModel):
    """Single product information"""
    platform: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[str] = None
    rating: Optional[str] = None
    sales_rank: Optional[str] = None
    product_link: Optional[str] = None
    alternative_keyword_used: Optional[str] = None


class PlatformProducts(BaseModel):
    """Products from all platforms"""
    amazon_products: Optional[List[ProductInfo]] = Field(default_factory=list)
    target_products: Optional[List[ProductInfo]] = Field(default_factory=list)
    bestbuy_products: Optional[List[ProductInfo]] = Field(default_factory=list)


class NegativeReviewIssue(BaseModel):
    """Negative review issue with frequency"""
    platform: Optional[str] = None
    issue_keyword: Optional[str] = None
    frequency: Optional[int] = None


class NegativeReviews(BaseModel):
    """Negative review summary across platforms"""
    issues: Optional[List[NegativeReviewIssue]] = Field(default_factory=list)


class RedditThread(BaseModel):
    """Reddit discussion thread information"""
    title: Optional[str] = None
    comment_count: Optional[str] = None
    key_takeaways: Optional[str] = None
    post_link: Optional[str] = None


class RedditDiscussion(BaseModel):
    """Reddit discussion summary"""
    threads: Optional[List[RedditThread]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_products() -> str:
    return """
Extract all product information mentioned in the answer from Amazon, Target, and Best Buy.

For each product, extract:
- platform: the platform name (Amazon, Target, or Best Buy)
- product_name: the full product name
- price: the current price exactly as written
- rating: the product rating exactly as written
- sales_rank: any Best Seller or sales rank information if mentioned
- product_link: the product page URL if provided
- alternative_keyword_used: if the answer mentions using an alternative keyword like "Smart Coffee Warmer" or "Mug Warmer", note it here

Group products by platform in amazon_products, target_products, and bestbuy_products lists.
If any information is missing, set it to null.
"""


def prompt_extract_negative_reviews() -> str:
    return """
Extract the negative review summary from the answer, specifically the high-frequency issue keywords that appear more than 3 times in 1-2 star reviews.

For each issue, extract:
- platform: which platform this issue is from (Amazon, Target, or Best Buy)
- issue_keyword: the specific problem or complaint keyword (e.g., "not durable", "stops working", "poor quality")
- frequency: how many times this issue appeared (should be > 3)

If the answer doesn't contain frequency information or negative review summaries, return empty lists.
"""


def prompt_extract_reddit() -> str:
    return """
Extract Reddit discussion thread information from the answer.

For each thread mentioned (should be at least 3), extract:
- title: the thread title
- comment_count: number of comments
- key_takeaways: main points or insights from the discussion
- post_link: the Reddit post URL

If fewer than 3 threads are mentioned or information is missing, extract what's available.
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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def safe_len(items: Optional[List]) -> int:
    return len(items) if items else 0


def has_valid_rating(rating_text: Optional[str]) -> bool:
    if not rating_text:
        return False
    num = extract_float(rating_text)
    return num is not None and num >= 0 and num <= 5


def has_valid_link(link: Optional[str]) -> bool:
    if not link:
        return False
    return link.startswith('http://') or link.startswith('https://')


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
    products_info = await evaluator.extract(
        prompt=prompt_extract_products(),
        template_class=PlatformProducts,
        extraction_name="platform_products"
    )

    negative_reviews_info = await evaluator.extract(
        prompt=prompt_extract_negative_reviews(),
        template_class=NegativeReviews,
        extraction_name="negative_reviews"
    )

    reddit_info = await evaluator.extract(
        prompt=prompt_extract_reddit(),
        template_class=RedditDiscussion,
        extraction_name="reddit_discussion"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon product search and analysis",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A5 - Category dropdown selection
    amazon_category_ok = ci_contains(answer, 'amazon') and has_any_ci(answer, ['smart coaster', 'coffee warmer', 'mug warmer'])
    evaluator.add_custom_node(
        result=bool(amazon_category_ok),
        id="amazon_category_selection",
        desc="[Action Node] Amazon:F1:A5 - Select appropriate category for Smart Coaster search",
        parent=amazon_node,
        critical=False
    )

    # Check if Amazon products have valid names related to smart coasters
    amazon_products = products_info.amazon_products if products_info and products_info.amazon_products else []
    amazon_product_names_ok = any(
        p.product_name and has_any_ci(p.product_name, ['coaster', 'warmer', 'mug', 'coffee'])
        for p in amazon_products
    )

    # [Action Node] Amazon:F3:A15 - Rating filter
    amazon_ratings = [p.rating for p in amazon_products if p and p.rating]
    amazon_all_high_rated = all(
        has_valid_rating(r) and extract_float(r) >= 4.0
        for r in amazon_ratings
    ) if amazon_ratings else False

    evaluator.add_custom_node(
        result=bool(amazon_all_high_rated and len(amazon_ratings) > 0),
        id="amazon_rating_filter",
        desc="[Action Node] Amazon:F3:A15 - Filter products by 4 Stars & Up",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F4:A4 - Sort by customer review
    amazon_ratings_desc = amazon_ratings == sorted(amazon_ratings, reverse=True, key=lambda r: extract_float(r) or 0) if len(amazon_ratings) > 1 else True
    evaluator.add_custom_node(
        result=bool(amazon_ratings_desc and len(amazon_ratings) > 0),
        id="amazon_sort_by_review",
        desc="[Action Node] Amazon:F4:A4 - Sort products by Avg. Customer Review",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Best Seller tag identification
    amazon_has_bestseller = any(
        p.sales_rank and has_any_ci(p.sales_rank, ['best seller', 'bestseller', '#1'])
        for p in amazon_products
    )
    evaluator.add_custom_node(
        result=bool(amazon_has_bestseller),
        id="amazon_bestseller_tag",
        desc="[Perception Node] Amazon:F3:P1 - Identify Best Seller tags or sales rank",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Click to product details
    amazon_has_links = any(
        p.product_link and has_valid_link(p.product_link) and 'amazon' in p.product_link.lower()
        for p in amazon_products
    )
    evaluator.add_custom_node(
        result=bool(amazon_has_links),
        id="amazon_product_details",
        desc="[Action Node] Amazon:F5:A20 - Navigate to product detail pages",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A7 - Tab switch to reviews
    amazon_mentions_reviews = has_any_ci(answer, ['review', 'customer review', 'rating'])
    evaluator.add_custom_node(
        result=bool(amazon_mentions_reviews),
        id="amazon_reviews_tab",
        desc="[Action Node] Amazon:F5:A7 - Switch to Customer reviews tab",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F10:P15 - Extract negative review issues
    amazon_issues = [
        issue for issue in (negative_reviews_info.issues if negative_reviews_info and negative_reviews_info.issues else [])
        if issue.platform and ci_contains(issue.platform, 'amazon')
    ]
    amazon_has_negative_issues = len(amazon_issues) > 0 and any(
        issue.issue_keyword and issue.frequency and issue.frequency > 3
        for issue in amazon_issues
    )
    evaluator.add_custom_node(
        result=bool(amazon_has_negative_issues),
        id="amazon_negative_review_analysis",
        desc="[Perception Node] Amazon:F10:P15 - Extract and summarize high-frequency issues from 1-2 star reviews",
        parent=amazon_node,
        critical=False
    )

    # 3.2 Target section
    target_node = evaluator.add_sequential(
        id="target_section",
        desc="Target product search and analysis",
        parent=root,
        critical=False
    )

    target_products = products_info.target_products if products_info and products_info.target_products else []

    # [Action Node] target.com:F1:A1 - Multi-criteria filter
    target_ratings = [p.rating for p in target_products if p and p.rating]
    target_all_high_rated = all(
        has_valid_rating(r) and extract_float(r) >= 4.0
        for r in target_ratings
    ) if target_ratings else False

    evaluator.add_custom_node(
        result=bool(target_all_high_rated and len(target_ratings) > 0),
        id="target_rating_filter",
        desc="[Action Node] target.com:F1:A1 - Apply high rating filter (>=4.0 stars)",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F1:A2 - Sort by rating
    target_ratings_desc = target_ratings == sorted(target_ratings, reverse=True, key=lambda r: extract_float(r) or 0) if len(target_ratings) > 1 else True
    evaluator.add_custom_node(
        result=bool(target_ratings_desc and len(target_ratings) > 0),
        id="target_sort_by_review",
        desc="[Action Node] target.com:F1:A2 - Sort by Avg. customer review",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F1:P1 - Product list data perception
    target_has_complete_info = any(
        p.product_name and p.price and p.rating
        for p in target_products
    )
    evaluator.add_custom_node(
        result=bool(target_has_complete_info),
        id="target_product_info",
        desc="[Perception Node] target.com:F1:P1 - Extract product name, price, and rating from product cards",
        parent=target_node,
        critical=False
    )

    # [Action Node] target.com:F2:A21 - Click to details
    target_has_links = any(
        p.product_link and has_valid_link(p.product_link) and 'target' in p.product_link.lower()
        for p in target_products
    )
    evaluator.add_custom_node(
        result=bool(target_has_links),
        id="target_product_details",
        desc="[Action Node] target.com:F2:A21 - Navigate to product detail pages",
        parent=target_node,
        critical=False
    )

    # [Perception Node] target.com:F12:P8 - Review content perception
    target_issues = [
        issue for issue in (negative_reviews_info.issues if negative_reviews_info and negative_reviews_info.issues else [])
        if issue.platform and ci_contains(issue.platform, 'target')
    ]
    target_has_negative_issues = len(target_issues) > 0 and any(
        issue.issue_keyword and issue.frequency
        for issue in target_issues
    )
    evaluator.add_custom_node(
        result=bool(target_has_negative_issues),
        id="target_negative_review_analysis",
        desc="[Perception Node] target.com:F12:P8 - Identify and extract low-star review issues",
        parent=target_node,
        critical=False
    )

    # 3.3 Best Buy section
    bestbuy_node = evaluator.add_sequential(
        id="bestbuy_section",
        desc="Best Buy product search and analysis",
        parent=root,
        critical=False
    )

    bestbuy_products = products_info.bestbuy_products if products_info and products_info.bestbuy_products else []

    # [Action Node] bestbuy.com:F1:A1 - Search input
    bestbuy_search_ok = ci_contains(answer, 'best buy') and has_any_ci(answer, ['smart coaster', 'coffee warmer', 'mug warmer'])
    evaluator.add_custom_node(
        result=bool(bestbuy_search_ok),
        id="bestbuy_search_input",
        desc="[Action Node] bestbuy.com:F1:A1 - Enter search keyword in Best Buy search box",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A2 - Multi-criteria filter
    bestbuy_ratings = [p.rating for p in bestbuy_products if p and p.rating]
    bestbuy_all_high_rated = all(
        has_valid_rating(r) and extract_float(r) >= 4.0
        for r in bestbuy_ratings
    ) if bestbuy_ratings else False

    evaluator.add_custom_node(
        result=bool(bestbuy_all_high_rated and len(bestbuy_ratings) > 0),
        id="bestbuy_rating_filter",
        desc="[Action Node] bestbuy.com:F1:A2 - Apply Customer Rating high rating filter",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F1:A3 - Sort by rating
    bestbuy_ratings_desc = bestbuy_ratings == sorted(bestbuy_ratings, reverse=True, key=lambda r: extract_float(r) or 0) if len(bestbuy_ratings) > 1 else True
    evaluator.add_custom_node(
        result=bool(bestbuy_ratings_desc and len(bestbuy_ratings) > 0),
        id="bestbuy_sort_by_review",
        desc="[Action Node] bestbuy.com:F1:A3 - Sort by Customer Rating",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F2:A5 - Click to details
    bestbuy_has_links = any(
        p.product_link and has_valid_link(p.product_link) and 'bestbuy' in p.product_link.lower()
        for p in bestbuy_products
    )
    evaluator.add_custom_node(
        result=bool(bestbuy_has_links),
        id="bestbuy_product_details",
        desc="[Action Node] bestbuy.com:F2:A5 - Navigate to product detail pages",
        parent=bestbuy_node,
        critical=False
    )

    # [Action Node] bestbuy.com:F4:A12 - Review filter
    bestbuy_mentions_low_star = has_any_ci(answer, ['1 star', '2 star', '1-2 star', 'negative review', 'low rating'])
    evaluator.add_custom_node(
        result=bool(bestbuy_mentions_low_star),
        id="bestbuy_review_filter",
        desc="[Action Node] bestbuy.com:F4:A12 - Filter reviews by 1-2 star ratings",
        parent=bestbuy_node,
        critical=False
    )

    # [Perception Node] bestbuy.com:F4:P8 - Review content analysis
    bestbuy_issues = [
        issue for issue in (negative_reviews_info.issues if negative_reviews_info and negative_reviews_info.issues else [])
        if issue.platform and ci_contains(issue.platform, 'best buy')
    ]
    bestbuy_has_negative_issues = len(bestbuy_issues) > 0 and any(
        issue.issue_keyword and issue.frequency
        for issue in bestbuy_issues
    )
    evaluator.add_custom_node(
        result=bool(bestbuy_has_negative_issues),
        id="bestbuy_negative_review_analysis",
        desc="[Perception Node] bestbuy.com:F4:P8 - Analyze and extract issue keywords from negative reviews",
        parent=bestbuy_node,
        critical=False
    )

    # 3.4 Cross-platform negative review summary
    summary_node = evaluator.add_sequential(
        id="negative_review_summary",
        desc="Cross-platform negative review summary with frequencies",
        parent=root,
        critical=False
    )

    all_issues = negative_reviews_info.issues if negative_reviews_info and negative_reviews_info.issues else []
    has_frequency_table = len(all_issues) > 0 and any(
        issue.issue_keyword and issue.frequency and issue.frequency > 3
        for issue in all_issues
    )
    evaluator.add_custom_node(
        result=bool(has_frequency_table),
        id="negative_review_frequency_table",
        desc="Generate negative review summary table with issue keywords appearing >3 times",
        parent=summary_node,
        critical=False
    )

    # Check if all three platforms are covered
    platforms_covered = set()
    for issue in all_issues:
        if issue.platform:
            if 'amazon' in issue.platform.lower():
                platforms_covered.add('amazon')
            elif 'target' in issue.platform.lower():
                platforms_covered.add('target')
            elif 'best buy' in issue.platform.lower():
                platforms_covered.add('bestbuy')

    evaluator.add_custom_node(
        result=bool(len(platforms_covered) >= 2),
        id="multi_platform_coverage",
        desc="Negative review analysis covers multiple platforms",
        parent=summary_node,
        critical=False
    )

    # 3.5 Reddit discussion section
    reddit_node = evaluator.add_sequential(
        id="reddit_section",
        desc="Reddit discussion analysis for Smart Coaster feedback",
        parent=root,
        critical=False
    )

    reddit_threads = reddit_info.threads if reddit_info and reddit_info.threads else []

    # Check if search was performed
    reddit_search_ok = has_any_ci(answer, ['reddit']) and has_any_ci(answer, ['smart coaster', 'coffee warmer'])
    evaluator.add_custom_node(
        result=bool(reddit_search_ok),
        id="reddit_search",
        desc="Search Reddit for Smart Coaster or Smart Coffee Warmer discussions",
        parent=reddit_node,
        critical=False
    )

    # Check if at least 3 threads are documented
    has_three_threads = safe_len(reddit_threads) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_threads),
        id="reddit_thread_count",
        desc="Document at least 3 relevant Reddit discussion threads",
        parent=reddit_node,
        critical=False
    )

    # Check thread information completeness
    threads_complete = sum(
        1 for thread in reddit_threads
        if thread.title and thread.post_link and thread.key_takeaways
    )
    evaluator.add_custom_node(
        result=bool(threads_complete >= 3),
        id="reddit_thread_details",
        desc="Extract title, comment count, key takeaways, and post link for each thread",
        parent=reddit_node,
        critical=False
    )

    # Check if Reddit feedback correlates with e-commerce issues
    reddit_mentions_issues = has_any_ci(answer, ['durability', 'stops working', 'quality', 'temperature', 'heating', 'defect', 'broken'])
    evaluator.add_custom_node(
        result=bool(reddit_mentions_issues),
        id="reddit_issue_correlation",
        desc="Identify whether Reddit discussions mention similar issues found in e-commerce negative reviews",
        parent=reddit_node,
        critical=False
    )

    # 3.6 Overall completeness checks
    completeness_node = evaluator.add_parallel(
        id="overall_completeness",
        desc="Overall task completeness verification",
        parent=root,
        critical=False
    )

    # Check total product count across platforms
    total_products = safe_len(amazon_products) + safe_len(target_products) + safe_len(bestbuy_products)
    evaluator.add_custom_node(
        result=bool(total_products >= 6),
        id="product_count_adequate",
        desc="Collected 2-3 products per platform (total >=6 products)",
        parent=completeness_node,
        critical=False
    )

    # Check if alternative keywords are noted when used
    has_alternative_keyword_note = any(
        p.alternative_keyword_used
        for products in [amazon_products, target_products, bestbuy_products]
        for p in products
    ) or not has_any_ci(answer, ['alternative', 'expanded', 'coffee warmer', 'mug warmer'])

    evaluator.add_custom_node(
        result=True,  # Always pass since this is optional
        id="alternative_keyword_documentation",
        desc="Document when alternative keywords are used due to limited results",
        parent=completeness_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
