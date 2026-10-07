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
TASK_ID = "task-79f5e0"
TASK_DESCRIPTION = 'I want to seriously learn watercolor painting on Udemy. Please find me one beginner course with these core requirements: a rating of at least 4.5 and more than 1,000 reviews. Also, prioritize courses with the “Bestseller” badge. If no course meets both the core requirements and has a Bestseller badge, then choose from courses that meet only the core requirements.\n\nAfter selecting the course, carefully review its course description or “Requirements” section. The instructor may specify exact paper brands (e.g., Arches) or specific brush models. If no exact brand/model is provided, extract actionable general specifications instead (e.g., 100% Cotton Paper, 300gsm, Cold Press, Round Brush sizes).\n\nThen extract the key painting materials required by the instructor, search for the same items or corresponding specs on Amazon, and filter for products rated above 4 stars.\n\nFinally, provide me with a checklist that includes:\n- Course link  \n- Instructor-required material names  \n- Corresponding Amazon product links, ratings, and prices  \n- A note for each material indicating whether it is an “exact brand/model match” or a “matched by general specifications”\n\nIf any material has no Amazon result above 4 stars, keep that search item and mark it as: “No products found that meet the rating requirement.”'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UdemyCourseInfo(BaseModel):
    """Udemy course details extracted from the answer"""
    course_link: Optional[str] = None
    course_title: Optional[str] = None
    rating: Optional[str] = None
    review_count: Optional[str] = None
    has_bestseller_badge: Optional[bool] = None


class MaterialItem(BaseModel):
    """A single material item with its Amazon search results"""
    material_name: Optional[str] = None
    amazon_product_link: Optional[str] = None
    amazon_rating: Optional[str] = None
    amazon_price: Optional[str] = None
    match_type: Optional[str] = None


class MaterialsList(BaseModel):
    """List of all materials extracted from the answer"""
    materials: List[MaterialItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_udemy_course() -> str:
    return """
Extract the Udemy course information from the answer:

- course_link: the full URL to the Udemy course
- course_title: the title of the course
- rating: the course rating as stated (e.g., "4.7", "4.8")
- review_count: the number of reviews as stated (e.g., "5,000", "1,200")
- has_bestseller_badge: true if the course has a Bestseller badge, false otherwise

If any field is missing, set it to null.
"""


def prompt_extract_materials_list() -> str:
    return """
Extract all the painting materials from the answer's checklist. For each material, extract:

- material_name: the name of the material as required by the instructor
- amazon_product_link: the Amazon product link provided
- amazon_rating: the Amazon product rating (e.g., "4.5 stars", "4.7")
- amazon_price: the price of the product on Amazon
- match_type: whether it's an "exact brand/model match" or "matched by general specifications" or similar wording

Return a list of all materials. If no materials are found, return an empty list.
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
    # Remove commas and extract digits
    clean = re.sub(r'[,\s]', '', text)
    m = re.search(r'(\d+)', clean)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def looks_like_rating(text: Optional[str]) -> bool:
    if not text:
        return False
    num = extract_float(text)
    if num is None:
        return False
    return 0 <= num <= 5


def looks_like_udemy_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'udemy.com' in text.lower()


def looks_like_amazon_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'amazon.com' in text.lower() or 'amzn.' in text.lower()


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
    course_info = await evaluator.extract(
        prompt=prompt_extract_udemy_course(),
        template_class=UdemyCourseInfo,
        extraction_name="udemy_course_info"
    )

    materials_list = await evaluator.extract(
        prompt=prompt_extract_materials_list(),
        template_class=MaterialsList,
        extraction_name="materials_list"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Udemy course selection part
    udemy_node = evaluator.add_sequential(
        id="udemy_course_selection",
        desc="Udemy course selection with rating and review filtering",
        parent=root,
        critical=False
    )

    # [Action Node] udemy.com:F1:A1 - Filter by rating (at least 4.5)
    rating_num = extract_float(course_info.rating)
    rating_meets_requirement = rating_num is not None and rating_num >= 4.5
    evaluator.add_custom_node(
        result=bool(rating_meets_requirement),
        id="udemy_filter_rating",
        desc="[Action Node] udemy.com:F1:A1 - Filter courses by rating of at least 4.5",
        parent=udemy_node,
        critical=False
    )

    # Check review count (at least 1,000)
    review_num = extract_int(course_info.review_count)
    review_meets_requirement = review_num is not None and review_num >= 1000
    evaluator.add_custom_node(
        result=bool(review_meets_requirement),
        id="udemy_check_reviews",
        desc="Check that the selected course has more than 1,000 reviews",
        parent=udemy_node,
        critical=False
    )

    # [Perception Node] udemy.com:F1:P1 - Identify Bestseller badge
    bestseller_identified = course_info.has_bestseller_badge is True or has_any_ci(answer, ['bestseller'])
    evaluator.add_custom_node(
        result=bool(bestseller_identified),
        id="udemy_bestseller_badge",
        desc="[Perception Node] udemy.com:F1:P1 - Identify courses with Bestseller badge and prioritize them",
        parent=udemy_node,
        critical=False
    )

    # Course link provided
    course_link_ok = looks_like_udemy_url(course_info.course_link)
    evaluator.add_custom_node(
        result=bool(course_link_ok),
        id="udemy_course_link",
        desc="Provide the Udemy course link in the checklist",
        parent=udemy_node,
        critical=False
    )

    # 3.2 Course description examination part
    course_details_node = evaluator.add_sequential(
        id="course_description_examination",
        desc="Examine course description and Requirements section for materials",
        parent=root,
        critical=False
    )

    # [Action Node] udemy.com:F2:A10 - Expand/click to view full course description
    description_examined = (has_any_ci(answer, ['course description', 'requirements', 'materials']) or
                           (materials_list and materials_list.materials))
    evaluator.add_custom_node(
        result=bool(description_examined),
        id="udemy_expand_description",
        desc="[Action Node] udemy.com:F2:A10 - Expand or navigate to view the full course description or Requirements section",
        parent=course_details_node,
        critical=False
    )

    # Extract materials with specifications
    materials_extracted = materials_list and len(materials_list.materials) > 0
    evaluator.add_custom_node(
        result=bool(materials_extracted),
        id="extract_materials_specs",
        desc="Extract painting materials with exact brands/models or general specifications from instructor requirements",
        parent=course_details_node,
        critical=False
    )

    # 3.3 Amazon search and filtering part
    amazon_node = evaluator.add_sequential(
        id="amazon_product_search",
        desc="Search Amazon for materials and filter by rating",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F3:A15 - Filter products by rating (above 4 stars)
    if materials_list and materials_list.materials:
        # Check if at least one material has an Amazon rating above 4 stars
        any_rating_above_4 = False
        for material in materials_list.materials:
            if material.amazon_rating:
                rating = extract_float(material.amazon_rating)
                if rating is not None and rating > 4.0:
                    any_rating_above_4 = True
                    break

        evaluator.add_custom_node(
            result=bool(any_rating_above_4),
            id="amazon_filter_rating",
            desc="[Action Node] Amazon:F3:A15 - Filter Amazon products to show only items rated above 4 stars",
            parent=amazon_node,
            critical=False
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="amazon_filter_rating",
            desc="[Action Node] Amazon:F3:A15 - Filter Amazon products to show only items rated above 4 stars",
            parent=amazon_node,
            critical=False
        )

    # Check Amazon links provided
    amazon_links_provided = False
    if materials_list and materials_list.materials:
        amazon_links_provided = any(
            looks_like_amazon_url(m.amazon_product_link) for m in materials_list.materials
            if m.amazon_product_link
        )

    evaluator.add_custom_node(
        result=bool(amazon_links_provided),
        id="amazon_links_in_checklist",
        desc="Provide Amazon product links for materials in the checklist",
        parent=amazon_node,
        critical=False
    )

    # 3.4 Final checklist completeness
    checklist_node = evaluator.add_parallel(
        id="final_checklist",
        desc="Complete checklist with all required information",
        parent=root,
        critical=False
    )

    # Material names listed
    material_names_ok = materials_list and len(materials_list.materials) > 0 and any(
        m.material_name for m in materials_list.materials
    )
    evaluator.add_custom_node(
        result=bool(material_names_ok),
        id="checklist_material_names",
        desc="Include instructor-required material names in the checklist",
        parent=checklist_node,
        critical=False
    )

    # Ratings and prices included
    ratings_prices_ok = False
    if materials_list and materials_list.materials:
        ratings_prices_ok = any(
            (m.amazon_rating or m.amazon_price) for m in materials_list.materials
        )

    evaluator.add_custom_node(
        result=bool(ratings_prices_ok),
        id="checklist_ratings_prices",
        desc="Include Amazon product ratings and prices in the checklist",
        parent=checklist_node,
        critical=False
    )

    # Match type annotations
    match_type_ok = False
    if materials_list and materials_list.materials:
        match_type_ok = any(
            m.match_type and (
                ci_contains(m.match_type, 'exact') or
                ci_contains(m.match_type, 'brand') or
                ci_contains(m.match_type, 'model') or
                ci_contains(m.match_type, 'general') or
                ci_contains(m.match_type, 'spec')
            ) for m in materials_list.materials
        )

    evaluator.add_custom_node(
        result=bool(match_type_ok),
        id="checklist_match_type",
        desc="Include match type notes (exact brand/model match vs. matched by general specifications)",
        parent=checklist_node,
        critical=False
    )

    # Handle products not meeting rating requirement
    handles_no_results = has_any_ci(answer, ['no products found', 'no result', 'not found', 'rating requirement'])
    evaluator.add_custom_node(
        result=bool(handles_no_results),
        id="checklist_handles_no_results",
        desc="Mark materials with no Amazon results above 4 stars appropriately",
        parent=checklist_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
