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
TASK_ID = "task-9e1250"
TASK_DESCRIPTION = "I'm currently researching science-backed weight loss methods and want to develop an actionable diet plan based on academic evidence.\n\nFirst, go to Semantic Scholar and search for papers related to 'intermittent fasting'. Filter for papers published since 2023, with over 150 citations, and available with PDF. Find 3 papers, sorted by citation count in descending order. Record the paper title, publication year, citation count, and core fasting method (e.g., 16:8, 5:2, etc.).\n\nNext, go to WebMD or Mayo Clinic and search for 'intermittent fasting'. Review the clinical recommendations and safety considerations from these medical institutions regarding these fasting methods. Confirm if the methods mentioned in the papers are recognized by mainstream medical organizations. Record the recommended fasting windows and contraindications.\n\nThen, go to AllRecipes and search for 'high protein low calorie'. Filter recipes with a preparation time under 30 minutes and calories between 300-500 kcal. Find 3 quick recipes suitable for the fasting window.\n\nFinally, go to Food Network and search for 'healthy dinner recipes'. Filter for recipes with a 5-star rating and under 400 kcal. Find 2 nutrient-dense dinner options. Expand the nutritional information to check if the protein content exceeds 25 grams.\n\n**Output:**\nFor each paper: title, publication year, citation count, Semantic Scholar link, and extracted fasting method.\nFor WebMD/Mayo Clinic: a summary of clinical recommendations for the method, safety considerations, recommended fasting windows, and the medical website link.\nFor AllRecipes: 3 recipe names, preparation time, calories, and recipe link.\nFor Food Network: 2 recipe names, calories, protein content, rating, and recipe link."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SemanticScholarPaper(BaseModel):
    """Single paper info from Semantic Scholar"""
    title: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: Optional[int] = None
    link: Optional[str] = None
    fasting_method: Optional[str] = None


class SemanticScholarPapers(BaseModel):
    """All papers extracted from Semantic Scholar"""
    papers: List[SemanticScholarPaper] = Field(default_factory=list)


class MedicalInfo(BaseModel):
    """Medical recommendations from WebMD/Mayo Clinic"""
    clinical_recommendations: Optional[str] = None
    safety_considerations: Optional[str] = None
    recommended_fasting_windows: Optional[str] = None
    source_link: Optional[str] = None


class AllRecipesRecipe(BaseModel):
    """Single recipe from AllRecipes"""
    name: Optional[str] = None
    prep_time: Optional[str] = None
    calories: Optional[str] = None
    link: Optional[str] = None


class AllRecipesData(BaseModel):
    """All recipes from AllRecipes"""
    recipes: List[AllRecipesRecipe] = Field(default_factory=list)


class FoodNetworkRecipe(BaseModel):
    """Single recipe from Food Network"""
    name: Optional[str] = None
    calories: Optional[str] = None
    protein: Optional[str] = None
    rating: Optional[str] = None
    link: Optional[str] = None


class FoodNetworkData(BaseModel):
    """All recipes from Food Network"""
    recipes: List[FoodNetworkRecipe] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_semantic_scholar() -> str:
    return """
Extract all Semantic Scholar papers mentioned in the answer for 'intermittent fasting'.

For each paper, extract:
- title: the paper title exactly as stated
- publication_year: the publication year as an integer
- citation_count: the citation count as an integer
- link: the Semantic Scholar link URL
- fasting_method: the extracted fasting method (e.g., "16:8", "5:2", "alternate day fasting")

Return a list of papers. If no papers are mentioned, return an empty list.
"""


def prompt_extract_medical_info() -> str:
    return """
Extract the medical information from WebMD or Mayo Clinic mentioned in the answer.

Extract:
- clinical_recommendations: summary of clinical recommendations for intermittent fasting
- safety_considerations: safety considerations or contraindications mentioned
- recommended_fasting_windows: recommended fasting windows or timing
- source_link: the medical website link (WebMD or Mayo Clinic)

If any field is missing, set it to null.
"""


def prompt_extract_allrecipes() -> str:
    return """
Extract all AllRecipes recipes mentioned in the answer for 'high protein low calorie'.

For each recipe, extract:
- name: the recipe name
- prep_time: preparation time as stated
- calories: calorie count as stated
- link: the recipe link URL

Return a list of recipes. If no recipes are mentioned, return an empty list.
"""


def prompt_extract_foodnetwork() -> str:
    return """
Extract all Food Network recipes mentioned in the answer for 'healthy dinner recipes'.

For each recipe, extract:
- name: the recipe name
- calories: calorie count as stated
- protein: protein content as stated
- rating: the rating (e.g., "5 stars", "5-star")
- link: the recipe link URL

Return a list of recipes. If no recipes are mentioned, return an empty list.
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


def extract_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    if isinstance(text, int):
        return text
    m = re.search(r'(\d+)', str(text))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.findall(r'(\d+(\.\d+)?)', str(text))
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_fasting_method(text: Optional[str]) -> bool:
    if not text:
        return False
    patterns = [
        r'\d+:\d+',  # 16:8, 18:6, etc.
        r'\d+/\d+',  # 5/2
        r'alternate.{0,10}day',
        r'time.{0,10}restricted',
        r'eat.{0,10}stop.{0,10}eat'
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def is_year_valid(year: Optional[int]) -> bool:
    if not year:
        return False
    return year >= 2023 and year <= 2026


def is_citation_valid(count: Optional[int]) -> bool:
    if count is None:
        return False
    return count >= 150


def are_citations_descending(papers: List[SemanticScholarPaper]) -> bool:
    citations = [p.citation_count for p in papers if p.citation_count is not None]
    if len(citations) < 2:
        return True
    for i in range(len(citations) - 1):
        if citations[i] < citations[i + 1]:
            return False
    return True


def looks_like_prep_time_under_30(text: Optional[str]) -> bool:
    if not text:
        return False
    minutes = extract_int(text)
    if minutes is None:
        return False
    return minutes <= 30


def looks_like_calories_in_range(text: Optional[str], min_cal: int, max_cal: int) -> bool:
    if not text:
        return False
    cal = extract_int(text)
    if cal is None:
        return False
    return min_cal <= cal <= max_cal


def looks_like_protein_over_25(text: Optional[str]) -> bool:
    if not text:
        return False
    protein = extract_int(text)
    if protein is None:
        return False
    return protein > 25


def looks_like_5_star(text: Optional[str]) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return '5' in text_lower and ('star' in text_lower or '★' in text)


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
    semantic_papers = await evaluator.extract(
        prompt=prompt_extract_semantic_scholar(),
        template_class=SemanticScholarPapers,
        extraction_name="semantic_scholar_papers"
    )

    medical_info = await evaluator.extract(
        prompt=prompt_extract_medical_info(),
        template_class=MedicalInfo,
        extraction_name="medical_info"
    )

    allrecipes_data = await evaluator.extract(
        prompt=prompt_extract_allrecipes(),
        template_class=AllRecipesData,
        extraction_name="allrecipes_data"
    )

    foodnetwork_data = await evaluator.extract(
        prompt=prompt_extract_foodnetwork(),
        template_class=FoodNetworkData,
        extraction_name="foodnetwork_data"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 Semantic Scholar section
    semantic_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar papers on intermittent fasting",
        parent=root,
        critical=False
    )

    # Check basic presence
    semantic_mentioned = has_any_ci(answer, ['semantic scholar'])
    evaluator.add_custom_node(
        result=bool(semantic_mentioned),
        id="semantic_scholar_mentioned",
        desc="Mentions Semantic Scholar as the source",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A3 - Date range filter (since 2023)
    papers_with_valid_year = [p for p in semantic_papers.papers if is_year_valid(p.publication_year)]
    date_filter_ok = len(papers_with_valid_year) >= 3
    evaluator.add_custom_node(
        result=bool(date_filter_ok),
        id="semantic_date_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Filter papers published since 2023",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citation count descending
    papers_with_valid_citations = [p for p in semantic_papers.papers if is_citation_valid(p.citation_count)]
    citations_descending = are_citations_descending(semantic_papers.papers)
    citation_sort_ok = len(papers_with_valid_citations) >= 3 and citations_descending
    evaluator.add_custom_node(
        result=bool(citation_sort_ok),
        id="semantic_citation_sort",
        desc="[Action Node] semanticscholar.org:F1:A4 - Sort papers by citation count (descending) with >150 citations",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:P2 - PDF availability filter
    pdf_mention = has_any_ci(answer, ['pdf'])
    evaluator.add_custom_node(
        result=bool(pdf_mention),
        id="semantic_pdf_filter",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Filter for papers with PDF available",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A7 - Pagination to find 3 papers
    found_three_papers = len(semantic_papers.papers) >= 3
    evaluator.add_custom_node(
        result=bool(found_three_papers),
        id="semantic_pagination",
        desc="[Action Node] semanticscholar.org:F1:A7 - Navigate search results to find 3 qualifying papers",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click into paper details
    papers_with_links = [p for p in semantic_papers.papers if p.link]
    detail_access_ok = len(papers_with_links) >= 3
    evaluator.add_custom_node(
        result=bool(detail_access_ok),
        id="semantic_detail_access",
        desc="[Action Node] semanticscholar.org:F1:A5 - Access paper detail pages to extract information",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P1 - Extract fasting method from abstract
    papers_with_fasting_method = [p for p in semantic_papers.papers if looks_like_fasting_method(p.fasting_method)]
    fasting_method_extraction_ok = len(papers_with_fasting_method) >= 1
    evaluator.add_custom_node(
        result=bool(fasting_method_extraction_ok),
        id="semantic_fasting_method_extraction",
        desc="[Perception Node] semanticscholar.org:F2:P1 - Extract core fasting method from paper abstracts",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F2:A10 - Expand TLDR or abstract
    abstract_mention = has_any_ci(answer, ['abstract', 'tldr', 'summary'])
    evaluator.add_custom_node(
        result=bool(abstract_mention),
        id="semantic_expand_abstract",
        desc="[Action Node] semanticscholar.org:F2:A10 - Expand TLDR or abstract section to extract fasting method details",
        parent=semantic_node,
        critical=False
    )

    # 3.2 WebMD/Mayo Clinic section
    medical_node = evaluator.add_sequential(
        id="medical_section",
        desc="Medical information from WebMD or Mayo Clinic",
        parent=root,
        critical=False
    )

    # Check source mentioned
    medical_source_ok = has_any_ci(answer, ['webmd', 'mayo clinic'])
    evaluator.add_custom_node(
        result=bool(medical_source_ok),
        id="medical_source_mentioned",
        desc="Mentions WebMD or Mayo Clinic as the medical source",
        parent=medical_node,
        critical=False
    )

    # [Action Node] webmd.com:F7:A14 or mayoclinic.org:F2:A3 - Navigate to detail page
    has_clinical_content = bool(medical_info.clinical_recommendations)
    evaluator.add_custom_node(
        result=bool(has_clinical_content and medical_source_ok),
        id="medical_detail_navigation",
        desc="[Action Node] webmd.com:F7:A14 or mayoclinic.org:F2:A3 - Navigate from search results to medical detail page",
        parent=medical_node,
        critical=False
    )

    # [Action Node] webmd.com:F7:A10 or mayoclinic.org:F2:A4 - Tab switching
    has_safety_content = bool(medical_info.safety_considerations)
    tab_switching_ok = has_clinical_content and has_safety_content
    evaluator.add_custom_node(
        result=bool(tab_switching_ok),
        id="medical_tab_switching",
        desc="[Action Node] webmd.com:F7:A10 or mayoclinic.org:F2:A4 - Switch between Overview/Treatment tabs to gather information",
        parent=medical_node,
        critical=False
    )

    # [Perception Node] webmd.com:F7:P10 or mayoclinic.org:F2:P2 - Extract medical info
    has_fasting_windows = bool(medical_info.recommended_fasting_windows)
    medical_extraction_ok = has_clinical_content and has_safety_content and has_fasting_windows
    evaluator.add_custom_node(
        result=bool(medical_extraction_ok),
        id="medical_info_extraction",
        desc="[Perception Node] webmd.com:F7:P10 or mayoclinic.org:F2:P2 - Extract clinical recommendations, safety considerations, and fasting windows",
        parent=medical_node,
        critical=False
    )

    # Check contraindications mentioned
    contraindications_ok = has_any_ci(str(medical_info.safety_considerations), ['contraindication', 'not recommended', 'avoid', 'should not'])
    evaluator.add_custom_node(
        result=bool(contraindications_ok),
        id="medical_contraindications",
        desc="Mentions contraindications or groups who should avoid intermittent fasting",
        parent=medical_node,
        critical=False
    )

    # 3.3 AllRecipes section
    allrecipes_node = evaluator.add_sequential(
        id="allrecipes_section",
        desc="AllRecipes high protein low calorie recipes",
        parent=root,
        critical=False
    )

    # [Action Node] allrecipes.com:F1:A2 - Search submission
    allrecipes_mentioned = has_any_ci(answer, ['allrecipes'])
    search_terms_ok = has_any_ci(answer, ['high protein', 'low calorie'])
    evaluator.add_custom_node(
        result=bool(allrecipes_mentioned and search_terms_ok),
        id="allrecipes_search",
        desc="[Action Node] allrecipes.com:F1:A2 - Search for 'high protein low calorie' recipes",
        parent=allrecipes_node,
        critical=False
    )

    # [Action Node] allrecipes.com:F3:A3 - Click into recipe details
    found_three_allrecipes = len(allrecipes_data.recipes) >= 3
    evaluator.add_custom_node(
        result=bool(found_three_allrecipes),
        id="allrecipes_detail_access",
        desc="[Action Node] allrecipes.com:F3:A3 - Access recipe detail pages to extract prep time and calories",
        parent=allrecipes_node,
        critical=False
    )

    # Check prep time and calorie constraints
    valid_allrecipes = [
        r for r in allrecipes_data.recipes
        if looks_like_prep_time_under_30(r.prep_time) and looks_like_calories_in_range(r.calories, 300, 500)
    ]
    allrecipes_filters_ok = len(valid_allrecipes) >= 3
    evaluator.add_custom_node(
        result=bool(allrecipes_filters_ok),
        id="allrecipes_filters",
        desc="Found 3 recipes with prep time <30 min and calories between 300-500 kcal",
        parent=allrecipes_node,
        critical=False
    )

    # 3.4 Food Network section
    foodnetwork_node = evaluator.add_sequential(
        id="foodnetwork_section",
        desc="Food Network healthy dinner recipes",
        parent=root,
        critical=False
    )

    # Check Food Network mentioned
    foodnetwork_mentioned = has_any_ci(answer, ['food network'])
    evaluator.add_custom_node(
        result=bool(foodnetwork_mentioned),
        id="foodnetwork_mentioned",
        desc="Mentions Food Network as the recipe source",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F1:P2 - Identify 5-star ratings
    recipes_with_5_star = [r for r in foodnetwork_data.recipes if looks_like_5_star(r.rating)]
    rating_identification_ok = len(recipes_with_5_star) >= 2
    evaluator.add_custom_node(
        result=bool(rating_identification_ok),
        id="foodnetwork_rating_identification",
        desc="[Perception Node] foodnetwork.com:F1:P2 - Identify recipes with 5-star ratings",
        parent=foodnetwork_node,
        critical=False
    )

    # [Action Node] foodnetwork.com:F1:A15 - Pagination to find qualifying recipes
    found_two_foodnetwork = len(foodnetwork_data.recipes) >= 2
    evaluator.add_custom_node(
        result=bool(found_two_foodnetwork),
        id="foodnetwork_pagination",
        desc="[Action Node] foodnetwork.com:F1:A15 - Navigate search results to find 2 recipes meeting both criteria",
        parent=foodnetwork_node,
        critical=False
    )

    # [Action Node] foodnetwork.com:F1:A6 - Click into recipe details
    recipes_with_links = [r for r in foodnetwork_data.recipes if r.link]
    foodnetwork_detail_ok = len(recipes_with_links) >= 2
    evaluator.add_custom_node(
        result=bool(foodnetwork_detail_ok),
        id="foodnetwork_detail_access",
        desc="[Action Node] foodnetwork.com:F1:A6 - Access recipe detail pages",
        parent=foodnetwork_node,
        critical=False
    )

    # [Action Node] foodnetwork.com:F2:A10 - Expand nutrition info
    recipes_with_protein = [r for r in foodnetwork_data.recipes if r.protein]
    nutrition_expansion_ok = len(recipes_with_protein) >= 2
    evaluator.add_custom_node(
        result=bool(nutrition_expansion_ok),
        id="foodnetwork_nutrition_expansion",
        desc="[Action Node] foodnetwork.com:F2:A10 - Expand nutritional information to view protein content",
        parent=foodnetwork_node,
        critical=False
    )

    # [Perception Node] foodnetwork.com:F2:P3 - Extract and verify protein >25g
    recipes_with_high_protein = [r for r in foodnetwork_data.recipes if looks_like_protein_over_25(r.protein)]
    protein_verification_ok = len(recipes_with_high_protein) >= 2
    evaluator.add_custom_node(
        result=bool(protein_verification_ok),
        id="foodnetwork_protein_verification",
        desc="[Perception Node] foodnetwork.com:F2:P3 - Extract protein content and verify >25 grams",
        parent=foodnetwork_node,
        critical=False
    )

    # Check calorie constraint for Food Network
    recipes_under_400_cal = [r for r in foodnetwork_data.recipes if looks_like_calories_in_range(r.calories, 0, 400)]
    foodnetwork_calorie_ok = len(recipes_under_400_cal) >= 2
    evaluator.add_custom_node(
        result=bool(foodnetwork_calorie_ok),
        id="foodnetwork_calorie_filter",
        desc="Found 2 recipes with calories under 400 kcal",
        parent=foodnetwork_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
