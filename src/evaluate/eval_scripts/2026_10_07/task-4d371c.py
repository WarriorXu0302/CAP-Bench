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
TASK_ID = "task-4d371c"
TASK_DESCRIPTION = "I want to undertake a systematic study of the 'Llama 3' model.\n\nFirst, please search for 'Llama 3' on HuggingFace and identify the model version with the highest download count that is tagged for 'text-generation'. Then, navigate to its Model Card. Specifically, examine the 'Uses' section for applicable scenarios and note the Deep Learning Framework (e.g., PyTorch or TensorFlow) indicated in the sidebar.\n\nOnce the framework and task type are confirmed, proceed to Coursera to find three relevant Specializations. These Specializations must include either 'Generative AI' or 'Natural Language Processing' as keywords, be rated as 'Intermediate' difficulty, and have a minimum rating of 4.5 stars. Please browse through multiple pages to check for courses offered by reputable institutions such as DeepLearning.AI or Stanford.\n\nFinally, provide me with the name of the identified model and the names, institutions, and ratings for the three selected courses."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HuggingFaceModelInfo(BaseModel):
    """Model information extracted from HuggingFace"""
    model_name: Optional[str] = None
    has_text_generation_tag: Optional[bool] = None
    framework_mentioned: Optional[str] = None
    uses_section_mentioned: Optional[bool] = None


class CourseraCoursesInfo(BaseModel):
    """Coursera courses information extracted from the answer"""
    course1_name: Optional[str] = None
    course1_institution: Optional[str] = None
    course1_rating: Optional[str] = None

    course2_name: Optional[str] = None
    course2_institution: Optional[str] = None
    course2_rating: Optional[str] = None

    course3_name: Optional[str] = None
    course3_institution: Optional[str] = None
    course3_rating: Optional[str] = None

    mentions_intermediate_difficulty: Optional[bool] = None
    mentions_multiple_pages: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_huggingface_model() -> str:
    return """
Extract the HuggingFace model information from the answer:

- model_name: the exact name of the Llama 3 model identified (e.g., "meta-llama/Meta-Llama-3-8B"). If not present, set null.
- has_text_generation_tag: true if the answer mentions that the model is tagged for 'text-generation', false if explicitly says it's not, null if not discussed.
- framework_mentioned: the Deep Learning Framework mentioned (e.g., "PyTorch", "TensorFlow"). If not present, set null.
- uses_section_mentioned: true if the answer discusses examining or mentioning the 'Uses' section of the Model Card, false otherwise.

If any field is missing, set it to null.
"""


def prompt_extract_coursera_courses() -> str:
    return """
Extract the three Coursera Specializations information from the answer:

For each of the three courses, extract:
- courseN_name: the exact name of the specialization
- courseN_institution: the institution offering it (e.g., "DeepLearning.AI", "Stanford")
- courseN_rating: the rating value (e.g., "4.8", "4.7")

Also extract:
- mentions_intermediate_difficulty: true if the answer mentions filtering or checking for 'Intermediate' difficulty
- mentions_multiple_pages: true if the answer mentions browsing through multiple pages

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
    m = re.findall(r'(\d+(\.\d+)?)', text)
    if not m:
        return None
    try:
        return float(m[0][0])
    except Exception:
        return None


def looks_like_valid_rating(text: Optional[str], min_rating: float = 4.5) -> bool:
    if not text:
        return False
    rating = extract_float(text)
    if rating is None:
        return False
    return rating >= min_rating


def mentions_llama3(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['llama 3', 'llama-3', 'llama3', 'meta-llama'])


def mentions_huggingface(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['huggingface', 'hugging face', 'hf'])


def mentions_coursera(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['coursera'])


def is_reputable_institution(institution: Optional[str]) -> bool:
    if not institution:
        return False
    return has_any_ci(institution, ['deeplearning.ai', 'deeplearning', 'stanford', 'deep learning'])


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
    Restrict evaluator.verify to at most one usage (not used here).
    Favor lenient, fault-tolerant checks and allow partial credit.
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
    hf_info = await evaluator.extract(
        prompt=prompt_extract_huggingface_model(),
        template_class=HuggingFaceModelInfo,
        extraction_name="huggingface_model_info"
    )

    coursera_info = await evaluator.extract(
        prompt=prompt_extract_coursera_courses(),
        template_class=CourseraCoursesInfo,
        extraction_name="coursera_courses_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 HuggingFace section
    huggingface_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="HuggingFace model identification and analysis",
        parent=root,
        critical=False
    )

    # Basic navigation to HuggingFace (non-prefixed)
    hf_navigation_ok = mentions_huggingface(answer) and mentions_llama3(answer)
    evaluator.add_custom_node(
        result=bool(hf_navigation_ok),
        id="huggingface_navigation",
        desc="Navigate to HuggingFace and search for Llama 3",
        parent=huggingface_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F1:P9 - Identify text-generation tag
    text_gen_tag_ok = bool(hf_info and hf_info.has_text_generation_tag)
    text_gen_mention_ok = has_any_ci(answer, ['text-generation', 'text generation'])
    evaluator.add_custom_node(
        result=bool(text_gen_tag_ok and text_gen_mention_ok),
        id="huggingface_text_generation_tag",
        desc="[Perception Node] huggingface.co:F1:P9 - Identify model with text-generation tag",
        parent=huggingface_node,
        critical=False
    )

    # Check for highest download count (non-prefixed)
    download_count_ok = has_any_ci(answer, ['download', 'downloads', 'most downloaded', 'highest download'])
    evaluator.add_custom_node(
        result=bool(download_count_ok),
        id="huggingface_highest_downloads",
        desc="Identify model with highest download count",
        parent=huggingface_node,
        critical=False
    )

    # Check Model Card navigation (non-prefixed)
    model_card_ok = has_any_ci(answer, ['model card', 'modelcard'])
    evaluator.add_custom_node(
        result=bool(model_card_ok),
        id="huggingface_model_card",
        desc="Navigate to Model Card",
        parent=huggingface_node,
        critical=False
    )

    # Check Uses section (non-prefixed)
    uses_section_ok = bool(hf_info and hf_info.uses_section_mentioned) or has_any_ci(answer, ['uses section', 'uses', 'applicable scenarios', 'use cases'])
    evaluator.add_custom_node(
        result=bool(uses_section_ok),
        id="huggingface_uses_section",
        desc="Examine the 'Uses' section for applicable scenarios",
        parent=huggingface_node,
        critical=False
    )

    # Check framework identification (non-prefixed)
    framework_ok = bool(hf_info and hf_info.framework_mentioned and len(hf_info.framework_mentioned.strip()) > 0)
    framework_keyword_ok = has_any_ci(answer, ['pytorch', 'tensorflow', 'framework'])
    evaluator.add_custom_node(
        result=bool(framework_ok and framework_keyword_ok),
        id="huggingface_framework",
        desc="Note the Deep Learning Framework from sidebar",
        parent=huggingface_node,
        critical=False
    )

    # Model name extracted (non-prefixed)
    model_name_ok = bool(hf_info and hf_info.model_name and len(hf_info.model_name.strip()) > 0)
    evaluator.add_custom_node(
        result=bool(model_name_ok),
        id="huggingface_model_name",
        desc="Provide the identified model name",
        parent=huggingface_node,
        critical=False
    )

    # 3.2 Coursera section
    coursera_node = evaluator.add_sequential(
        id="coursera_section",
        desc="Coursera Specializations search and selection",
        parent=root,
        critical=False
    )

    # Basic navigation to Coursera (non-prefixed)
    coursera_navigation_ok = mentions_coursera(answer)
    evaluator.add_custom_node(
        result=bool(coursera_navigation_ok),
        id="coursera_navigation",
        desc="Navigate to Coursera",
        parent=coursera_node,
        critical=False
    )

    # Check for relevant keywords (non-prefixed)
    keyword_check_ok = has_any_ci(answer, ['generative ai', 'natural language processing', 'nlp', 'gen ai'])
    evaluator.add_custom_node(
        result=bool(keyword_check_ok),
        id="coursera_keyword_search",
        desc="Search for Specializations with 'Generative AI' or 'Natural Language Processing' keywords",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A12 - Select Intermediate difficulty
    intermediate_filter_ok = bool(coursera_info and coursera_info.mentions_intermediate_difficulty) or has_any_ci(answer, ['intermediate'])
    evaluator.add_custom_node(
        result=bool(intermediate_filter_ok),
        id="coursera_intermediate_filter",
        desc="[Action Node] coursera.org:F1:A12 - Filter for Intermediate difficulty",
        parent=coursera_node,
        critical=False
    )

    # Check for 4.5+ rating requirement (non-prefixed)
    rating_requirement_ok = has_any_ci(answer, ['4.5', 'minimum rating', 'rating'])
    evaluator.add_custom_node(
        result=bool(rating_requirement_ok),
        id="coursera_rating_requirement",
        desc="Check for minimum 4.5 star rating requirement",
        parent=coursera_node,
        critical=False
    )

    # [Action Node] coursera.org:F1:A7 - Browse multiple pages
    multiple_pages_ok = bool(coursera_info and coursera_info.mentions_multiple_pages) or has_any_ci(answer, ['multiple pages', 'browse', 'page', 'next page'])
    evaluator.add_custom_node(
        result=bool(multiple_pages_ok),
        id="coursera_browse_pages",
        desc="[Action Node] coursera.org:F1:A7 - Browse through multiple pages",
        parent=coursera_node,
        critical=False
    )

    # [Perception Node] coursera.org:F3:P3 - Identify reputable institutions
    course1_inst_ok = is_reputable_institution(coursera_info.course1_institution) if coursera_info else False
    course2_inst_ok = is_reputable_institution(coursera_info.course2_institution) if coursera_info else False
    course3_inst_ok = is_reputable_institution(coursera_info.course3_institution) if coursera_info else False

    institution_mention_ok = has_any_ci(answer, ['deeplearning.ai', 'stanford', 'deep learning'])
    reputable_institution_ok = (course1_inst_ok or course2_inst_ok or course3_inst_ok) and institution_mention_ok

    evaluator.add_custom_node(
        result=bool(reputable_institution_ok),
        id="coursera_reputable_institutions",
        desc="[Perception Node] coursera.org:F3:P3 - Identify courses from reputable institutions (DeepLearning.AI or Stanford)",
        parent=coursera_node,
        critical=False
    )

    # Check three courses provided (non-prefixed)
    course1_ok = bool(coursera_info and coursera_info.course1_name)
    course2_ok = bool(coursera_info and coursera_info.course2_name)
    course3_ok = bool(coursera_info and coursera_info.course3_name)
    three_courses_ok = course1_ok and course2_ok and course3_ok

    evaluator.add_custom_node(
        result=bool(three_courses_ok),
        id="coursera_three_courses",
        desc="Provide three Specializations",
        parent=coursera_node,
        critical=False
    )

    # Check course details completeness (non-prefixed)
    course1_complete = bool(coursera_info and coursera_info.course1_name and
                           coursera_info.course1_institution and coursera_info.course1_rating)
    course2_complete = bool(coursera_info and coursera_info.course2_name and
                           coursera_info.course2_institution and coursera_info.course2_rating)
    course3_complete = bool(coursera_info and coursera_info.course3_name and
                           coursera_info.course3_institution and coursera_info.course3_rating)

    all_courses_complete = course1_complete and course2_complete and course3_complete

    evaluator.add_custom_node(
        result=bool(all_courses_complete),
        id="coursera_course_details",
        desc="Provide names, institutions, and ratings for all three courses",
        parent=coursera_node,
        critical=False
    )

    # Check ratings are 4.5+ (non-prefixed)
    course1_rating_ok = looks_like_valid_rating(coursera_info.course1_rating if coursera_info else None)
    course2_rating_ok = looks_like_valid_rating(coursera_info.course2_rating if coursera_info else None)
    course3_rating_ok = looks_like_valid_rating(coursera_info.course3_rating if coursera_info else None)

    all_ratings_ok = course1_rating_ok and course2_rating_ok and course3_rating_ok

    evaluator.add_custom_node(
        result=bool(all_ratings_ok),
        id="coursera_ratings_valid",
        desc="All three courses have ratings of 4.5 or higher",
        parent=coursera_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
