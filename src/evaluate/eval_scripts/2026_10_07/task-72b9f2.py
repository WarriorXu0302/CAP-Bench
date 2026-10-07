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
TASK_ID = "task-72b9f2"
TASK_DESCRIPTION = 'I work in AI video generation and recently want to catch up on new technologies in the Text-to-Video domain.\nPlease help me search for models in this field on Hugging Face, filter them by the PyTorch framework, and sort them by their most recent update time.\nAmong the first two pages of results, identify the model with the highest download count. (Important: exclude models that are not solely Text-to-Video but also include other complex task tags; focus only on models dedicated to Text-to-Video).\nOnce found, navigate to its Model Card and check if it references an ArXiv paper. If it does, proceed to ArXiv to locate that paper. Pay particular attention to the core architecture mentioned in the Abstract (e.g., Diffusion-based or Transformer-based).\nFinally, report the model name, paper title, and core architecture.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    """Information about the identified model from HuggingFace"""
    model_name: Optional[str] = None
    download_count_text: Optional[str] = None
    framework_mention: Optional[str] = None
    sort_method_mention: Optional[str] = None


class PaperInfo(BaseModel):
    """Information about the ArXiv paper"""
    paper_title: Optional[str] = None
    core_architecture: Optional[str] = None


class TaskTagsInfo(BaseModel):
    """Information about task tags filtering"""
    mentioned_exclusion_logic: Optional[str] = None
    task_tags_mentioned: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_model_info() -> str:
    return """
Extract the model identification details from the answer:

- model_name: the name of the HuggingFace model identified as having the highest download count (include full path if present, e.g., "username/model-name").
- download_count_text: the download count for this model as stated in the answer (include units if present).
- framework_mention: any mention of PyTorch framework filtering.
- sort_method_mention: any mention of sorting by recent update time or most recent updates.

If any field is missing in the answer, set it to null.
"""


def prompt_extract_paper_info() -> str:
    return """
Extract the ArXiv paper details from the answer:

- paper_title: the title of the ArXiv paper referenced in the model card, exactly as stated.
- core_architecture: the core architecture mentioned in the Abstract (e.g., "Diffusion-based", "Transformer-based", "GAN-based", etc.).

If any field is missing, set it to null.
"""


def prompt_extract_task_tags_info() -> str:
    return """
Extract information about how the answer handled task tag filtering:

- mentioned_exclusion_logic: any text explaining the exclusion of models with multiple complex task tags.
- task_tags_mentioned: any specific task tags mentioned in the filtering process.

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


def looks_like_model_name(text: Optional[str]) -> bool:
    if not text:
        return False
    # HuggingFace model names typically contain "/" or "-"
    return '/' in text or '-' in text or len(text.strip()) > 3


def looks_like_download_count(text: Optional[str]) -> bool:
    if not text:
        return False
    # Should contain digits, may contain "k", "M", "downloads", etc.
    return contains_digits(text)


def looks_like_paper_title(text: Optional[str]) -> bool:
    if not text:
        return False
    # Paper titles are usually substantial text
    return len(text.strip()) > 10


def looks_like_architecture(text: Optional[str]) -> bool:
    if not text:
        return False
    # Common architecture keywords
    arch_keywords = ['diffusion', 'transformer', 'gan', 'vae', 'unet', 'attention', 'autoregressive']
    return has_any_ci(text, arch_keywords)


def mentions_pytorch(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['pytorch', 'torch'])


def mentions_sort_by_update(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['recent', 'update', 'recently updated', 'last modified', 'sort'])


def mentions_pagination(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['first two pages', 'two pages', 'page 1', 'page 2', 'pagination'])


def mentions_text_to_video(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['text-to-video', 'text to video', 'text2video'])


def mentions_tag_exclusion(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['exclude', 'exclusion', 'solely', 'dedicated', 'only text-to-video', 'complex task'])


def mentions_arxiv(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['arxiv', 'arXiv'])


def mentions_abstract(answer_text: Optional[str]) -> bool:
    if not answer_text:
        return False
    return has_any_ci(answer_text, ['abstract'])


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
    model_info = await evaluator.extract(
        prompt=prompt_extract_model_info(),
        template_class=ModelInfo,
        extraction_name="model_info"
    )

    paper_info = await evaluator.extract(
        prompt=prompt_extract_paper_info(),
        template_class=PaperInfo,
        extraction_name="paper_info"
    )

    task_tags_info = await evaluator.extract(
        prompt=prompt_extract_task_tags_info(),
        template_class=TaskTagsInfo,
        extraction_name="task_tags_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 HuggingFace search and filtering section
    huggingface_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="HuggingFace model search, filtering, and sorting operations",
        parent=root,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A7 - Filter by PyTorch framework
    pytorch_filter_ok = mentions_pytorch(answer) and has_any_ci(answer, ['filter', 'framework'])
    evaluator.add_custom_node(
        result=bool(pytorch_filter_ok),
        id="huggingface_action_pytorch_filter",
        desc="[Action Node] huggingface.co:F2:A7 - Filter models by PyTorch framework using checkbox/filter selection",
        parent=huggingface_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A5 - Sort by most recent update time
    sort_by_update_ok = mentions_sort_by_update(answer)
    evaluator.add_custom_node(
        result=bool(sort_by_update_ok),
        id="huggingface_action_sort_recent",
        desc="[Action Node] huggingface.co:F2:A5 - Sort models by most recent update time using dropdown menu",
        parent=huggingface_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A4 - Navigate through first two pages
    pagination_ok = mentions_pagination(answer)
    evaluator.add_custom_node(
        result=bool(pagination_ok),
        id="huggingface_action_pagination",
        desc="[Action Node] huggingface.co:F2:A4 - Navigate through the first two pages of search results",
        parent=huggingface_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F2:P9 - Identify and exclude models with multiple task tags
    tag_exclusion_ok = mentions_tag_exclusion(answer)
    text_to_video_ok = mentions_text_to_video(answer)
    evaluator.add_custom_node(
        result=bool(tag_exclusion_ok and text_to_video_ok),
        id="huggingface_perception_task_tags",
        desc="[Perception Node] huggingface.co:F2:P9 - Identify task tags on model cards and exclude models with multiple complex task tags (not solely Text-to-Video)",
        parent=huggingface_node,
        critical=False
    )

    # Additional check: Identified model with highest download count
    model_name_ok = looks_like_model_name(model_info.model_name)
    download_count_ok = looks_like_download_count(model_info.download_count_text)
    highest_download_mention = has_any_ci(answer, ['highest', 'most downloads', 'top'])

    evaluator.add_custom_node(
        result=bool(model_name_ok and download_count_ok and highest_download_mention),
        id="huggingface_identify_top_model",
        desc="Identified the model with the highest download count among filtered results",
        parent=huggingface_node,
        critical=False
    )

    # 3.2 Model Card and ArXiv navigation section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="Navigate to model card, find ArXiv reference, and extract paper details",
        parent=root,
        critical=False
    )

    # Check if answer mentions navigating to model card
    model_card_mention = has_any_ci(answer, ['model card', 'modelcard', 'model page'])
    evaluator.add_custom_node(
        result=bool(model_card_mention),
        id="arxiv_action_model_card",
        desc="Navigate to the model's Model Card page on HuggingFace",
        parent=arxiv_node,
        critical=False
    )

    # Check if answer mentions finding ArXiv reference
    arxiv_reference_ok = mentions_arxiv(answer) and has_any_ci(answer, ['reference', 'paper', 'link'])
    evaluator.add_custom_node(
        result=bool(arxiv_reference_ok),
        id="arxiv_action_find_reference",
        desc="Locate ArXiv paper reference in the Model Card",
        parent=arxiv_node,
        critical=False
    )

    # Check if answer mentions navigating to ArXiv
    arxiv_navigation_ok = mentions_arxiv(answer) and has_any_ci(answer, ['navigate', 'go to', 'visit', 'proceed'])
    evaluator.add_custom_node(
        result=bool(arxiv_navigation_ok),
        id="arxiv_action_navigate",
        desc="Navigate to ArXiv to access the paper",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P5 - Extract core architecture from Abstract
    abstract_mention_ok = mentions_abstract(answer)
    paper_title_ok = looks_like_paper_title(paper_info.paper_title)
    architecture_ok = looks_like_architecture(paper_info.core_architecture)

    evaluator.add_custom_node(
        result=bool(abstract_mention_ok and architecture_ok),
        id="arxiv_perception_architecture",
        desc="[Perception Node] arxiv.org:F3:P5 - Read the Abstract and identify the core architecture (e.g., Diffusion-based, Transformer-based)",
        parent=arxiv_node,
        critical=False
    )

    # 3.3 Final reporting section
    reporting_node = evaluator.add_parallel(
        id="reporting_section",
        desc="Final report contains all required information",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(model_name_ok),
        id="report_model_name",
        desc="Report includes the model name",
        parent=reporting_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(paper_title_ok),
        id="report_paper_title",
        desc="Report includes the paper title",
        parent=reporting_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(architecture_ok),
        id="report_architecture",
        desc="Report includes the core architecture",
        parent=reporting_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
