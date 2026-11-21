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
TASK_ID = "task-273c1b"
TASK_DESCRIPTION = 'I’ve recently been exploring fine-tuning techniques for large language models, and I heard there’s a classic paper on **“Llama 2”** fine-tuning (the title likely includes “Llama 2” and “Open Foundation”). Please help me find this paper on arXiv.  \n\nAfter finding it, I want to check whether there are any official or highly popular community reproductions on Hugging Face. First, extract the paper’s **Code link** or **project homepage** from arXiv (if no usable code/project link is provided on the page, then use the paper title or author names as search clues), and then search for the corresponding model on Hugging Face.  \n\nFind the version with the highest download count, and check whether its Model Card explicitly cites or mentions which dataset(s) were used. Also confirm whether this model is officially released by **meta-llama**.  \n\nFinally, tell me the **model name**, **whether it is official**, and the **dataset name(s) mentioned**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArxivPaperInfo(BaseModel):
    """Information extracted from the answer about the arXiv paper"""
    paper_title: Optional[str] = None
    code_link: Optional[str] = None
    project_homepage: Optional[str] = None
    search_clues_used: Optional[str] = None


class HuggingFaceModelInfo(BaseModel):
    """Information extracted from the answer about the Hugging Face model"""
    model_name: Optional[str] = None
    is_official: Optional[bool] = None
    is_meta_llama: Optional[bool] = None
    dataset_names: Optional[List[str]] = Field(default_factory=list)
    download_count_mentioned: Optional[bool] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arxiv_info() -> str:
    return """
Extract information about the arXiv paper the user found:

Return:
- paper_title: the title of the Llama 2 paper found on arXiv
- code_link: any code repository link extracted from the arXiv page
- project_homepage: any project homepage link extracted from the arXiv page
- search_clues_used: if no code/project links were found, what clues (title, authors, etc.) did the user mention using for the Hugging Face search

If any field is missing, set it to null or empty list.
"""


def prompt_extract_huggingface_info() -> str:
    return """
Extract information about the Hugging Face model the user found:

Return:
- model_name: the name/identifier of the model found
- is_official: whether the answer explicitly states this is an official model
- is_meta_llama: whether the answer confirms this model is from the meta-llama organization
- dataset_names: list of dataset names mentioned in the Model Card (empty list if none)
- download_count_mentioned: whether the answer mentions this model has the highest download count

If any field is missing, set appropriate defaults (null, False, or empty list).
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


def mentions_arxiv(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['arxiv', 'arXiv'])


def mentions_llama2(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['llama 2', 'llama2', 'llama-2'])


def mentions_open_foundation(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['open foundation', 'open-source'])


def mentions_huggingface(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['hugging face', 'huggingface', 'hf'])


def mentions_model_card(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['model card', 'modelcard', 'card'])


def mentions_meta_llama(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['meta-llama', 'meta llama', 'meta_llama'])


def mentions_download_or_popular(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['download', 'most popular', 'highest', 'most downloaded', 'trending'])


def has_dataset_names(datasets: Optional[List[str]]) -> bool:
    if not datasets:
        return False
    return len(datasets) > 0 and any(d and d.strip() for d in datasets)


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
    arxiv_info = await evaluator.extract(
        prompt=prompt_extract_arxiv_info(),
        template_class=ArxivPaperInfo,
        extraction_name="arxiv_paper_info"
    )

    hf_info = await evaluator.extract(
        prompt=prompt_extract_huggingface_info(),
        template_class=HuggingFaceModelInfo,
        extraction_name="huggingface_model_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 ArXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="ArXiv paper search and information extraction",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A1 - Search for paper with title filters
    arxiv_search_ok = (mentions_arxiv(answer) and
                       mentions_llama2(answer) and
                       (mentions_open_foundation(answer) or ci_contains(arxiv_info.paper_title, 'open foundation')))
    evaluator.add_custom_node(
        result=bool(arxiv_search_ok),
        id="arxiv_search_action",
        desc="[Action Node] arxiv.org:F1:A1 - Search arXiv for Llama 2 paper with title containing 'Llama 2' and 'Open Foundation'",
        parent=arxiv_node,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A20 - Extract code link or project homepage
    code_extraction_ok = bool(
        arxiv_info.code_link or
        arxiv_info.project_homepage or
        arxiv_info.search_clues_used or
        has_any_ci(answer, ['code', 'github', 'project', 'homepage', 'repository'])
    )
    evaluator.add_custom_node(
        result=bool(code_extraction_ok),
        id="arxiv_code_extraction",
        desc="[Action Node] arxiv.org:F3:A20 - Extract Code link or project homepage from arXiv paper page, or identify search clues",
        parent=arxiv_node,
        critical=False
    )

    # Additional check: mentions paper title was found
    paper_title_found = bool(arxiv_info.paper_title and arxiv_info.paper_title.strip())
    evaluator.add_custom_node(
        result=bool(paper_title_found),
        id="arxiv_paper_title_found",
        desc="Paper title was successfully identified",
        parent=arxiv_node,
        critical=False
    )

    # 3.2 Hugging Face section
    hf_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="Hugging Face model search and analysis",
        parent=root,
        critical=False
    )

    # Basic search action (non-prefixed)
    hf_search_ok = mentions_huggingface(answer) and mentions_llama2(answer)
    evaluator.add_custom_node(
        result=bool(hf_search_ok),
        id="hf_search_action",
        desc="Search Hugging Face for Llama 2 models",
        parent=hf_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F1:P1 - Identify highest download count model
    download_identified = bool(
        hf_info.download_count_mentioned or
        mentions_download_or_popular(answer)
    )
    evaluator.add_custom_node(
        result=bool(download_identified),
        id="hf_highest_downloads",
        desc="[Perception Node] huggingface.co:F1:P1 - Identify model with highest download count or popularity indicators",
        parent=hf_node,
        critical=False
    )

    # [Action Node] huggingface.co:F3:A3 - Navigate to Model Card
    model_card_visited = mentions_model_card(answer)
    evaluator.add_custom_node(
        result=bool(model_card_visited),
        id="hf_model_card_navigation",
        desc="[Action Node] huggingface.co:F3:A3 - Navigate to and read the Model Card (tab switch if needed)",
        parent=hf_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F3:P6 - Verify meta-llama official organization
    meta_llama_verified = bool(
        hf_info.is_meta_llama or
        (hf_info.is_official and mentions_meta_llama(answer))
    )
    evaluator.add_custom_node(
        result=bool(meta_llama_verified),
        id="hf_meta_llama_verification",
        desc="[Perception Node] huggingface.co:F3:P6 - Verify model is officially released by meta-llama organization",
        parent=hf_node,
        critical=False
    )

    # Dataset extraction from Model Card (non-prefixed)
    datasets_found = has_dataset_names(hf_info.dataset_names)
    evaluator.add_custom_node(
        result=bool(datasets_found),
        id="hf_dataset_extraction",
        desc="Extract dataset names mentioned in the Model Card",
        parent=hf_node,
        critical=False
    )

    # Model name identified (non-prefixed)
    model_name_found = bool(hf_info.model_name and hf_info.model_name.strip())
    evaluator.add_custom_node(
        result=bool(model_name_found),
        id="hf_model_name_found",
        desc="Model name was successfully identified",
        parent=hf_node,
        critical=False
    )

    # 3.3 Final output completeness
    output_node = evaluator.add_parallel(
        id="output_completeness",
        desc="Final answer contains all requested information",
        parent=root,
        critical=False
    )

    # Check if answer provides model name
    evaluator.add_custom_node(
        result=bool(model_name_found),
        id="output_model_name",
        desc="Answer provides the model name",
        parent=output_node,
        critical=False
    )

    # Check if answer states whether official
    official_stated = hf_info.is_official is not None or has_any_ci(answer, ['official', 'meta-llama'])
    evaluator.add_custom_node(
        result=bool(official_stated),
        id="output_is_official",
        desc="Answer states whether the model is official",
        parent=output_node,
        critical=False
    )

    # Check if answer provides dataset names
    evaluator.add_custom_node(
        result=bool(datasets_found),
        id="output_datasets",
        desc="Answer provides dataset name(s) mentioned in Model Card",
        parent=output_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
