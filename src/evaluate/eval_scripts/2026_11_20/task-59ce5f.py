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
TASK_ID = "task-59ce5f"
TASK_DESCRIPTION = "I need to select a sentiment analysis model for my company's FinTech project.\n\nPlease go to Hugging Face and search for the keyword 'financial sentiment'. Use the left-hand filter bar to set the task to 'Text Classification' and library support to 'PyTorch'. Sort by 'Most Downloads' and select the top 3 models with the highest download counts.\n\nFor each of these 3 models, navigate to its detail page to find the associated GitHub repository link. After navigating to GitHub, click to view the commit history, and record the specific date of the most recent commit on the main branch to assess maintenance activity. Additionally, look for the arXiv paper link referenced by the model, either on the HF model card or in the GitHub README.\n\nOutput the following information for these 3 models: Model Name, HF Detail Page Link, HF Download Count, GitHub Repository Link, GitHub Most Recent Commit Date, arXiv Paper Link (if none, state 'N/A')."


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    """Information for a single model"""
    model_name: Optional[str] = None
    hf_detail_page_link: Optional[str] = None
    hf_download_count: Optional[str] = None
    github_repository_link: Optional[str] = None
    github_most_recent_commit_date: Optional[str] = None
    arxiv_paper_link: Optional[str] = None


class ModelsExtraction(BaseModel):
    """All three models extracted from the answer"""
    model_1: Optional[ModelInfo] = None
    model_2: Optional[ModelInfo] = None
    model_3: Optional[ModelInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_models_from_answer() -> str:
    return """
Extract information for the top 3 models reported in the answer. The user was asked to find models from Hugging Face matching 'financial sentiment' with filters for 'Text Classification' and 'PyTorch', sorted by 'Most Downloads'.

For each of the 3 models, extract:
- model_name: the model's name exactly as stated
- hf_detail_page_link: the Hugging Face detail page URL
- hf_download_count: the download count exactly as stated (may include commas, 'k', 'M', etc.)
- github_repository_link: the GitHub repository URL
- github_most_recent_commit_date: the date of the most recent commit on the main branch
- arxiv_paper_link: the arXiv paper link, or "N/A" if not found

If any field is missing for a model, set it to null. If fewer than 3 models are present, set the missing model objects to null.
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


def looks_like_hf_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'huggingface.co')


def looks_like_github_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'github.com')


def looks_like_arxiv_url(url: Optional[str]) -> bool:
    if not url:
        return False
    return ci_contains(url, 'arxiv.org')


def looks_like_date(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept various date formats: YYYY-MM-DD, MM/DD/YYYY, Month DD, YYYY, etc.
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\w+ \d{1,2},? \d{4}',
        r'\d{1,2} \w+ \d{4}'
    ]
    return any(re.search(p, text) for p in date_patterns)


def looks_like_download_count(text: Optional[str]) -> bool:
    if not text:
        return False
    # Accept numbers with commas, k, M, or plain digits
    return bool(re.search(r'\d', text))


def extract_model_list(extraction: ModelsExtraction) -> List[ModelInfo]:
    models = []
    if extraction.model_1:
        models.append(extraction.model_1)
    if extraction.model_2:
        models.append(extraction.model_2)
    if extraction.model_3:
        models.append(extraction.model_3)
    return models


def check_downloads_descending(models: List[ModelInfo]) -> bool:
    """Check if download counts appear in descending order (lenient)"""
    if len(models) < 2:
        return True

    counts = []
    for m in models:
        if not m.hf_download_count:
            return False
        # Try to extract numeric value
        text = m.hf_download_count.replace(',', '')
        if 'k' in text.lower():
            text = text.lower().replace('k', '000')
        elif 'm' in text.lower():
            text = text.lower().replace('m', '000000')

        match = re.search(r'(\d+(\.\d+)?)', text)
        if match:
            try:
                counts.append(float(match.group(1)))
            except:
                return False
        else:
            return False

    # Check descending
    for i in range(len(counts) - 1):
        if counts[i] < counts[i + 1]:
            return False
    return True


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
    models_data = await evaluator.extract(
        prompt=prompt_extract_models_from_answer(),
        template_class=ModelsExtraction,
        extraction_name="top_3_models"
    )

    models_list = extract_model_list(models_data)

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Hugging Face search and filtering
    hf_search_node = evaluator.add_sequential(
        id="hf_search_and_filter",
        desc="Hugging Face search with filters and sorting",
        parent=root,
        critical=False
    )

    # [Action Node] huggingface.co:F1:A9 - Search for 'financial sentiment'
    search_action_ok = has_any_ci(answer, ['financial sentiment', 'hugging face', 'huggingface'])
    evaluator.add_custom_node(
        result=bool(search_action_ok),
        id="hf_search_action",
        desc="[Action Node] huggingface.co:F1:A9 - Search for keyword 'financial sentiment' on Hugging Face",
        parent=hf_search_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A6 - Filter by Text Classification
    text_classification_ok = has_any_ci(answer, ['text classification'])
    evaluator.add_custom_node(
        result=bool(text_classification_ok),
        id="hf_filter_text_classification",
        desc="[Action Node] huggingface.co:F2:A6 - Filter task to 'Text Classification' using checkbox",
        parent=hf_search_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A7 - Filter by PyTorch
    pytorch_ok = has_any_ci(answer, ['pytorch'])
    evaluator.add_custom_node(
        result=bool(pytorch_ok),
        id="hf_filter_pytorch",
        desc="[Action Node] huggingface.co:F2:A7 - Filter library to 'PyTorch' using checkbox",
        parent=hf_search_node,
        critical=False
    )

    # [Action Node] huggingface.co:F2:A5 - Sort by Most Downloads
    sort_downloads_ok = has_any_ci(answer, ['most downloads', 'download']) and check_downloads_descending(models_list)
    evaluator.add_custom_node(
        result=bool(sort_downloads_ok),
        id="hf_sort_most_downloads",
        desc="[Action Node] huggingface.co:F2:A5 - Sort by 'Most Downloads' and verify descending order",
        parent=hf_search_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F2:P9 - Verify filtered results match criteria
    all_models_valid = len(models_list) == 3 and all(
        m.model_name and looks_like_hf_url(m.hf_detail_page_link) and looks_like_download_count(m.hf_download_count)
        for m in models_list
    )
    evaluator.add_custom_node(
        result=bool(all_models_valid),
        id="hf_state_awareness",
        desc="[Perception Node] huggingface.co:F2:P9 - Verify 3 models extracted with proper filtering applied",
        parent=hf_search_node,
        critical=False
    )

    # 3.2 Model details extraction
    models_details_node = evaluator.add_parallel(
        id="models_details",
        desc="Extract details for each of the 3 models",
        parent=root,
        critical=False
    )

    for idx, model_info in enumerate(models_list, 1):
        model_node = evaluator.add_sequential(
            id=f"model_{idx}_details",
            desc=f"Model {idx}: {model_info.model_name if model_info.model_name else 'Unknown'}",
            parent=models_details_node,
            critical=False
        )

        # Basic info present
        basic_info_ok = bool(
            model_info.model_name and
            looks_like_hf_url(model_info.hf_detail_page_link) and
            looks_like_download_count(model_info.hf_download_count)
        )
        evaluator.add_custom_node(
            result=bool(basic_info_ok),
            id=f"model_{idx}_basic_info",
            desc=f"Model {idx} has name, HF link, and download count",
            parent=model_node,
            critical=False
        )

        # [Perception Node] github.com:F3:P14 - Locate GitHub link from HF page
        github_link_ok = looks_like_github_url(model_info.github_repository_link)
        evaluator.add_custom_node(
            result=bool(github_link_ok),
            id=f"model_{idx}_github_link",
            desc=f"[Perception Node] github.com:F3:P14 - Model {idx} has valid GitHub repository link",
            parent=model_node,
            critical=False
        )

        # [Action Node] github.com:F10:A24 - Navigate to commit history
        commit_date_ok = looks_like_date(model_info.github_most_recent_commit_date)
        evaluator.add_custom_node(
            result=bool(commit_date_ok),
            id=f"model_{idx}_commit_action",
            desc=f"[Action Node] github.com:F10:A24 - Model {idx}: Navigate to commit history page",
            parent=model_node,
            critical=False
        )

        # [Perception Node] github.com:F10:P6 - Extract most recent commit date
        evaluator.add_custom_node(
            result=bool(commit_date_ok),
            id=f"model_{idx}_commit_date_perception",
            desc=f"[Perception Node] github.com:F10:P6 - Model {idx}: Extract specific date of most recent commit",
            parent=model_node,
            critical=False
        )

        # [Action Node] huggingface.co:F3:A3 - Tab switching to find arXiv link
        arxiv_present = (
            model_info.arxiv_paper_link and
            (looks_like_arxiv_url(model_info.arxiv_paper_link) or
             ci_contains(model_info.arxiv_paper_link, 'n/a'))
        )
        evaluator.add_custom_node(
            result=bool(arxiv_present),
            id=f"model_{idx}_arxiv_tab_switching",
            desc=f"[Action Node] huggingface.co:F3:A3 - Model {idx}: Search for arXiv link (may require tab switching on HF)",
            parent=model_node,
            critical=False
        )

        # Verify arXiv link is valid or explicitly marked N/A
        arxiv_valid = False
        if model_info.arxiv_paper_link:
            if looks_like_arxiv_url(model_info.arxiv_paper_link):
                arxiv_valid = True
            elif ci_contains(model_info.arxiv_paper_link, 'n/a'):
                arxiv_valid = True

        evaluator.add_custom_node(
            result=bool(arxiv_valid),
            id=f"model_{idx}_arxiv_link",
            desc=f"Model {idx} has arXiv link or explicitly states 'N/A'",
            parent=model_node,
            critical=False
        )

    # 3.3 Overall completeness check
    all_complete = len(models_list) == 3 and all(
        m.model_name and
        looks_like_hf_url(m.hf_detail_page_link) and
        looks_like_download_count(m.hf_download_count) and
        looks_like_github_url(m.github_repository_link) and
        looks_like_date(m.github_most_recent_commit_date) and
        m.arxiv_paper_link
        for m in models_list
    )

    evaluator.add_custom_node(
        result=bool(all_complete),
        id="all_models_complete",
        desc="All 3 models have complete information (name, HF link, downloads, GitHub link, commit date, arXiv link/N/A)",
        parent=root,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
