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
TASK_ID = "task-8acc4f"
TASK_DESCRIPTION = 'I came across a survey paper on 3D Gaussian Splatting on arXiv. The title is roughly **“A Survey on 3D Gaussian Splatting”**, and the authors seem to be Guo et al. I want to find the latest code implementations mentioned in or related to this survey.\n\nFirst, help me confirm the paper’s exact title and arXiv ID on arXiv. Then search GitHub using the paper title, find related repositories, and prioritize them by Star count (highest to lowest), while also checking whether they have been updated within the last three months. If any repository satisfies both **“high stars + updated in the last three months,”** return the one with the highest Star count among them.\n\nNext, check whether that repository’s README includes a **“Requirements”** or **“Installation”** section, and confirm that it is PyTorch-based.\n\nFinally, search Hugging Face to see whether there is a Papers page with the same title.  \n- If yes, identify the linked Model with the highest download count.  \n- If no such Papers page exists, search Models using the paper title keywords, pick a highly downloaded relevant model, and state the matching rationale.\n\nPlease compile and return the **arXiv ID**, **GitHub repository link**, and the **most popular model ID on Hugging Face**.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ArxivInfo(BaseModel):
    """ArXiv paper details extracted from the answer"""
    paper_title: Optional[str] = None
    arxiv_id: Optional[str] = None
    authors_mention: Optional[str] = None


class GitHubRepoInfo(BaseModel):
    """GitHub repository details extracted from the answer"""
    repo_url: Optional[str] = None
    repo_name: Optional[str] = None
    star_count_text: Optional[str] = None
    last_update_text: Optional[str] = None
    has_requirements_or_installation: Optional[bool] = None
    is_pytorch_based: Optional[bool] = None


class HuggingFaceInfo(BaseModel):
    """Hugging Face model details extracted from the answer"""
    has_papers_page: Optional[bool] = None
    model_id: Optional[str] = None
    download_count_text: Optional[str] = None
    matching_rationale: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_arxiv_from_answer() -> str:
    return """
Extract the arXiv paper information from the answer.

Return:
- paper_title: the exact title of the paper as confirmed on arXiv.
- arxiv_id: the arXiv ID (e.g., "2401.03890" or similar format).
- authors_mention: any mention of authors, especially "Guo" or "Guo et al."

If any field is missing in the answer, set it to null.
"""


def prompt_extract_github_from_answer() -> str:
    return """
Extract the GitHub repository information from the answer.

Return:
- repo_url: the full URL to the GitHub repository.
- repo_name: the repository name (e.g., "owner/repo").
- star_count_text: the star count as stated in the answer (e.g., "1.5k stars", "2000 stars").
- last_update_text: information about the last update or commit time.
- has_requirements_or_installation: whether the README includes a "Requirements" or "Installation" section (true/false).
- is_pytorch_based: whether the repository is confirmed to be PyTorch-based (true/false).

If any field is missing, set it to null.
"""


def prompt_extract_huggingface_from_answer() -> str:
    return """
Extract the Hugging Face model information from the answer.

Return:
- has_papers_page: whether a Papers page with the same title was found (true/false).
- model_id: the Hugging Face model ID (e.g., "username/model-name").
- download_count_text: the download count as stated in the answer.
- matching_rationale: if no Papers page exists, the rationale for selecting this model.

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


def looks_like_arxiv_id(text: Optional[str]) -> bool:
    if not text:
        return False
    # ArXiv IDs typically match patterns like YYMM.NNNNN or YYMM.NNNNNVN
    return bool(re.search(r'\d{4}\.\d{4,5}(v\d+)?', text))


def looks_like_github_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return ci_contains(text, 'github.com') and '/' in text


def extract_number_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Handle formats like "1.5k", "2000", "1,500", etc.
    text_clean = text.replace(',', '')
    m = re.search(r'(\d+(\.\d+)?)\s*k', text_clean, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) * 1000
        except Exception:
            return None
    m = re.search(r'(\d+(\.\d+)?)', text_clean)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def mentions_three_months_update(text: Optional[str]) -> bool:
    if not text:
        return False
    return has_any_ci(text, ['three months', '3 months', 'last three months', 'within three months', 'recent', 'recently updated'])


def looks_like_hf_model_id(text: Optional[str]) -> bool:
    if not text:
        return False
    # HF model IDs typically have format "username/model-name"
    return bool(re.search(r'[\w-]+/[\w-]+', text))


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
    arxiv_info = await evaluator.extract(
        prompt=prompt_extract_arxiv_from_answer(),
        template_class=ArxivInfo,
        extraction_name="arxiv_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_from_answer(),
        template_class=GitHubRepoInfo,
        extraction_name="github_info"
    )

    hf_info = await evaluator.extract(
        prompt=prompt_extract_huggingface_from_answer(),
        template_class=HuggingFaceInfo,
        extraction_name="huggingface_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 ArXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="ArXiv paper confirmation for 3D Gaussian Splatting survey",
        parent=root,
        critical=False
    )

    # [Perception Node] arxiv.org:F1:P11 - Locate correct paper from fuzzy information
    title_ok = bool(arxiv_info.paper_title and len(arxiv_info.paper_title.strip()) > 10)
    title_mentions_3dgs = has_any_ci(arxiv_info.paper_title, ['3d gaussian splatting', 'gaussian splatting'])
    title_mentions_survey = has_any_ci(arxiv_info.paper_title, ['survey'])
    authors_ok = has_any_ci(arxiv_info.authors_mention, ['guo']) or has_any_ci(answer, ['guo'])
    arxiv_id_ok = looks_like_arxiv_id(arxiv_info.arxiv_id)

    arxiv_perception_ok = title_ok and title_mentions_3dgs and title_mentions_survey and arxiv_id_ok

    evaluator.add_custom_node(
        result=bool(arxiv_perception_ok),
        id="arxiv_perception_locate_paper",
        desc="[Perception Node] arxiv.org:F1:P11 - Locate correct paper from fuzzy information (survey, author Guo, 3D Gaussian Splatting)",
        parent=arxiv_node,
        critical=False
    )

    # Additional lenient check: mentions arXiv and provides ID
    mentions_arxiv = has_any_ci(answer, ['arxiv'])
    evaluator.add_custom_node(
        result=bool(mentions_arxiv and arxiv_id_ok),
        id="arxiv_provides_id",
        desc="Provides arXiv ID in correct format",
        parent=arxiv_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository search and validation",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by star count (highest to lowest)
    repo_url_ok = looks_like_github_url(github_info.repo_url)
    mentions_stars = has_any_ci(answer, ['star', 'stars', 'starred'])
    star_count_num = extract_number_from_text(github_info.star_count_text)
    star_sorting_ok = mentions_stars and (star_count_num is not None)

    evaluator.add_custom_node(
        result=bool(star_sorting_ok),
        id="github_action_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - Prioritize repositories by star count (highest to lowest)",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P1 - Check last commit time (last three months)
    update_time_ok = bool(github_info.last_update_text and len(github_info.last_update_text.strip()) > 0)
    mentions_three_months = mentions_three_months_update(answer) or mentions_three_months_update(github_info.last_update_text)

    evaluator.add_custom_node(
        result=bool(update_time_ok and mentions_three_months),
        id="github_perception_recent_update",
        desc="[Perception Node] github.com:F3:P1 - Check whether repository was updated within last three months",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F3:P22 - Scan README for Requirements/Installation section
    has_requirements_section = github_info.has_requirements_or_installation is True
    mentions_readme = has_any_ci(answer, ['readme'])

    evaluator.add_custom_node(
        result=bool(has_requirements_section),
        id="github_perception_readme_requirements",
        desc="[Perception Node] github.com:F3:P22 - Scan README for Requirements or Installation section",
        parent=github_node,
        critical=False
    )

    # Additional check: Confirms PyTorch-based
    pytorch_ok = github_info.is_pytorch_based is True
    mentions_pytorch = has_any_ci(answer, ['pytorch', 'torch'])
    evaluator.add_custom_node(
        result=bool(pytorch_ok or mentions_pytorch),
        id="github_pytorch_based",
        desc="Confirms repository is PyTorch-based",
        parent=github_node,
        critical=False
    )

    # Additional check: Provides valid GitHub URL
    evaluator.add_custom_node(
        result=bool(repo_url_ok),
        id="github_provides_url",
        desc="Provides valid GitHub repository URL",
        parent=github_node,
        critical=False
    )

    # 3.3 Hugging Face section
    hf_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="Hugging Face model search and identification",
        parent=root,
        critical=False
    )

    # [Perception Node] huggingface.co:F1:P1 - Identify model with highest download count
    model_id_ok = looks_like_hf_model_id(hf_info.model_id)
    download_count_ok = bool(hf_info.download_count_text and len(hf_info.download_count_text.strip()) > 0)
    mentions_downloads = has_any_ci(answer, ['download', 'downloads'])

    evaluator.add_custom_node(
        result=bool(model_id_ok and download_count_ok),
        id="hf_perception_highest_downloads",
        desc="[Perception Node] huggingface.co:F1:P1 - Identify model with highest download count",
        parent=hf_node,
        critical=False
    )

    # Additional checks: Papers page handling
    papers_page_checked = hf_info.has_papers_page is not None
    mentions_papers_page = has_any_ci(answer, ['papers page', 'papers'])
    evaluator.add_custom_node(
        result=bool(papers_page_checked),
        id="hf_checks_papers_page",
        desc="Checks whether a Papers page exists for the same title",
        parent=hf_node,
        critical=False
    )

    # If no Papers page, rationale should be provided
    rationale_ok = bool(hf_info.matching_rationale and len(hf_info.matching_rationale.strip()) > 10)
    if hf_info.has_papers_page is False:
        evaluator.add_custom_node(
            result=bool(rationale_ok),
            id="hf_provides_rationale",
            desc="Provides matching rationale when no Papers page exists",
            parent=hf_node,
            critical=False
        )

    # Additional check: Provides valid model ID
    evaluator.add_custom_node(
        result=bool(model_id_ok),
        id="hf_provides_model_id",
        desc="Provides valid Hugging Face model ID",
        parent=hf_node,
        critical=False
    )

    # 3.4 Final compilation check
    compilation_node = evaluator.add_parallel(
        id="final_compilation",
        desc="Final compilation of all three required outputs",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(arxiv_id_ok),
        id="compilation_arxiv_id",
        desc="Compiled arXiv ID is present and valid",
        parent=compilation_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(repo_url_ok),
        id="compilation_github_url",
        desc="Compiled GitHub repository link is present and valid",
        parent=compilation_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(model_id_ok),
        id="compilation_hf_model_id",
        desc="Compiled Hugging Face model ID is present and valid",
        parent=compilation_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
