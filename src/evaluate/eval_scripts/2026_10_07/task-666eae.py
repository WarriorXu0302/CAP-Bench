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
TASK_ID = "task-666eae"
TASK_DESCRIPTION = "I'm working on a Named Entity Recognition (NER) project and found the `dslim/bert-base-NER` model on HuggingFace to be effective. I need help understanding the research team behind this model.\n\nPlease perform the following steps:\n1.  Go to HuggingFace, locate the `dslim/bert-base-NER` model, and extract the associated paper link and author information from its Model Card.\n2.  Navigate to arXiv, find the paper, and note down the full name of the first author and their institutional affiliation.\n3.  Search for this first author on Semantic Scholar. Identify their top 5 most cited papers, listed in descending order of citation count.\n4.  Finally, search GitHub for the author's username (you can use their name and affiliation as keywords). Find their personal profile and list at least their top 3 open-source projects by star count.\n\nOutput format:\n*   Model Name\n*   HuggingFace Model Link\n*   Paper Title\n*   arXiv Link\n*   First Author's Full Name\n*   Author's Affiliation\n*   Author's Semantic Scholar Profile Link\n*   Author's 5 Most Cited Papers (for each: Paper Title, Publication Year, Citation Count, Paper Link)\n*   Author's GitHub Username\n*   GitHub Profile Link\n*   Author's 3 Highest-Starred Projects (for each: Project Name, Star Count, Project Link)"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ModelInfo(BaseModel):
    """Model information extracted from HuggingFace"""
    model_name: Optional[str] = None
    huggingface_link: Optional[str] = None
    paper_title: Optional[str] = None
    arxiv_link: Optional[str] = None


class AuthorInfo(BaseModel):
    """Author information extracted from arXiv paper"""
    first_author_name: Optional[str] = None
    author_affiliation: Optional[str] = None


class SemanticScholarInfo(BaseModel):
    """Semantic Scholar information"""
    semantic_scholar_link: Optional[str] = None
    top_papers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class GitHubInfo(BaseModel):
    """GitHub information"""
    github_username: Optional[str] = None
    github_profile_link: Optional[str] = None
    top_projects: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_model_info() -> str:
    return """
Extract the HuggingFace model information from the answer:

- model_name: the model identifier (e.g., "dslim/bert-base-NER")
- huggingface_link: the full URL to the HuggingFace model page
- paper_title: the title of the associated paper
- arxiv_link: the arXiv URL for the paper

If any field is missing, set it to null.
"""


def prompt_extract_author_info() -> str:
    return """
Extract the first author's information from the answer:

- first_author_name: the full name of the first author
- author_affiliation: the institutional affiliation of the first author

If any field is missing, set it to null.
"""


def prompt_extract_semantic_scholar_info() -> str:
    return """
Extract the Semantic Scholar information from the answer:

- semantic_scholar_link: the URL to the author's Semantic Scholar profile
- top_papers: a list of the top 5 most cited papers, each containing:
  - title: paper title
  - year: publication year
  - citations: citation count
  - link: paper link

If any field is missing, set it to null or an empty list.
"""


def prompt_extract_github_info() -> str:
    return """
Extract the GitHub information from the answer:

- github_username: the author's GitHub username
- github_profile_link: the URL to the author's GitHub profile
- top_projects: a list of the top 3 highest-starred projects, each containing:
  - name: project name
  - stars: star count
  - link: project link

If any field is missing, set it to null or an empty list.
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


def looks_like_url(text: Optional[str], domain: Optional[str] = None) -> bool:
    if not text:
        return False
    url_pattern = r'https?://[^\s]+'
    if not re.search(url_pattern, text):
        return False
    if domain:
        return domain.lower() in text.lower()
    return True


def looks_like_model_identifier(text: Optional[str]) -> bool:
    if not text:
        return False
    # Pattern like "dslim/bert-base-NER" or similar
    return bool(re.search(r'\w+/[\w-]+', text))


def looks_like_person_name(text: Optional[str]) -> bool:
    if not text:
        return False
    # At least two words, each capitalized
    words = text.split()
    return len(words) >= 2 and all(w[0].isupper() for w in words if w)


def looks_like_citation_count(value: Any) -> bool:
    if value is None:
        return False
    try:
        num = int(value) if not isinstance(value, int) else value
        return num >= 0
    except (ValueError, TypeError):
        return False


def looks_like_star_count(value: Any) -> bool:
    return looks_like_citation_count(value)


def is_descending_order(numbers: List[Any]) -> bool:
    if not numbers:
        return False
    try:
        nums = [int(n) if not isinstance(n, int) else n for n in numbers]
        return all(nums[i] >= nums[i+1] for i in range(len(nums)-1))
    except (ValueError, TypeError):
        return False


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
    model_info = await evaluator.extract(
        prompt=prompt_extract_model_info(),
        template_class=ModelInfo,
        extraction_name="model_info"
    )

    author_info = await evaluator.extract(
        prompt=prompt_extract_author_info(),
        template_class=AuthorInfo,
        extraction_name="author_info"
    )

    semantic_info = await evaluator.extract(
        prompt=prompt_extract_semantic_scholar_info(),
        template_class=SemanticScholarInfo,
        extraction_name="semantic_scholar_info"
    )

    github_info = await evaluator.extract(
        prompt=prompt_extract_github_info(),
        template_class=GitHubInfo,
        extraction_name="github_info"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 HuggingFace section
    huggingface_node = evaluator.add_sequential(
        id="huggingface_section",
        desc="HuggingFace model exploration and Model Card extraction",
        parent=root,
        critical=False
    )

    # [Action Node] huggingface.co:F1:A9 - Search for model
    model_search_ok = (
        looks_like_model_identifier(model_info.model_name) and
        ci_contains(model_info.model_name, 'bert-base-ner')
    )
    evaluator.add_custom_node(
        result=bool(model_search_ok),
        id="huggingface_search_model",
        desc="[Action Node] huggingface.co:F1:A9 - Search and locate the dslim/bert-base-NER model on HuggingFace",
        parent=huggingface_node,
        critical=False
    )

    # [Action Node] huggingface.co:F1:A10 - Click into model details
    huggingface_url_ok = looks_like_url(model_info.huggingface_link, 'huggingface.co')
    evaluator.add_custom_node(
        result=bool(huggingface_url_ok),
        id="huggingface_enter_details",
        desc="[Action Node] huggingface.co:F1:A10 - Navigate to the model's detail page",
        parent=huggingface_node,
        critical=False
    )

    # [Perception Node] huggingface.co:F3:P10 - Extract paper link and author info from Model Card
    paper_title_ok = bool(model_info.paper_title and len(model_info.paper_title.strip()) > 5)
    arxiv_link_ok = looks_like_url(model_info.arxiv_link, 'arxiv.org')
    evaluator.add_custom_node(
        result=bool(paper_title_ok and arxiv_link_ok),
        id="huggingface_extract_paper_info",
        desc="[Perception Node] huggingface.co:F3:P10 - Extract paper link and author information from the Model Card",
        parent=huggingface_node,
        critical=False
    )

    # 3.2 arXiv section
    arxiv_node = evaluator.add_sequential(
        id="arxiv_section",
        desc="arXiv paper exploration and author extraction",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F3:A6 - Navigate to paper details
    arxiv_access_ok = looks_like_url(model_info.arxiv_link, 'arxiv.org')
    evaluator.add_custom_node(
        result=bool(arxiv_access_ok),
        id="arxiv_enter_paper",
        desc="[Action Node] arxiv.org:F3:A6 - Navigate to the arXiv paper details page",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P4 - Identify first author and affiliation
    author_name_ok = looks_like_person_name(author_info.first_author_name)
    affiliation_ok = bool(author_info.author_affiliation and len(author_info.author_affiliation.strip()) > 3)
    evaluator.add_custom_node(
        result=bool(author_name_ok and affiliation_ok),
        id="arxiv_extract_author",
        desc="[Perception Node] arxiv.org:F3:P4 - Extract first author's full name and institutional affiliation",
        parent=arxiv_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F3:P5 - Understand paper content (title extraction)
    paper_understanding_ok = bool(model_info.paper_title and len(model_info.paper_title.strip()) > 5)
    evaluator.add_custom_node(
        result=bool(paper_understanding_ok),
        id="arxiv_understand_paper",
        desc="[Perception Node] arxiv.org:F3:P5 - Understand paper abstract and metadata, including title",
        parent=arxiv_node,
        critical=False
    )

    # 3.3 Semantic Scholar section
    semantic_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar author profile and citation analysis",
        parent=root,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A1 - Search for author
    semantic_search_ok = (
        looks_like_person_name(author_info.first_author_name) and
        has_any_ci(answer, ['semantic scholar'])
    )
    evaluator.add_custom_node(
        result=bool(semantic_search_ok),
        id="semantic_scholar_search_author",
        desc="[Action Node] semanticscholar.org:F1:A1 - Search for the first author on Semantic Scholar",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F4:A14 - Enter author profile
    semantic_profile_ok = looks_like_url(semantic_info.semantic_scholar_link, 'semanticscholar.org')
    evaluator.add_custom_node(
        result=bool(semantic_profile_ok),
        id="semantic_scholar_enter_profile",
        desc="[Action Node] semanticscholar.org:F4:A14 - Navigate to the author's profile page",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F4:P7 - Identify citation metrics
    top_papers = semantic_info.top_papers if semantic_info.top_papers else []
    has_five_papers = len(top_papers) >= 5
    citations_present = all(
        'citations' in p and looks_like_citation_count(p.get('citations'))
        for p in top_papers[:5]
    ) if has_five_papers else False
    evaluator.add_custom_node(
        result=bool(has_five_papers and citations_present),
        id="semantic_scholar_identify_citations",
        desc="[Perception Node] semanticscholar.org:F4:P7 - Identify citation counts for top papers",
        parent=semantic_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citation count (descending)
    citation_counts = [p.get('citations') for p in top_papers[:5]] if has_five_papers else []
    descending_ok = is_descending_order(citation_counts) if citation_counts else False
    evaluator.add_custom_node(
        result=bool(descending_ok),
        id="semantic_scholar_sort_citations",
        desc="[Action Node] semanticscholar.org:F1:A4 - List top 5 papers sorted by citation count in descending order",
        parent=semantic_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F1:P2 - Identify paper accessibility status
    papers_have_links = all(
        'link' in p and looks_like_url(p.get('link'))
        for p in top_papers[:5]
    ) if has_five_papers else False
    evaluator.add_custom_node(
        result=bool(papers_have_links),
        id="semantic_scholar_paper_accessibility",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Identify accessible links for top papers",
        parent=semantic_node,
        critical=False
    )

    # 3.4 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub author profile and project discovery",
        parent=root,
        critical=False
    )

    # [Action Node] github.com:F1:A10 - Search for author username
    github_search_ok = (
        bool(github_info.github_username and len(github_info.github_username.strip()) > 0) and
        has_any_ci(answer, ['github'])
    )
    evaluator.add_custom_node(
        result=bool(github_search_ok),
        id="github_search_username",
        desc="[Action Node] github.com:F1:A10 - Search for the author's GitHub username using name and affiliation",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F1:P1 - Understand search results and locate profile
    github_profile_ok = looks_like_url(github_info.github_profile_link, 'github.com')
    evaluator.add_custom_node(
        result=bool(github_profile_ok),
        id="github_locate_profile",
        desc="[Perception Node] github.com:F1:P1 - Locate and understand the author's GitHub profile from search results",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F7:A3 - Switch to Repositories tab
    top_projects = github_info.top_projects if github_info.top_projects else []
    has_three_projects = len(top_projects) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_projects),
        id="github_switch_repositories",
        desc="[Action Node] github.com:F7:A3 - Navigate to the Repositories tab to view open-source projects",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F7:P11 - Identify public project status
    projects_are_public = all(
        'link' in p and looks_like_url(p.get('link'), 'github.com')
        for p in top_projects[:3]
    ) if has_three_projects else False
    evaluator.add_custom_node(
        result=bool(projects_are_public),
        id="github_identify_public_projects",
        desc="[Perception Node] github.com:F7:P11 - Identify publicly accessible projects",
        parent=github_node,
        critical=False
    )

    # [Action Node] github.com:F1:A7 - Sort by star count
    star_counts = [p.get('stars') for p in top_projects[:3]] if has_three_projects else []
    stars_descending = is_descending_order(star_counts) if star_counts else False
    evaluator.add_custom_node(
        result=bool(stars_descending),
        id="github_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - List top 3 projects sorted by star count in descending order",
        parent=github_node,
        critical=False
    )

    # [Perception Node] github.com:F7:P18 - Judge activity trends
    username_looks_active = bool(
        github_info.github_username and
        len(github_info.github_username.strip()) > 0 and
        github_profile_ok
    )
    evaluator.add_custom_node(
        result=bool(username_looks_active),
        id="github_judge_activity",
        desc="[Perception Node] github.com:F7:P18 - Assess author's activity level through profile and contribution patterns",
        parent=github_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
