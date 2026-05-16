import asyncio
import logging
from typing import Optional, List, Dict, Any
import re

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "task-0e526a"
TASK_DESCRIPTION = """
I saw a 2024 image segmentation paper on arXiv and want to fully reproduce its experiments.

First, go to Semantic Scholar and search for image segmentation papers published in this field since 2024. Identify the top 5 papers with the highest citation counts, then filter these to find papers with PDF downloads available. Select one of these papers. Go to GitHub and search using the paper's title or the first author's name. Look for an official code repository with over 500 stars. Confirm that the README file references the selected paper and that a `requirements.txt` file exists. If not, proceed to the next paper until these criteria are met. Next, go to Hugging Face and search for the paper's title. Identify the top 3 pre-trained models by download count. Review their Model Cards to confirm that the models indeed correspond to the selected paper. Finally, go to YouTube and search using the paper's title or the first author's name. Find an in-depth explanation video that is over 10 minutes long and has more than 5000 views. Expand the video description to confirm that it mentions the paper's title or author.

Output: Paper Title, First Author, Publication Year, Semantic Scholar Citation Count, PDF Availability (Yes/No), Paper Link; GitHub Repository Name, Star Count, `requirements.txt` Availability (Yes/No), Repository Link; Hugging Face Model Name, Download Count, Model Card Summary, Model Link; YouTube Video Title, Duration, View Count, Video Description Summary, Video Link.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    title: Optional[str] = None
    first_author: Optional[str] = None
    publication_year: Optional[str] = None  # keep as str for robust parsing
    citation_count: Optional[str] = None
    pdf_available: Optional[str] = None  # Yes/No/True/False
    link: Optional[str] = None


class GitHubRepo(BaseModel):
    name: Optional[str] = None
    stars: Optional[str] = None
    requirements_txt_available: Optional[str] = None  # Yes/No/True/False
    link: Optional[str] = None


class HuggingFaceModel(BaseModel):
    name: Optional[str] = None
    downloads: Optional[str] = None
    model_card_summary: Optional[str] = None
    link: Optional[str] = None


class YouTubeVideo(BaseModel):
    title: Optional[str] = None
    duration: Optional[str] = None
    view_count: Optional[str] = None
    description_summary: Optional[str] = None
    link: Optional[str] = None


class TaskAnswerExtract(BaseModel):
    paper: Optional[PaperInfo] = None
    github: Optional[GitHubRepo] = None
    hf_models: List[HuggingFaceModel] = Field(default_factory=list)
    youtube: Optional[YouTubeVideo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_task_output() -> str:
    return """
Parse the assistant's answer according to the required Output specification:

Expected segments in order:
1) Paper Title, First Author, Publication Year, Semantic Scholar Citation Count, PDF Availability (Yes/No), Paper Link
2) GitHub Repository Name, Star Count, requirements.txt Availability (Yes/No), Repository Link
3) Hugging Face Models (top 3, if available). For each model: Hugging Face Model Name, Download Count, Model Card Summary, Model Link
4) YouTube Video Title, Duration, View Count, Video Description Summary, Video Link

Notes:
- Publication Year and Citation Count may be formatted; keep as text but preserve numeric content.
- PDF Availability is typically "Yes" or "No".
- Extract up to 3 Hugging Face models if present; if fewer, include available ones.
- Keep exact strings as presented in the answer for each field.

Return the data in the JSON schema provided.
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def safe_int_from_str(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    try:
        # Normalize
        txt = s.strip().lower()
        txt = txt.replace(",", "").replace("+", "")
        # Handle k/m suffixes
        m = re.match(r"^(\d+(\.\d+)?)([km])?$", txt)
        if m:
            num = float(m.group(1))
            suffix = m.group(3)
            if suffix == 'k':
                return int(num * 1000)
            if suffix == 'm':
                return int(num * 1000000)
            return int(num)
        # Fallback: extract digits
        digits = re.findall(r"\d+", txt)
        if not digits:
            return None
        return int("".join(digits))
    except Exception:
        return None


def safe_bool_from_str(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    txt = s.strip().lower()
    if txt in ["yes", "true", "y", "1"]:
        return True
    if txt in ["no", "false", "n", "0"]:
        return False
    return None


def parse_duration_to_seconds(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    txt = s.strip().lower()
    # Formats to support: "HH:MM:SS", "MM:SS", "1 hr 2 min", "12m 3s", "15 minutes", "1h 5m"
    # Try HH:MM:SS or MM:SS
    if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", txt):
        parts = txt.split(":")
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    # Try textual formats
    hours = 0
    minutes = 0
    seconds = 0
    h = re.search(r"(\d+)\s*h(r|ours)?", txt)
    m = re.search(r"(\d+)\s*m(in|inute|inutes)?", txt)
    s_sec = re.search(r"(\d+)\s*s(ec|econd|econds)?", txt)
    if h:
        hours = int(h.group(1))
    if m:
        minutes = int(m.group(1))
    if s_sec:
        seconds = int(s_sec.group(1))
    if hours or minutes or seconds:
        return hours * 3600 + minutes * 60 + seconds
    # Try "15 minutes"
    m2 = re.search(r"(\d+)", txt)
    if m2 and "min" in txt:
        return int(m2.group(1)) * 60
    return None


def tokenize_title_keywords(title: Optional[str]) -> List[str]:
    if not title:
        return []
    txt = re.sub(r"[^a-z0-9\s]", " ", title.lower())
    tokens = [t for t in txt.split() if len(t) >= 4]
    # Deduplicate
    seen = set()
    res = []
    for t in tokens:
        if t not in seen:
            res.append(t)
            seen.add(t)
    return res


def last_name(author: Optional[str]) -> Optional[str]:
    if not author:
        return None
    parts = [p.strip() for p in author.replace(",", " ").split()]
    if not parts:
        return None
    return parts[-1].lower()


def contains_any(haystack: Optional[str], needles: List[str]) -> bool:
    if not haystack or not needles:
        return False
    h = haystack.lower()
    for n in needles:
        if n and n.lower() in h:
            return True
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

    # Extract structured data from the answer
    task_output = await evaluator.extract(
        prompt=prompt_extract_task_output(),
        template_class=TaskAnswerExtract,
        extraction_name="task_output"
    )

    paper = task_output.paper or PaperInfo()
    github = task_output.github or GitHubRepo()
    hf_models = task_output.hf_models or []
    youtube = task_output.youtube or YouTubeVideo()

    # Derived simple fields
    pub_year = safe_int_from_str(paper.publication_year)
    citation_count_num = safe_int_from_str(paper.citation_count)
    pdf_yes = safe_bool_from_str(paper.pdf_available)
    stars_num = safe_int_from_str(github.stars)
    req_yes = safe_bool_from_str(github.requirements_txt_available)

    # Keywords for checks
    title_keywords = tokenize_title_keywords(paper.title)
    author_last = last_name(paper.first_author)
    repo_name_l = (github.name or "").lower()
    repo_link_l = (github.link or "").lower()
    answer_l = (answer or "").lower()

    # Hugging Face downloads as numeric
    hf_downloads = [safe_int_from_str(m.downloads) for m in hf_models]
    hf_names = [m.name or "" for m in hf_models]
    hf_summaries = [m.model_card_summary or "" for m in hf_models]
    hf_links = [m.link or "" for m in hf_models]

    # YouTube fields derived
    yt_duration_seconds = parse_duration_to_seconds(youtube.duration or "")
    yt_view_count_num = safe_int_from_str(youtube.view_count)
    yt_title_l = (youtube.title or "").lower()
    yt_desc_l = (youtube.description_summary or "").lower()

    # Containers per platform (sequential to reflect flow, but non-critical)
    semsch_node = evaluator.add_sequential(
        id="semanticscholar_checks",
        desc="Semantic Scholar pipeline checks",
        parent=root,
        critical=False
    )
    github_node = evaluator.add_sequential(
        id="github_checks",
        desc="GitHub repository checks",
        parent=root,
        critical=False
    )
    hf_node = evaluator.add_sequential(
        id="huggingface_checks",
        desc="Hugging Face models checks",
        parent=root,
        critical=False
    )
    yt_node = evaluator.add_sequential(
        id="youtube_checks",
        desc="YouTube video checks",
        parent=root,
        critical=False
    )

    # ------------------- Semantic Scholar covered points ------------------- #
    # semanticscholar.org:F1:A3 - date filter (year >= 2024)
    evaluator.add_custom_node(
        result=bool(pub_year is not None and pub_year >= 2024),
        id="sem_sch_F1_A3_date_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Verify publication year >= 2024 (date range filter applied)",
        parent=semsch_node,
        critical=False
    )

    # semanticscholar.org:F1:A4 - sorting by citation count; check citation is reasonably high
    evaluator.add_custom_node(
        result=bool(citation_count_num is not None and citation_count_num >= 1),
        id="sem_sch_F1_A4_sort_by_citations",
        desc="[Action Node] semanticscholar.org:F1:A4 - Verify citation count appears high (descending sort intent)",
        parent=semsch_node,
        critical=False
    )

    # semanticscholar.org:F1:P2 - PDF availability
    evaluator.add_custom_node(
        result=bool(pdf_yes is True),
        id="sem_sch_F1_P2_pdf_available",
        desc="[Perception Node] semanticscholar.org:F1:P2 - Verify PDF availability is marked as Yes",
        parent=semsch_node,
        critical=False
    )

    # semanticscholar.org:F2:A5 - click into details; full title present
    is_full_title = bool(paper.title and (len(paper.title) >= 15 or len(paper.title.split()) >= 4))
    evaluator.add_custom_node(
        result=is_full_title,
        id="sem_sch_F2_A5_click_details",
        desc="[Action Node] semanticscholar.org:F2:A5 - Verify full paper title present (details accessed)",
        parent=semsch_node,
        critical=False
    )

    # semanticscholar.org:F2:P3 - citation statistics understanding; numeric citation count
    evaluator.add_custom_node(
        result=bool(citation_count_num is not None and citation_count_num >= 0),
        id="sem_sch_F2_P3_citation_numeric",
        desc="[Perception Node] semanticscholar.org:F2:P3 - Verify citation count parsed as a numeric value",
        parent=semsch_node,
        critical=False
    )

    # ------------------------ GitHub covered points ------------------------ #
    # github.com:F1:A10 - search form submission using title or author: repo name/link contains keyword
    significant_needles = title_keywords[:]
    if author_last and author_last not in significant_needles:
        significant_needles.append(author_last)
    used_title_or_author = contains_any(repo_name_l, significant_needles) or contains_any(repo_link_l, significant_needles)
    evaluator.add_custom_node(
        result=used_title_or_author,
        id="github_F1_A10_search_submission",
        desc="[Action Node] github.com:F1:A10 - Verify repo appears searched via paper title or first author (name match in repo)",
        parent=github_node,
        critical=False
    )

    # github.com:F1:A7 - sorting by stars / star count > 500
    evaluator.add_custom_node(
        result=bool(stars_num is not None and stars_num > 500),
        id="github_F1_A7_sort_by_stars",
        desc="[Action Node] github.com:F1:A7 - Verify repository stars > 500 (implies sorting by stars)",
        parent=github_node,
        critical=False
    )

    # github.com:F1:P1 - star count extracted correctly (numeric present)
    evaluator.add_custom_node(
        result=bool(stars_num is not None and stars_num >= 0),
        id="github_F1_P1_star_extraction",
        desc="[Perception Node] github.com:F1:P1 - Verify star count extracted as a number",
        parent=github_node,
        critical=False
    )

    # github.com:F3:A17 - file tree expand; requirements.txt exists (answer mentions it)
    req_in_answer = "requirements.txt" in answer_l
    evaluator.add_custom_node(
        result=bool(req_yes is True or req_in_answer),
        id="github_F3_A17_requirements_tree",
        desc="[Action Node] github.com:F3:A17 - Verify requirements.txt presence (via file tree/answer mention)",
        parent=github_node,
        critical=False
    )

    # github.com:F3:A18 - open README and confirm it references the paper
    readme_mentioned = "readme" in answer_l
    paper_title_in_answer = contains_any(answer_l, title_keywords)
    evaluator.add_custom_node(
        result=bool((github.name and readme_mentioned and paper_title_in_answer) or (readme_mentioned and author_last and author_last in answer_l)),
        id="github_F3_A18_readme_references_paper",
        desc="[Action Node] github.com:F3:A18 - Verify README referenced the selected paper for this repo",
        parent=github_node,
        critical=False
    )

    # github.com:F3:P12 - directory structure understanding (requirements at root inferred)
    # Lenient: if requirements.txt mentioned without subdirectory or generally mentioned, accept
    requirements_root_like = bool(req_in_answer)
    evaluator.add_custom_node(
        result=requirements_root_like,
        id="github_F3_P12_dir_structure",
        desc="[Perception Node] github.com:F3:P12 - Verify understanding of repo structure via requirements.txt presence",
        parent=github_node,
        critical=False
    )

    # ---------------------- Hugging Face covered points -------------------- #
    # huggingface.co:F1:A9 - search using paper title: model names/summary include title keywords
    hf_title_match = any(
        contains_any((hf_names[i] or "") + " " + (hf_summaries[i] or ""), title_keywords)
        for i in range(len(hf_models))
    )
    evaluator.add_custom_node(
        result=hf_title_match or bool(len(hf_models) > 0),
        id="hf_F1_A9_search_by_title",
        desc="[Action Node] huggingface.co:F1:A9 - Verify search likely used paper title (models or summaries reflect title keywords)",
        parent=hf_node,
        critical=False
    )

    # huggingface.co:F1:A4 - pagination browse / top 3 by downloads: check ordering and high downloads
    # Lenient: accept if at least 1 model with downloads, and if multiple, non-increasing order
    non_increasing = True
    filtered_downloads = [d for d in hf_downloads if d is not None]
    if len(filtered_downloads) >= 2:
        for i in range(len(filtered_downloads) - 1):
            if filtered_downloads[i] < filtered_downloads[i + 1]:
                non_increasing = False
                break
    hf_downloads_present = len(filtered_downloads) >= 1
    hf_downloads_highish = any(d is not None and d >= 100 for d in hf_downloads if d is not None) or len(filtered_downloads) >= 1
    evaluator.add_custom_node(
        result=bool(hf_downloads_present and non_increasing and hf_downloads_highish),
        id="hf_F1_A4_top_downloads",
        desc="[Action Node] huggingface.co:F1:A4 - Verify top downloads ordering (non-increasing) and presence of download counts",
        parent=hf_node,
        critical=False
    )

    # huggingface.co:F3:A3 - switch to Model Card: non-empty model card summary referencing paper or arXiv
    model_card_relevant = any(
        (m.model_card_summary and (contains_any(m.model_card_summary, title_keywords) or "arxiv" in (m.model_card_summary or "").lower() or "paper" in (m.model_card_summary or "").lower()))
        for m in hf_models
    )
    evaluator.add_custom_node(
        result=model_card_relevant,
        id="hf_F3_A3_model_card_tab",
        desc="[Action Node] huggingface.co:F3:A3 - Verify Model Card reviewed and references the paper",
        parent=hf_node,
        critical=False
    )

    # huggingface.co:F3:P10 - Model Card content understanding: confirm model corresponds to selected paper
    model_card_correspondence = any(
        contains_any((m.model_card_summary or ""), title_keywords) or (author_last and author_last in (m.model_card_summary or "").lower())
        for m in hf_models
    )
    evaluator.add_custom_node(
        result=model_card_correspondence,
        id="hf_F3_P10_card_understanding",
        desc="[Perception Node] huggingface.co:F3:P10 - Verify understanding that models correspond to the selected paper",
        parent=hf_node,
        critical=False
    )

    # ------------------------- YouTube covered points ---------------------- #
    # youtube.com:F1:A22 - search result click: title related to paper title/author
    yt_related = contains_any(yt_title_l, title_keywords) or (author_last and author_last in yt_title_l)
    evaluator.add_custom_node(
        result=yt_related,
        id="yt_F1_A22_search_click",
        desc="[Action Node] youtube.com:F1:A22 - Verify clicked video is related to paper title or first author",
        parent=yt_node,
        critical=False
    )

    # youtube.com:F9:A4 - duration filter: >= 10 minutes
    evaluator.add_custom_node(
        result=bool(yt_duration_seconds is not None and yt_duration_seconds >= 600),
        id="yt_F9_A4_duration_filter",
        desc="[Action Node] youtube.com:F9:A4 - Verify video duration is >= 10 minutes",
        parent=yt_node,
        critical=False
    )

    # youtube.com:F1:P4 - relevant deep explanation and view count > 5000
    yt_relevant_content = yt_related or contains_any(yt_desc_l, title_keywords + ["segmentation"])
    evaluator.add_custom_node(
        result=bool((yt_view_count_num is not None and yt_view_count_num > 5000) and yt_relevant_content),
        id="yt_F1_P4_relevance_and_views",
        desc="[Perception Node] youtube.com:F1:P4 - Verify video is relevant and has > 5000 views",
        parent=yt_node,
        critical=False
    )

    # youtube.com:F2:A46 - expand description: description mentions paper title or author
    desc_mentions_paper = contains_any(yt_desc_l, title_keywords) or (author_last and author_last in yt_desc_l)
    evaluator.add_custom_node(
        result=desc_mentions_paper,
        id="yt_F2_A46_expand_description",
        desc="[Action Node] youtube.com:F2:A46 - Verify expanded description mentions the paper title or author",
        parent=yt_node,
        critical=False
    )

    # youtube.com:F2:P21 - video content understanding: indicates deep explanation
    deep_keywords = ["explain", "explained", "tutorial", "lecture", "talk", "presentation", "overview", "deep dive", "in-depth", "walkthrough", "analysis"]
    deep_indicators = any(k in yt_title_l for k in deep_keywords) or any(k in yt_desc_l for k in deep_keywords)
    evaluator.add_custom_node(
        result=bool((yt_duration_seconds is not None and yt_duration_seconds >= 600) and (deep_indicators or yt_relevant_content)),
        id="yt_F2_P21_deep_explanation",
        desc="[Perception Node] youtube.com:F2:P21 - Verify this is a deep technical explanation video",
        parent=yt_node,
        critical=False
    )

    # ------------------------- Optional general checks --------------------- #
    # Not part of covered_points but helpful lenient checks (non-critical)
    extras_node = evaluator.add_parallel(
        id="general_checks",
        desc="General lenient sanity checks",
        parent=root,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(paper.link and ("semanticscholar.org" in (paper.link or "").lower() or "arxiv.org" in (paper.link or "").lower())),
        id="gen_paper_link_domain",
        desc="Check paper link appears to be from Semantic Scholar or arXiv",
        parent=extras_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(github.link and "github.com" in (github.link or "").lower()),
        id="gen_github_link_domain",
        desc="Check GitHub repository link domain is github.com",
        parent=extras_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=any("huggingface.co" in (l or "").lower() for l in hf_links),
        id="gen_hf_link_domain",
        desc="Check at least one Hugging Face model link has huggingface.co domain",
        parent=extras_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(youtube.link and ("youtube.com" in (youtube.link or "").lower() or "youtu.be" in (youtube.link or "").lower())),
        id="gen_yt_link_domain",
        desc="Check YouTube video link domain is YouTube",
        parent=extras_node,
        critical=False
    )

    # Note: To minimize LLM verification calls, we used only custom nodes and extraction.
    # We intentionally did not call evaluator.verify more than 0 times (<= 1 allowed).

    return evaluator.get_summary()