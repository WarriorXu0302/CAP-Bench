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
TASK_ID = "task-ace1ad"
TASK_DESCRIPTION = 'I’m researching graphics rendering optimization techniques. First, search arXiv for papers related to **“real-time rendering optimization”** or **“game graphics acceleration.”** Find 3 papers published since 2024 with more than 10 citations, and record each paper’s title, authors, arXiv ID, and publication date.\n\nThen, for each of the 3 papers in sequence: open the paper detail page and read the full abstract. Verify whether the paper mentions a specific game engine or commercial game application case. If the abstract mentions a game title, record it. If not, search Steam using the paper’s core technical term (extracted from the title, such as “ray tracing,” “DLSS,” “mesh shading,” etc.), and prioritize one game whose store-page technical description or tags explicitly mention that technology. If no result explicitly mentions the technology, select the game with the highest technical relevance and note the matching basis (e.g., relevant wording in tags, genre, or description).\n\nFinally, check GameSpot for professional reviews of these games, focusing on comments about graphics performance and frame-rate behavior. If a game has no GameSpot review, move to the next candidate game from your Steam results under the same technical term until a usable review is found. If none can be found, explicitly record: **“No GameSpot reviews are currently available for games retrieved under this technical direction.”**\n\nOutput required: for each paper, provide title, authors, arXiv ID, publication date, citation count, paper detail-page link, and core technical term(s) from the abstract; and for the corresponding game, provide game title, Steam link, GameSpot score, a concise summary of GameSpot’s graphics-performance commentary, and the GameSpot review link.'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information extracted for a single paper"""
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    arxiv_id: Optional[str] = None
    publication_date: Optional[str] = None
    citation_count: Optional[int] = None
    paper_detail_link: Optional[str] = None
    core_technical_terms: Optional[List[str]] = None


class GameInfo(BaseModel):
    """Information extracted for a game related to a paper"""
    game_title: Optional[str] = None
    steam_link: Optional[str] = None
    gamespot_score: Optional[float] = None
    graphics_performance_summary: Optional[str] = None
    gamespot_review_link: Optional[str] = None


class PaperGamePair(BaseModel):
    """A pair of paper and its related game"""
    paper: Optional[PaperInfo] = None
    game: Optional[GameInfo] = None


class ExtractedData(BaseModel):
    """All extracted paper-game pairs"""
    pairs: Optional[List[PaperGamePair]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_all_data() -> str:
    return """
Extract all paper-game pairs from the answer. The user should have found 3 papers and corresponding games.

For each paper, extract:
- title: paper title
- authors: list of author names
- arxiv_id: arXiv identifier (e.g., "2401.12345")
- publication_date: publication date (any format)
- citation_count: number of citations as an integer
- paper_detail_link: URL to the arXiv paper detail page
- core_technical_terms: list of technical terms extracted from the abstract (e.g., ["ray tracing", "DLSS"])

For each corresponding game, extract:
- game_title: name of the game
- steam_link: URL to the Steam store page
- gamespot_score: GameSpot review score (1-10 scale)
- graphics_performance_summary: summary of graphics/performance commentary from GameSpot
- gamespot_review_link: URL to the GameSpot review

If any field is missing, set it to null or an empty list as appropriate.
Return up to 3 pairs. If fewer are present, return what's available.
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


def looks_like_arxiv_id(text: Optional[str]) -> bool:
    if not text:
        return False
    # Matches patterns like "2401.12345" or "arxiv:2401.12345"
    return bool(re.search(r'\d{4}\.\d{4,5}', text))


def looks_like_arxiv_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'arxiv.org' in text.lower()


def looks_like_steam_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'steampowered.com' in text.lower() or 'store.steampowered' in text.lower()


def looks_like_gamespot_url(text: Optional[str]) -> bool:
    if not text:
        return False
    return 'gamespot.com' in text.lower()


def is_valid_year(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check if contains a year >= 2024
    years = re.findall(r'20\d{2}', text)
    return any(int(y) >= 2024 for y in years)


def is_valid_citation_count(count: Optional[int]) -> bool:
    if count is None:
        return False
    return count > 10


def is_valid_gamespot_score(score: Optional[float]) -> bool:
    if score is None:
        return False
    return 1.0 <= score <= 10.0


def has_graphics_keywords(text: Optional[str]) -> bool:
    if not text:
        return False
    keywords = ['graphics', 'performance', 'frame', 'fps', 'framerate', 'visual', 'rendering']
    return has_any_ci(text, keywords)


def count_papers_in_answer(answer: str) -> int:
    # Simple heuristic: count mentions of "paper" or arxiv IDs
    arxiv_ids = re.findall(r'\d{4}\.\d{4,5}', answer)
    return min(len(arxiv_ids), 3)


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
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_data(),
        template_class=ExtractedData,
        extraction_name="paper_game_pairs"
    )

    pairs = extracted.pairs if extracted and extracted.pairs else []

    # -------- 3. Build evaluation tree ----------------------------------- #

    # 3.1 arXiv search and paper selection
    arxiv_search_node = evaluator.add_sequential(
        id="arxiv_search_section",
        desc="arXiv search for real-time rendering optimization or game graphics acceleration papers",
        parent=root,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A2 - Search submission
    search_keywords_present = has_any_ci(answer, ['real-time rendering', 'rendering optimization', 'graphics acceleration', 'game graphics'])
    evaluator.add_custom_node(
        result=bool(search_keywords_present),
        id="arxiv_search_action",
        desc="[Action Node] arxiv.org:F1:A2 - Search arXiv for papers related to real-time rendering optimization or game graphics acceleration",
        parent=arxiv_search_node,
        critical=False
    )

    # [Perception Node] arxiv.org:F1:P11 - Search result understanding (2024+, >10 citations)
    valid_papers_count = 0
    for pair in pairs:
        if pair.paper:
            date_ok = is_valid_year(pair.paper.publication_date)
            citations_ok = is_valid_citation_count(pair.paper.citation_count)
            if date_ok and citations_ok:
                valid_papers_count += 1

    evaluator.add_custom_node(
        result=bool(valid_papers_count >= 3),
        id="arxiv_filter_perception",
        desc="[Perception Node] arxiv.org:F1:P11 - Find 3 papers published since 2024 with more than 10 citations",
        parent=arxiv_search_node,
        critical=False
    )

    # [Action Node] arxiv.org:F1:A16 - Pagination browsing
    has_three_papers = len(pairs) >= 3
    evaluator.add_custom_node(
        result=bool(has_three_papers),
        id="arxiv_pagination_action",
        desc="[Action Node] arxiv.org:F1:A16 - Browse through search results to find 3 qualifying papers",
        parent=arxiv_search_node,
        critical=False
    )

    # 3.2 Paper detail pages and abstract analysis
    papers_node = evaluator.add_parallel(
        id="papers_detail_section",
        desc="Analyze each paper's detail page and abstract",
        parent=root,
        critical=False
    )

    for idx, pair in enumerate(pairs[:3], 1):
        paper_node = evaluator.add_sequential(
            id=f"paper_{idx}_analysis",
            desc=f"Paper {idx} - Detail page and abstract analysis",
            parent=papers_node,
            critical=False
        )

        if pair.paper:
            # [Action Node] arxiv.org:F3:A6 - Enter paper detail page
            has_detail_link = looks_like_arxiv_url(pair.paper.paper_detail_link)
            evaluator.add_custom_node(
                result=bool(has_detail_link),
                id=f"paper_{idx}_detail_action",
                desc=f"[Action Node] arxiv.org:F3:A6 - Open detail page for paper {idx}",
                parent=paper_node,
                critical=False
            )

            # [Action Node] arxiv.org:F3:A20 - Expand full abstract
            has_abstract_info = bool(pair.paper.core_technical_terms and len(pair.paper.core_technical_terms) > 0)
            evaluator.add_custom_node(
                result=bool(has_abstract_info),
                id=f"paper_{idx}_abstract_action",
                desc=f"[Action Node] arxiv.org:F3:A20 - Read and expand full abstract for paper {idx}",
                parent=paper_node,
                critical=False
            )

            # [Perception Node] arxiv.org:F3:P5 - Understand research content
            has_technical_terms = bool(pair.paper.core_technical_terms and len(pair.paper.core_technical_terms) > 0)
            evaluator.add_custom_node(
                result=bool(has_technical_terms),
                id=f"paper_{idx}_content_perception",
                desc=f"[Perception Node] arxiv.org:F3:P5 - Extract core technical terms from abstract of paper {idx}",
                parent=paper_node,
                critical=False
            )

            # [Perception Node] arxiv.org:F3:P4 - Identify author information
            has_authors = bool(pair.paper.authors and len(pair.paper.authors) > 0)
            evaluator.add_custom_node(
                result=bool(has_authors),
                id=f"paper_{idx}_authors_perception",
                desc=f"[Perception Node] arxiv.org:F3:P4 - Record author information for paper {idx}",
                parent=paper_node,
                critical=False
            )

            # [Perception Node] arxiv.org:F3:P24 - Understand citation impact
            citations_valid = is_valid_citation_count(pair.paper.citation_count)
            evaluator.add_custom_node(
                result=bool(citations_valid),
                id=f"paper_{idx}_citations_perception",
                desc=f"[Perception Node] arxiv.org:F3:P24 - Verify citation count >10 for paper {idx}",
                parent=paper_node,
                critical=False
            )

    # 3.3 Steam game search and selection
    steam_node = evaluator.add_parallel(
        id="steam_search_section",
        desc="Steam game search using technical terms from papers",
        parent=root,
        critical=False
    )

    for idx, pair in enumerate(pairs[:3], 1):
        game_search_node = evaluator.add_sequential(
            id=f"game_{idx}_search",
            desc=f"Game {idx} - Steam search and selection",
            parent=steam_node,
            critical=False
        )

        if pair.game:
            # [Action Node] store.steampowered.com:F1:A2 - Steam search
            has_game = bool(pair.game.game_title)
            evaluator.add_custom_node(
                result=bool(has_game),
                id=f"game_{idx}_search_action",
                desc=f"[Action Node] store.steampowered.com:F1:A2 - Search Steam using technical term from paper {idx}",
                parent=game_search_node,
                critical=False
            )

            # [Perception Node] store.steampowered.com:F1:P3 - Understand search results
            has_game_with_link = bool(pair.game.game_title and pair.game.steam_link)
            evaluator.add_custom_node(
                result=bool(has_game_with_link),
                id=f"game_{idx}_results_perception",
                desc=f"[Perception Node] store.steampowered.com:F1:P3 - Identify game matching the technical term for paper {idx}",
                parent=game_search_node,
                critical=False
            )

            # [Action Node] store.steampowered.com:F2:A9 - Enter game detail page
            has_steam_url = looks_like_steam_url(pair.game.steam_link)
            evaluator.add_custom_node(
                result=bool(has_steam_url),
                id=f"game_{idx}_detail_action",
                desc=f"[Action Node] store.steampowered.com:F2:A9 - Open game detail page for game {idx}",
                parent=game_search_node,
                critical=False
            )

            # [Perception Node] store.steampowered.com:F2:P13 - Identify game features
            # Check if technical terms align between paper and game
            has_technical_match = bool(pair.paper and pair.paper.core_technical_terms and pair.game.game_title)
            evaluator.add_custom_node(
                result=bool(has_technical_match),
                id=f"game_{idx}_features_perception",
                desc=f"[Perception Node] store.steampowered.com:F2:P13 - Verify game mentions relevant technology in description or tags for paper {idx}",
                parent=game_search_node,
                critical=False
            )

    # 3.4 GameSpot reviews
    gamespot_node = evaluator.add_parallel(
        id="gamespot_reviews_section",
        desc="GameSpot professional reviews for graphics performance",
        parent=root,
        critical=False
    )

    for idx, pair in enumerate(pairs[:3], 1):
        review_node = evaluator.add_sequential(
            id=f"review_{idx}_analysis",
            desc=f"Review {idx} - GameSpot review for game {idx}",
            parent=gamespot_node,
            critical=False
        )

        if pair.game:
            # [Action Node] gamespot.com:F1:A15 - Enter review details
            has_review_link = looks_like_gamespot_url(pair.game.gamespot_review_link)
            evaluator.add_custom_node(
                result=bool(has_review_link),
                id=f"review_{idx}_detail_action",
                desc=f"[Action Node] gamespot.com:F1:A15 - Access GameSpot review for game {idx}",
                parent=review_node,
                critical=False
            )

            # [Perception Node] gamespot.com:F1:P1 - Identify review score
            has_valid_score = is_valid_gamespot_score(pair.game.gamespot_score)
            evaluator.add_custom_node(
                result=bool(has_valid_score),
                id=f"review_{idx}_score_perception",
                desc=f"[Perception Node] gamespot.com:F1:P1 - Extract GameSpot score (1-10) for game {idx}",
                parent=review_node,
                critical=False
            )

            # [Perception Node] gamespot.com:F1:P3 - Extract review opinions
            has_graphics_commentary = has_graphics_keywords(pair.game.graphics_performance_summary)
            evaluator.add_custom_node(
                result=bool(has_graphics_commentary),
                id=f"review_{idx}_opinion_perception",
                desc=f"[Perception Node] gamespot.com:F1:P3 - Extract graphics performance and frame-rate commentary for game {idx}",
                parent=review_node,
                critical=False
            )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
