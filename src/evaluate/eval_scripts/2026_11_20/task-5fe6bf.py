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
TASK_ID = "task-5fe6bf"
TASK_DESCRIPTION = 'I am a product manager at an AI startup, and my team is evaluating the feasibility of commercializing **Monocular Depth Estimation** technology as an API service. Please help me conduct a technical commercialization research task:\n\nFirst, go to Semantic Scholar and search for **"monocular depth estimation"**. Sort by citation count and identify the 3 most-cited papers published since 2024. Record each paper’s title, authors, publication year, citation count, and TLDR summary.\n\nThen open each paper’s detail page and check whether it includes a GitHub link. If it does, visit the corresponding GitHub repository and review the README to extract the GPU model and VRAM requirement needed for model inference (e.g., RTX 3090 / 24GB). If the README does not explicitly state GPU requirements, check the repository’s requirements, docs, or issues for minimum hardware information. If still not found, mark it as **“GPU requirements not explicitly specified.”**\n\nNext, search Amazon for those GPU models, filter for **new** condition, and prioritize products sold by **Amazon**. Find the currently lowest-priced option and record the GPU model, price, seller type, and product link. If no Amazon-sold listing is available under new condition for that model, select the lowest-priced non-Amazon seller listing under new condition as a fallback, and clearly label it as a fallback choice.\n\nFinally, go to Google Scholar and search for **"depth estimation API pricing"** or **"3D reconstruction API cost"** to find pricing clues for existing competitor services (e.g., Replicate, Hugging Face Inference API). Then visit each service’s public pricing page to verify pricing details. If found, record the provider name, pricing model (per call / monthly subscription), and exact price.\n\n**Output required:**\n- For each paper: title, authors, publication year, citation count, TLDR summary, whether a GitHub link exists, and Semantic Scholar paper link  \n- Extracted GPU requirements from the corresponding GitHub repo (model and VRAM; explicitly note if unspecified), plus GitHub link  \n- Lowest Amazon price for that GPU, seller type, and product link (and whether it is Amazon-sold)  \n- Competitor service name, pricing model, and exact price (if not found, explain why and list attempted paths)'


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PaperInfo(BaseModel):
    """Information about a single paper from Semantic Scholar"""
    title: Optional[str] = None
    authors: Optional[str] = None
    publication_year: Optional[int] = None
    citation_count: Optional[int] = None
    tldr_summary: Optional[str] = None
    has_github_link: Optional[bool] = None
    semantic_scholar_link: Optional[str] = None


class PapersCollection(BaseModel):
    """Collection of papers extracted from the answer"""
    papers: List[PaperInfo] = Field(default_factory=list)


class GPURequirement(BaseModel):
    """GPU requirements extracted from GitHub README"""
    gpu_model: Optional[str] = None
    vram: Optional[str] = None
    github_link: Optional[str] = None
    explicitly_specified: Optional[bool] = None


class GPURequirementsCollection(BaseModel):
    """Collection of GPU requirements for different papers"""
    requirements: List[GPURequirement] = Field(default_factory=list)


class AmazonProduct(BaseModel):
    """Amazon product information"""
    gpu_model: Optional[str] = None
    price: Optional[str] = None
    seller_type: Optional[str] = None
    is_amazon_sold: Optional[bool] = None
    product_link: Optional[str] = None
    is_fallback: Optional[bool] = None


class AmazonProductsCollection(BaseModel):
    """Collection of Amazon products"""
    products: List[AmazonProduct] = Field(default_factory=list)


class CompetitorPricing(BaseModel):
    """Competitor service pricing information"""
    provider_name: Optional[str] = None
    pricing_model: Optional[str] = None
    exact_price: Optional[str] = None
    verification_attempted: Optional[bool] = None
    not_found_explanation: Optional[str] = None


class CompetitorPricingCollection(BaseModel):
    """Collection of competitor pricing information"""
    competitors: List[CompetitorPricing] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_papers() -> str:
    return """
Extract information about the 3 most-cited papers on monocular depth estimation published since 2024 from the answer.

For each paper, extract:
- title: the paper title
- authors: author names
- publication_year: year of publication (as integer)
- citation_count: number of citations (as integer)
- tldr_summary: the TLDR summary text
- has_github_link: whether the paper has a GitHub link (boolean)
- semantic_scholar_link: the Semantic Scholar paper detail page URL

Return a list of papers. If any field is missing, set it to null.
"""


def prompt_extract_gpu_requirements() -> str:
    return """
Extract GPU requirements information from GitHub repositories mentioned in the answer.

For each repository, extract:
- gpu_model: the GPU model name (e.g., RTX 3090, A100)
- vram: the VRAM requirement (e.g., 24GB, 16GB)
- github_link: the GitHub repository URL
- explicitly_specified: whether GPU requirements were explicitly stated in README/docs (boolean)

If GPU requirements were not found, the answer should mention "GPU requirements not explicitly specified" or similar.

Return a list of GPU requirements. If any field is missing, set it to null.
"""


def prompt_extract_amazon_products() -> str:
    return """
Extract Amazon product information for GPUs from the answer.

For each GPU product, extract:
- gpu_model: the GPU model being searched
- price: the price (include currency symbol if present)
- seller_type: whether sold by Amazon or third-party seller
- is_amazon_sold: whether the product is sold by Amazon (boolean)
- product_link: the Amazon product page URL
- is_fallback: whether this is a fallback choice (non-Amazon seller) (boolean)

Return a list of products. If any field is missing, set it to null.
"""


def prompt_extract_competitor_pricing() -> str:
    return """
Extract competitor API service pricing information from the answer.

For each competitor service found, extract:
- provider_name: the service provider name (e.g., Replicate, Hugging Face)
- pricing_model: the pricing model (e.g., per call, monthly subscription)
- exact_price: the exact price mentioned
- verification_attempted: whether the answer attempted to verify pricing on the service's official page (boolean)
- not_found_explanation: if pricing was not found, extract the explanation of why and what was attempted

Return a list of competitors. If any field is missing, set it to null.
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


def contains_url(text: Optional[str], domain: str) -> bool:
    if not text:
        return False
    return domain.lower() in text.lower()


def is_valid_year_since_2024(year: Optional[int]) -> bool:
    if year is None:
        return False
    return year >= 2024


def is_positive_number(num: Optional[int]) -> bool:
    if num is None:
        return False
    return num > 0


def looks_like_gpu_model(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for common GPU model patterns
    gpu_patterns = [r'rtx\s*\d+', r'gtx\s*\d+', r'a\d{3,4}', r'v100', r'titan', r'tesla']
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in gpu_patterns)


def looks_like_vram(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for VRAM patterns like "24GB", "16 GB", etc.
    return bool(re.search(r'\d+\s*gb', text.lower()))


def looks_like_price(text: Optional[str]) -> bool:
    if not text:
        return False
    # Check for price patterns with currency symbols or numbers
    return bool(re.search(r'[\$€£¥]\s*\d+|^\d+[\.,]\d{2}', text))


def extract_paper_count(papers: List[PaperInfo]) -> int:
    return len([p for p in papers if p.title])


def check_papers_sorted_by_citations(papers: List[PaperInfo]) -> bool:
    citations = [p.citation_count for p in papers if p.citation_count is not None]
    if len(citations) < 2:
        return True
    for i in range(len(citations) - 1):
        if citations[i] < citations[i + 1]:
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
    papers_info = await evaluator.extract(
        prompt=prompt_extract_papers(),
        template_class=PapersCollection,
        extraction_name="papers_collection"
    )

    gpu_requirements = await evaluator.extract(
        prompt=prompt_extract_gpu_requirements(),
        template_class=GPURequirementsCollection,
        extraction_name="gpu_requirements"
    )

    amazon_products = await evaluator.extract(
        prompt=prompt_extract_amazon_products(),
        template_class=AmazonProductsCollection,
        extraction_name="amazon_products"
    )

    competitor_pricing = await evaluator.extract(
        prompt=prompt_extract_competitor_pricing(),
        template_class=CompetitorPricingCollection,
        extraction_name="competitor_pricing"
    )

    # -------- 3. Build evaluation tree ----------------------------------- #
    # 3.1 Semantic Scholar section
    semantic_scholar_node = evaluator.add_sequential(
        id="semantic_scholar_section",
        desc="Semantic Scholar paper search and extraction",
        parent=root,
        critical=False
    )

    # Check if 3 papers were found
    papers_count_ok = extract_paper_count(papers_info.papers) >= 3
    evaluator.add_custom_node(
        result=bool(papers_count_ok),
        id="papers_count_check",
        desc="Found at least 3 papers",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A3 - Date range filtering (2024+)
    years_valid = all(is_valid_year_since_2024(p.publication_year) for p in papers_info.papers if p.publication_year)
    evaluator.add_custom_node(
        result=bool(years_valid and len([p for p in papers_info.papers if p.publication_year]) > 0),
        id="date_range_filter",
        desc="[Action Node] semanticscholar.org:F1:A3 - Papers published since 2024",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A4 - Sort by citation count
    citations_sorted = check_papers_sorted_by_citations(papers_info.papers)
    citations_present = any(p.citation_count is not None for p in papers_info.papers)
    evaluator.add_custom_node(
        result=bool(citations_sorted and citations_present),
        id="citation_sort",
        desc="[Action Node] semanticscholar.org:F1:A4 - Papers sorted by citation count (descending)",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Action Node] semanticscholar.org:F1:A5 - Click into paper details
    has_semantic_links = any(p.semantic_scholar_link and contains_url(p.semantic_scholar_link, 'semanticscholar') for p in papers_info.papers)
    evaluator.add_custom_node(
        result=bool(has_semantic_links),
        id="paper_detail_navigation",
        desc="[Action Node] semanticscholar.org:F1:A5 - Navigate to paper detail pages",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P1 - TLDR summary extraction
    has_tldr = any(p.tldr_summary and len(p.tldr_summary.strip()) > 0 for p in papers_info.papers)
    evaluator.add_custom_node(
        result=bool(has_tldr),
        id="tldr_extraction",
        desc="[Perception Node] semanticscholar.org:F2:P1 - Extract TLDR summaries",
        parent=semantic_scholar_node,
        critical=False
    )

    # [Perception Node] semanticscholar.org:F2:P2 - GitHub link identification
    github_status_checked = any(p.has_github_link is not None for p in papers_info.papers)
    evaluator.add_custom_node(
        result=bool(github_status_checked),
        id="github_link_identification",
        desc="[Perception Node] semanticscholar.org:F2:P2 - Identify whether papers have GitHub links",
        parent=semantic_scholar_node,
        critical=False
    )

    # 3.2 GitHub section
    github_node = evaluator.add_sequential(
        id="github_section",
        desc="GitHub repository GPU requirements extraction",
        parent=root,
        critical=False
    )

    # Check if GitHub links were visited
    has_github_links = any(req.github_link and contains_url(req.github_link, 'github') for req in gpu_requirements.requirements)
    evaluator.add_custom_node(
        result=bool(has_github_links),
        id="github_links_present",
        desc="GitHub repository links extracted from papers",
        parent=github_node,
        critical=False
    )

    # Check GPU model extraction
    gpu_models_extracted = any(req.gpu_model and looks_like_gpu_model(req.gpu_model) for req in gpu_requirements.requirements)
    evaluator.add_custom_node(
        result=bool(gpu_models_extracted),
        id="gpu_model_extraction",
        desc="GPU models extracted from GitHub repositories",
        parent=github_node,
        critical=False
    )

    # Check VRAM extraction
    vram_extracted = any(req.vram and looks_like_vram(req.vram) for req in gpu_requirements.requirements)
    evaluator.add_custom_node(
        result=bool(vram_extracted),
        id="vram_extraction",
        desc="VRAM requirements extracted from GitHub repositories",
        parent=github_node,
        critical=False
    )

    # Check explicit specification acknowledgment
    explicit_spec_handled = any(req.explicitly_specified is not None for req in gpu_requirements.requirements)
    evaluator.add_custom_node(
        result=bool(explicit_spec_handled),
        id="explicit_spec_check",
        desc="Checked whether GPU requirements were explicitly specified",
        parent=github_node,
        critical=False
    )

    # 3.3 Amazon section
    amazon_node = evaluator.add_sequential(
        id="amazon_section",
        desc="Amazon GPU product search and pricing",
        parent=root,
        critical=False
    )

    # [Action Node] Amazon:F1:A5 - Category filtering (Electronics)
    has_amazon_products = len(amazon_products.products) > 0
    evaluator.add_custom_node(
        result=bool(has_amazon_products),
        id="amazon_category_search",
        desc="[Action Node] Amazon:F1:A5 - Search for GPU products on Amazon",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A14 - Brand/seller filtering
    seller_type_checked = any(p.seller_type or p.is_amazon_sold is not None for p in amazon_products.products)
    evaluator.add_custom_node(
        result=bool(seller_type_checked),
        id="amazon_seller_filter",
        desc="[Action Node] Amazon:F3:A14 - Filter by Amazon-sold products",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F3:A17 - Price filtering/sorting
    prices_extracted = any(p.price and looks_like_price(p.price) for p in amazon_products.products)
    evaluator.add_custom_node(
        result=bool(prices_extracted),
        id="amazon_price_sort",
        desc="[Action Node] Amazon:F3:A17 - Find lowest-priced GPU products",
        parent=amazon_node,
        critical=False
    )

    # [Perception Node] Amazon:F3:P1 - Product condition identification (new)
    new_condition_mentioned = has_any_ci(answer, ['new', 'new condition', 'brand new']) or any(p.seller_type and ci_contains(p.seller_type, 'new') for p in amazon_products.products)
    evaluator.add_custom_node(
        result=bool(new_condition_mentioned),
        id="amazon_condition_check",
        desc="[Perception Node] Amazon:F3:P1 - Filter for new condition products",
        parent=amazon_node,
        critical=False
    )

    # [Action Node] Amazon:F5:A20 - Navigate to product details
    has_product_links = any(p.product_link and contains_url(p.product_link, 'amazon') for p in amazon_products.products)
    evaluator.add_custom_node(
        result=bool(has_product_links),
        id="amazon_product_detail",
        desc="[Action Node] Amazon:F5:A20 - Navigate to product detail pages",
        parent=amazon_node,
        critical=False
    )

    # Check fallback handling
    fallback_handled = any(p.is_fallback is not None for p in amazon_products.products)
    evaluator.add_custom_node(
        result=bool(fallback_handled),
        id="amazon_fallback_check",
        desc="Fallback choices clearly labeled when Amazon-sold unavailable",
        parent=amazon_node,
        critical=False
    )

    # 3.4 Google Scholar and competitor pricing section
    competitor_node = evaluator.add_sequential(
        id="competitor_pricing_section",
        desc="Competitor API pricing research",
        parent=root,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A1 - Search for API pricing
    google_scholar_mentioned = has_any_ci(answer, ['google scholar']) and (has_any_ci(answer, ['depth estimation api', 'api pricing', '3d reconstruction api', 'api cost']))
    evaluator.add_custom_node(
        result=bool(google_scholar_mentioned),
        id="google_scholar_search",
        desc="[Action Node] scholar.google.com:F1:A1 - Search Google Scholar for API pricing information",
        parent=competitor_node,
        critical=False
    )

    # [Action Node] scholar.google.com:F1:A8 - Click search results
    verification_attempted = any(c.verification_attempted for c in competitor_pricing.competitors if c.verification_attempted is not None)
    has_competitor_info = len(competitor_pricing.competitors) > 0
    evaluator.add_custom_node(
        result=bool(has_competitor_info),
        id="competitor_service_navigation",
        desc="[Action Node] scholar.google.com:F1:A8 - Navigate to competitor service pricing pages",
        parent=competitor_node,
        critical=False
    )

    # [Perception Node] scholar.google.com:F1:P1 - Extract pricing information
    pricing_extracted = any(c.provider_name and (c.exact_price or c.not_found_explanation) for c in competitor_pricing.competitors)
    evaluator.add_custom_node(
        result=bool(pricing_extracted),
        id="pricing_extraction",
        desc="[Perception Node] scholar.google.com:F1:P1 - Extract competitor pricing details or explain why not found",
        parent=competitor_node,
        critical=False
    )

    # Check pricing model extraction
    pricing_model_extracted = any(c.pricing_model and len(c.pricing_model.strip()) > 0 for c in competitor_pricing.competitors)
    evaluator.add_custom_node(
        result=bool(pricing_model_extracted),
        id="pricing_model_check",
        desc="Pricing models identified (per call / monthly subscription / etc.)",
        parent=competitor_node,
        critical=False
    )

    # -------- 4. Return structured result -------------------------------- #
    return evaluator.get_summary()
