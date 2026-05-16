import argparse
import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from construct import Proposer, Refiner, Sampler
from construct.datasets import SiteCardDataset, TaskDataset


def run_pipeline(anchor_cluster: str = None, num_proposals: int = 5):
    """Run the full construction pipeline.

    Executes the complete sample -> propose -> refine workflow.

    Args:
        anchor_cluster: Anchor cluster to use. If ``None``, the
            least-frequently-used cluster is auto-selected.
        num_proposals: Number of task proposals to generate per
            sampling round (default: 5).

    Returns:
        tuple: ``(proposal_ids, task_ids)`` — the generated proposal
        and task identifiers.

    Raises:
        FileNotFoundError: If a required data directory is missing.
        RuntimeError: If an LLM invocation fails.
    """
    logger.info("Initializing datasets...")
    sitecard_dataset = SiteCardDataset()
    task_dataset = TaskDataset()

    logger.info(f"SiteCard: {sitecard_dataset}")
    logger.info(f"Task: {task_dataset}")

    # Step 1: sampling
    logger.info("=" * 50)
    logger.info("Step 1: cluster sampling")
    sampler = Sampler(sitecard_dataset, task_dataset)
    sampling_result = sampler.run(anchor_cluster)

    logger.info(f"Anchor cluster: {sampling_result['anchor_cluster']}")
    logger.info(f"Selected clusters: {sampling_result['selected_clusters']}")

    # Step 2: task proposal
    logger.info("=" * 50)
    logger.info(f"Step 2: generating {num_proposals} task proposals")
    proposer = Proposer(sitecard_dataset, task_dataset)
    try:
        proposal_ids = proposer.run(sampling_result, num_proposals)
    except Exception as e:
        logger.error(f"Task proposal failed: {e}")
        proposal_ids = []

    for pid in proposal_ids:
        proposal = task_dataset.get_proposal(pid)
        logger.info(f"  {pid}: {proposal.get('title', 'N/A')}")

    # Step 3: task refinement
    logger.info("=" * 50)
    logger.info("Step 3: refining tasks")
    refiner = Refiner(sitecard_dataset, task_dataset)

    task_ids = []
    for pid in proposal_ids:
        try:
            tid = refiner.run(pid)
            task_ids.append(tid)
            logger.info(f"  {pid} -> {tid} OK")
        except Exception as e:
            logger.error(f"  {pid} task refinement failed: {e}")

    logger.info("=" * 50)
    logger.info(
        f"Done: generated {len(proposal_ids)} proposals, {len(task_ids)} tasks"
    )

    return proposal_ids, task_ids


def setup_logging(level: str = "INFO") -> None:
    """Configure logging output.

    Logs are emitted to stderr and to ``logs/construct/`` with daily
    rotation and a 7-day retention window.

    Args:
        level: Log level — one of ``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``.
    """
    logger.remove()
    logger.add(sys.stderr, level=level)
    log_dir = Path(__file__).parent.parent / "logs" / "construct"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "construct_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments — anchor cluster, batch
        size, total task count, and log level.
    """
    p = argparse.ArgumentParser(
        description="CAP task construction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--anchor",
        type=str,
        help="Anchor cluster name. If omitted, the lowest-usage cluster is auto-selected.",
    )
    p.add_argument("--num", type=int, default=5, help="Task proposal batch size")
    p.add_argument("--total", type=int, default=500, help="Total number of tasks to construct")
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    """Entry point of the CAP construction pipeline.

    Runs three stages in a loop until the requested number of tasks
    is produced:
        1. Sampling — choose a coherent cluster combination.
        2. Proposal — generate task semantic skeletons.
        3. Refinement — instantiate concrete, executable tasks.
    """
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)

    iterations = math.ceil(args.total / args.num)
    logger.info(
        f"Starting batch construction: target {args.total} tasks, "
        f"{args.num} per batch, {iterations} rounds total"
    )

    for i in range(iterations):
        logger.info(f"Round {i + 1}/{iterations}")
        run_pipeline(args.anchor, args.num)


if __name__ == "__main__":
    """
    CAP automatic task construction pipeline with CLI configuration.

    Examples:
        >>> python src/main_construct.py
        >>> python src/main_construct.py --total 100 --num 10 --anchor "navigation"
        >>> python src/main_construct.py --total 50 --num 5 --log_level DEBUG
    """
    main()
