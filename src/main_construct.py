import argparse
import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from construct import Proposer, Refiner, Sampler
from construct.datasets import SiteCardDataset, TaskDataset


def run_pipeline(anchor_cluster: str = None, num_proposals: int = 5):
    """运行完整Pipeline

    执行采样 -> 提议 -> 完善的完整流程。

    Args:
        anchor_cluster: 指定锚定集团，为None则自动选择（低频优先）
        num_proposals: 每次采样生成的提案数量，默认5个

    Returns:
        tuple: (proposal_ids, task_ids) 生成的提案ID列表和任务ID列表

    Raises:
        FileNotFoundError: 当数据目录不存在时抛出
        RuntimeError: 当LLM调用失败时抛出
    """
    # 初始化数据集
    logger.info("初始化数据集...")
    sitecard_dataset = SiteCardDataset()
    task_dataset = TaskDataset()

    logger.info(f"SiteCard: {sitecard_dataset}")
    logger.info(f"Task: {task_dataset}")

    # Step 1: 采样
    logger.info("=" * 50)
    logger.info("Step 1: 采样分析")
    sampler = Sampler(sitecard_dataset, task_dataset)
    sampling_result = sampler.run(anchor_cluster)

    logger.info(f"锚定集团: {sampling_result['anchor_cluster']}")
    logger.info(f"选中集团: {sampling_result['selected_clusters']}")

    # Step 2: 任务提议
    logger.info("=" * 50)
    logger.info(f"Step 2: 生成 {num_proposals} 个任务提案")
    proposer = Proposer(sitecard_dataset, task_dataset)
    try:
        proposal_ids = proposer.run(sampling_result, num_proposals)
    except Exception as e:
        logger.error(f"任务提议失败: {e}")
        proposal_ids = []

    for pid in proposal_ids:
        proposal = task_dataset.get_proposal(pid)
        logger.info(f"  {pid}: {proposal.get('title', 'N/A')}")

    # Step 3: 任务完善
    logger.info("=" * 50)
    logger.info("Step 3: 完善任务")
    refiner = Refiner(sitecard_dataset, task_dataset)

    task_ids = []
    for pid in proposal_ids:
        try:
            tid = refiner.run(pid)
            task_ids.append(tid)
            logger.info(f"  {pid} -> {tid} ✓")
        except Exception as e:
            logger.error(f"  {pid} 任务完善失败: {e}")

    # 完成
    logger.info("=" * 50)
    logger.info(f"完成: 生成 {len(proposal_ids)} 个提案, {len(task_ids)} 个任务")

    return proposal_ids, task_ids


def setup_logging(level: str = "INFO") -> None:
    """配置日志输出

    同时输出到控制台与 `logs/construct`，按天轮转并保留7天。

    Args:
        level (str): 日志级别，可选 DEBUG/INFO/WARNING/ERROR
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
    """解析命令行参数

    Returns:
        argparse.Namespace: 包含锚定集团、批处理数量和总任务数
    """
    p = argparse.ArgumentParser(
        description="任务构建流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--anchor", type=str, help="指定锚定集团名称，不指定则自动选择低频集团"
    )
    p.add_argument("--num", type=int, default=5, help="任务提案批处理数量")
    p.add_argument("--total", type=int, default=500, help="计划构建任务总数")
    p.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    """Benchmark Pipeline 主入口

    Benchmark任务自动构建Pipeline，包含三个阶段：
        1. 采样：选择功能集团组合
        2. 提议：生成任务语义骨架
        3. 完善：细化为可执行任务
    """
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)

    # 计算需要循环的轮数
    iterations = math.ceil(args.total / args.num)
    logger.info(
        f"开始批量处理: 计划构建 {args.total} 个任务, "
        f"每批 {args.num} 个, 共 {iterations} 轮"
    )

    for i in range(iterations):
        logger.info(f"第 {i + 1}/{iterations} 轮处理")
        run_pipeline(args.anchor, args.num)


if __name__ == "__main__":
    """
    Benchmark任务自动构建Pipeline，支持CLI参数配置。
        Example:
            >>> python src/main_construct.py
            >>> python src/main_construct.py --total 100 --num 10 --anchor "navigation"
            >>> python src/main_construct.py --total 50 --num 5 --log_level DEBUG
    """
    main()
