from pathlib import Path
from collections import Counter
from typing import Dict, Any

from construct.datasets.task import TaskDataset
from construct.datasets.sitecard import SiteCardDataset


class TaskMetricsAnalyzer:
    """Analyzer for FellouBench task dataset metrics."""

    def __init__(self, task_dataset: TaskDataset, sitecard_dataset: SiteCardDataset) -> None:
        self.task_dataset = task_dataset
        self.sitecard_dataset = sitecard_dataset

    def analyze(self) -> Dict[str, Any]:
        self.task_dataset.load_all()

        total_tasks = len(self.task_dataset._tasks)
        total_queries = total_tasks

        total_websites = 0
        website_counter = Counter()

        action_items = Counter()
        perception_items = Counter()

        cluster_counter = Counter()

        for task in self.task_dataset._tasks.values():
            metadata = task.get("metadata", {})

            websites = metadata.get("websites_involved", [])
            total_websites += len(websites)
            website_counter.update(websites)

            action_items_list = metadata.get("action_points_used", [])
            perception_items_list = metadata.get("perception_points_used", [])

            action_items.update(action_items_list)
            perception_items.update(perception_items_list)

            clusters = metadata.get("clusters_used", [])
            cluster_counter.update(clusters)

        avg_websites = total_websites / total_tasks if total_tasks > 0 else 0
        avg_actions = sum(action_items.values()) / total_tasks if total_tasks > 0 else 0
        avg_perceptions = (
            sum(perception_items.values()) / total_tasks if total_tasks > 0 else 0
        )

        total_action_items = sum(action_items.values())
        total_perception_items = sum(perception_items.values())

        total_functions = len(self.sitecard_dataset._functions)
        total_clusters = len(self.sitecard_dataset._clusters)

        return {
            "total_tasks": total_tasks,
            "total_queries": total_queries,
            "avg_websites_per_task": avg_websites,
            "total_unique_websites": len(website_counter),
            "website_coverage": dict(website_counter),
            "total_action_items": total_action_items,
            "avg_action_items_per_task": avg_actions,
            "total_perception_items": total_perception_items,
            "avg_perception_items_per_task": avg_perceptions,
            "cluster_coverage": dict(cluster_counter),
            "total_functions": total_functions,
            "total_clusters": total_clusters,
        }


class MetricsReporter:
    """Reporter for displaying analysis results in a scientific and formal way."""

    @staticmethod
    def print_report(metrics: Dict[str, Any]) -> None:
        print("=" * 60)
        print("FellouBench Task Metrics Analysis Report")
        print("=" * 60)

        print("\n核心指标:")
        print(f"  Query数量: {metrics['total_queries']}")
        print(f"  平均涉及的网站: {metrics['avg_websites_per_task']:.2f}")
        print(f"  复杂执行项总数: {metrics['total_action_items']}")
        print(f"  平均每个Query的复杂执行项: {metrics['avg_action_items_per_task']:.2f}")
        print(f"  复杂感知项总数: {metrics['total_perception_items']}")
        print(f"  平均每个Query的复杂感知项: {metrics['avg_perception_items_per_task']:.2f}")

        print("\n覆盖情况:")
        print("  网站覆盖情况:")
        for website, count in metrics["website_coverage"].items():
            print(f"    - {website}: {count} 次")

        print("  功能集团覆盖情况:")
        for cluster, count in metrics["cluster_coverage"].items():
            print(f"    - {cluster}: {count} 次")

        print("\n数据集规模:")
        print(f"  总功能点数: {metrics['total_functions']}")
        print(f"  总功能集团数: {metrics['total_clusters']}")


def main():
    task_dataset = TaskDataset()
    sitecard_dataset = SiteCardDataset()

    analyzer = TaskMetricsAnalyzer(task_dataset, sitecard_dataset)
    metrics = analyzer.analyze()

    MetricsReporter.print_report(metrics)


if __name__ == "__main__":
    main()

