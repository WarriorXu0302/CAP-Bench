import re
import csv
from pathlib import Path

from construct.module_configs import PROJECT_ROOT


RATIONALITY_PATTERN = r"合理性得分:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
COMPLEXITY_PATTERN = r"复杂性得分:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
EXECUTABILITY_PATTERN = r"可执行性得分:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
EVALUABILITY_PATTERN = r"可评估性得分:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"
COMPREHENSIVE_PATTERN = r"综合评分:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)"


class VerificationReportFilter:
    def __init__(self, reports_path: Path, output_path: Path):
        self.reports_path = reports_path
        self.output_path = output_path
        self.reports = {}

    def extract_scores_from_content(self, content: str) -> dict:
        scores = {}

        rationality_match = re.search(RATIONALITY_PATTERN, content)
        if rationality_match:
            scores["rationality_score"] = float(rationality_match.group(1))

        complexity_match = re.search(COMPLEXITY_PATTERN, content)
        if complexity_match:
            scores["complexity_score"] = float(complexity_match.group(1))

        executability_match = re.search(EXECUTABILITY_PATTERN, content)
        if executability_match:
            scores["executability_score"] = float(executability_match.group(1))

        evaluability_match = re.search(EVALUABILITY_PATTERN, content)
        if evaluability_match:
            scores["evaluability_score"] = float(evaluability_match.group(1))

        comprehensive_match = re.search(COMPREHENSIVE_PATTERN, content)
        if comprehensive_match:
            scores["comprehensive_score"] = float(comprehensive_match.group(1))

        return scores

    def load_reports(self) -> None:
        if not self.reports_path.exists():
            print(f"警告: 报告目录不存在: {self.reports_path}")
            return

        for report_file in self.reports_path.glob("*.md"):
            task_id = report_file.stem
            task_match = re.match(r"(task-[a-zA-Z0-9]+)", task_id)
            if task_match:
                task_id = task_match.group(1)

            try:
                content = report_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = report_file.read_text(encoding="gbk")
                except Exception:
                    print(f"无法读取文件: {report_file}")
                    continue
            except Exception as e:
                print(f"加载报告失败 {report_file}: {e}")
                continue

            scores = self.extract_scores_from_content(content)
            self.reports[task_id] = scores
            print(f"已加载报告: {task_id}")

    def filter_reports(self, filters: dict) -> list:
        filtered_reports = []

        for task_id, scores in self.reports.items():
            include = True

            for metric, (min_val, max_val) in filters.items():
                if metric in scores:
                    score = scores[metric]
                    if not (min_val <= score <= max_val):
                        include = False
                        break
                else:
                    include = False
                    break

            if include:
                filtered_reports.append({"task_id": task_id, "scores": scores})

        return filtered_reports

    def export_to_csv(self, filtered_reports: list) -> None:
        if not filtered_reports:
            print("没有符合条件的报告")
            return

        headers = ["task_id"]
        all_metrics = set()
        for report in filtered_reports:
            all_metrics.update(report["scores"].keys())
        headers.extend(sorted(list(all_metrics)))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for report in filtered_reports:
                row = {"task_id": report["task_id"]}
                row.update(report["scores"])
                writer.writerow(row)

        print(f"结果已导出到: {self.output_path}")


def main():
    filters = {
        "comprehensive_score": (8.0, 10.0),
        "rationality_score": (7.0, 10.0),
        "complexity_score": (6.0, 10.0),
        "executability_score": (7.0, 10.0),
        "evaluability_score": (6.0, 10.0),
    }

    reports_dir = PROJECT_ROOT / "assets" / "verify_report"
    output_file = PROJECT_ROOT / "output" / "verify" / "filtered_tasks.csv"

    analyzer = VerificationReportFilter(reports_dir, output_file)

    print("正在加载验证报告...")
    analyzer.load_reports()

    print(f"总共找到 {len(analyzer.reports)} 个验证报告")
    print("按照配置的筛选条件进行筛选...")
    for metric, (min_val, max_val) in filters.items():
        print(f"  - {metric}: {min_val} ~ {max_val}")

    filtered_reports = analyzer.filter_reports(filters)
    print(f"筛选出符合要求的任务 {len(filtered_reports)} 个")
    analyzer.export_to_csv(filtered_reports)


if __name__ == "__main__":
    main()
