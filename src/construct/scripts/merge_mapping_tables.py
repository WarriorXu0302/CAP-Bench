import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any

from construct.module_configs import PROJECT_ROOT as project_root


class MappingTableMerger:
    """Merge all mapping tables and generate statistics."""

    def __init__(self, mapping_dir: Path, output_dir: Path):
        """Initialize with mapping directory and output directory.

        Args:
            mapping_dir: Directory containing mapping table files
            output_dir: Directory to save merged mapping table and statistics
        """
        self.mapping_dir = mapping_dir
        self.output_dir = output_dir
        self.all_action_mappings = {}
        self.all_perception_mappings = {}
        self.action_types_reference = {}
        self.perception_types_reference = {}

    def load_all_mappings(self):
        """加载所有映射表文件"""
        print("正在加载所有映射表文件...")

        mapping_files = list(self.mapping_dir.glob("*_mapping.json"))
        print(f"找到 {len(mapping_files)} 个映射文件")

        for i, mapping_file in enumerate(mapping_files):
            print(f"  处理: {mapping_file.name}")
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                action_mapping = data.get("action_mapping", {})
                self.all_action_mappings.update(action_mapping)

                perception_mapping = data.get("perception_mapping", {})
                self.all_perception_mappings.update(perception_mapping)

                if i == 0:
                    self.action_types_reference = data.get("action_types_reference", {})
                    self.perception_types_reference = data.get(
                        "perception_types_reference", {}
                    )

            except Exception as e:
                print(f"    错误: 无法加载文件 {mapping_file.name}: {e}")

        print(
            f"合并完成 - Action映射: {len(self.all_action_mappings)}, Perception映射: {len(self.all_perception_mappings)}"
        )

    def generate_statistics(self) -> Dict[str, Any]:
        """生成统计信息"""
        print("正在生成统计信息...")

        action_counts = Counter(self.all_action_mappings.values())
        perception_counts = Counter(self.all_perception_mappings.values())

        total_action_types = len(self.action_types_reference)
        total_perception_types = len(self.perception_types_reference)

        covered_action_types = len(action_counts)
        covered_perception_types = len(perception_counts)

        action_coverage_rate = (
            covered_action_types / total_action_types if total_action_types > 0 else 0
        )
        perception_coverage_rate = (
            covered_perception_types / total_perception_types
            if total_perception_types > 0
            else 0
        )

        stats = {
            "summary": {
                "total_action_mappings": len(self.all_action_mappings),
                "total_perception_mappings": len(self.all_perception_mappings),
                "unique_action_types_used": covered_action_types,
                "unique_perception_types_used": covered_perception_types,
                "total_action_types": total_action_types,
                "total_perception_types": total_perception_types,
                "action_coverage_rate": round(action_coverage_rate * 100, 2),
                "perception_coverage_rate": round(perception_coverage_rate * 100, 2),
            }
        }

        return stats

    def save_merged_mapping(self, stats: Dict[str, Any]):
        """保存合并后的映射表和统计信息"""
        print("正在保存合并后的映射表...")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        merged_data = {
            "mapping_tables": {
                "action_mappings": self.all_action_mappings,
                "perception_mappings": self.all_perception_mappings,
            },
            "action_types_reference": self.action_types_reference,
            "perception_types_reference": self.perception_types_reference,
            "statistics": stats,
        }

        output_file = self.output_dir / "unified_ui_complexity_mapping.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f"合并后的映射表已保存到: {output_file}")

        self._generate_summary_report(stats)

    def _generate_summary_report(self, stats: Dict[str, Any]):
        """生成简要统计报告"""
        report_file = self.output_dir / "coverage_summary.txt"

        action_counts = Counter(self.all_action_mappings.values())
        perception_counts = Counter(self.all_perception_mappings.values())

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("FellouBench UI复杂度映射覆盖率统计\n")
            f.write("=" * 40 + "\n\n")

            summary = stats["summary"]
            f.write("覆盖率统计:\n")
            f.write(
                f"  Action类型覆盖率: {summary['action_coverage_rate']}% ({summary['unique_action_types_used']}/{summary['total_action_types']})\n"
            )
            f.write(
                f"  Perception类型覆盖率: {summary['perception_coverage_rate']}% ({summary['unique_perception_types_used']}/{summary['total_perception_types']})\n\n"
            )

            f.write("映射总数:\n")
            f.write(f"  Action映射总数: {summary['total_action_mappings']}\n")
            f.write(f"  Perception映射总数: {summary['total_perception_mappings']}\n\n")

            f.write("Action类型使用频率:\n")
            for at_id, count in sorted(
                action_counts.items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"  {at_id}: {count} 次\n")
            f.write("\n")

            if self.action_types_reference:
                covered_action_ids = set(action_counts.keys())
                all_action_ids = set(self.action_types_reference.keys())
                uncovered_action_ids = sorted(all_action_ids - covered_action_ids)
                f.write("未覆盖的Action类型ID:\n")
                if uncovered_action_ids:
                    for at_id in uncovered_action_ids:
                        f.write(f"  {at_id}\n")
                else:
                    f.write("  无\n")
            f.write("\n")

            f.write("Perception类型使用频率:\n")
            for pt_id, count in sorted(
                perception_counts.items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"  {pt_id}: {count} 次\n")
            f.write("\n")

            if self.perception_types_reference:
                covered_perception_ids = set(perception_counts.keys())
                all_perception_ids = set(self.perception_types_reference.keys())
                uncovered_perception_ids = sorted(all_perception_ids - covered_perception_ids)
                f.write("未覆盖的Perception类型ID:\n")
                if uncovered_perception_ids:
                    for pt_id in uncovered_perception_ids:
                        f.write(f"  {pt_id}\n")
                else:
                    f.write("  无\n")
            f.write("\n")

        print(f"统计报告已保存到: {report_file}")

    def run(self):
        """运行合并和统计过程"""
        print("开始合并映射表...")

        self.load_all_mappings()
        stats = self.generate_statistics()
        self.save_merged_mapping(stats)
        print("映射表合并和统计完成!")


def main():
    mapping_dir = project_root / "output" / "map_table"
    output_dir = project_root / "output" / "map_reduce_res"

    merger = MappingTableMerger(mapping_dir, output_dir)
    merger.run()


if __name__ == "__main__":
    main()
