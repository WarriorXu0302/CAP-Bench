import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import random

from dotenv import load_dotenv

from construct.datasets.sitecard import SiteCardDataset
from construct.utils.llm_util import LLMService
from construct.module_configs import PROJECT_ROOT as project_root


class MappingTableGenerator:
    """Generate mapping table between sitecard IDs and ui_complexity IDs."""

    def __init__(self, output_path: Path):
        """Initialize with output path.

        Args:
            output_path: Path to output directory for mapping table
        """
        self.output_path = output_path
        self.dataset = SiteCardDataset()
        self.llm = LLMService()

        self.action_types = self._load_action_types()
        self.perception_types = self._load_perception_types()

        self.action_mapping_prompt = """你是一个专业的UI复杂度分类专家，负责将具体的用户界面操作映射到标准的UI复杂度分类体系中。

以下是UI复杂度分类体系中的操作类型(Action Types)：

{action_types_list}

现在，我将提供一系列具体的用户界面操作项，你需要根据它们的组件类别、操作名称、复杂项描述等信息，判断它们最符合哪一个Action Types ID。

请返回一个JSON格式的映射表，格式为：{{"sitecard_id": "ATXX"}}，其中sitecard_id的格式为"网站名:操作ID"。只有当操作项确实与某个Action Types匹配时，才包含在映射表中；如果某个操作项与所有Action Types都不匹配，请完全忽略它，不要包含在映射表中。

输入的操作项列表：
{action_items_list}

返回的JSON映射表: """

        self.perception_mapping_prompt = """你是一个专业的UI复杂度分类专家，负责将具体的用户界面感知项映射到标准的UI复杂度分类体系中。

以下是UI复杂度分类体系中的感知类型(Perception Types)：

{perception_types_list}

现在，我将提供一系列具体的用户界面感知项，你需要根据它们的组件类别、感知名称、复杂项描述等信息，判断它们最符合哪一个Perception Types ID。

请返回一个JSON格式的映射表，格式为：{{"sitecard_id": "PTXX"}}，其中sitecard_id的格式为"网站名:感知ID"。只有当感知项确实与某个Perception Types匹配时，才包含在映射表中；如果某个感知项与所有Perception Types都不匹配，请完全忽略它，不要包含在映射表中。

输入的感知项列表：
{perception_items_list}

返回的JSON映射表: """

    def _load_action_types(self) -> Dict[str, Dict[str, str]]:
        """加载Action Types数据"""
        ui_complexity_path = project_root / "assets" / "ui_complexity" / "ui_complexity.md"
        content = ui_complexity_path.read_text(encoding="utf-8")

        action_types = {}
        lines = content.split("\n")
        in_action_section = False

        for line in lines:
            if line.strip() == "## Action Types (64 Items)":
                in_action_section = True
                continue
            elif line.strip() == "## Perception Types (56 Items)":
                break

            if in_action_section and line.startswith("| ") and "ID" not in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 6:
                    at_id = parts[1]
                    category = parts[2]
                    component = parts[3]
                    action_type = parts[4]
                    description = parts[5]

                    if at_id and at_id != "----":
                        action_types[at_id] = {
                            "category": category,
                            "component": component,
                            "action_type": action_type,
                            "description": description,
                        }

        return action_types

    def _load_perception_types(self) -> Dict[str, Dict[str, str]]:
        """加载Perception Types数据"""
        ui_complexity_path = project_root / "assets" / "ui_complexity" / "ui_complexity.md"
        content = ui_complexity_path.read_text(encoding="utf-8")

        perception_types = {}
        in_perception_section = False
        started = False

        for line in content.split("\n"):
            if line.strip() == "## Perception Types (56 Items)":
                started = True
                in_perception_section = True
                continue

            if in_perception_section and started:
                if line.startswith("| ") and "ID" not in line:
                    parts = [part.strip() for part in line.split("|")]
                    if len(parts) >= 6:
                        pt_id = parts[1]
                        category = parts[2]
                        component = parts[3]
                        perception_type = parts[4]
                        description = parts[5]

                        if pt_id and pt_id != "----":
                            perception_types[pt_id] = {
                                "category": category,
                                "component": component,
                                "perception_type": perception_type,
                                "description": description,
                            }

        return perception_types

    def _format_action_types_list(self) -> str:
        """格式化Action Types列表用于提示词"""
        result = []
        for at_id, data in self.action_types.items():
            result.append(
                f"{at_id}: {data['category']} - {data['component']} - {data['action_type']} - {data['description']}"
            )
        return "\n".join(result)

    def _format_perception_types_list(self) -> str:
        """格式化Perception Types列表用于提示词"""
        result = []
        for pt_id, data in self.perception_types.items():
            result.append(
                f"{pt_id}: {data['category']} - {data['component']} - {data['perception_type']} - {data['description']}"
            )
        return "\n".join(result)

    def _map_action_items(self, action_items: List[Dict], site_name: str) -> Dict[str, str]:
        """映射多个action项到AT ID"""
        if not action_items:
            return {}

        action_types_list = self._format_action_types_list()

        action_items_list = []
        for item in action_items:
            action_items_list.append(
                f"ID: {site_name}:{item['id']}, "
                f"组件类别: {item['category']}, "
                f"组件名称: {item['component']}, "
                f"操作名称: {item['action']}, "
                f"复杂项描述: {item['description']}, "
                f"页面位置: {item['location']}"
            )

        user_prompt = self.action_mapping_prompt.format(
            action_types_list=action_types_list, action_items_list="\n".join(action_items_list)
        )

        print("    正在发送Action映射请求...")
        print(f"    完整请求内容: {user_prompt[:500]}...")

        start_time = time.time()
        try:
            response = self.llm.chat(
                system_prompt="你是一个专业的UI复杂度分类专家，专注于将具体的用户界面操作映射到标准的UI复杂度分类体系中。只有当操作项确实与某个Action Types匹配时才包含在结果中，不匹配的请忽略。",
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"    LLM调用失败: {e}")
            return {}

        end_time = time.time()

        print(f"    完整响应内容: {response}")
        print(f"    映射完成，耗时: {end_time - start_time:.2f}秒")

        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                mapping = json.loads(json_str)
                return mapping
            else:
                print("    无法解析JSON响应")
                return {}
        except json.JSONDecodeError as e:
            print(f"    JSON解析错误: {e}")
            return {}

    def _map_perception_items(
        self, perception_items: List[Dict], site_name: str
    ) -> Dict[str, str]:
        """映射多个perception项到PT ID"""
        if not perception_items:
            return {}

        perception_types_list = self._format_perception_types_list()

        perception_items_list = []
        for item in perception_items:
            perception_items_list.append(
                f"ID: {site_name}:{item['id']}, "
                f"组件类别: {item['category']}, "
                f"组件名称: {item['component']}, "
                f"感知名称: {item['perception']}, "
                f"复杂项描述: {item['description']}, "
                f"页面位置: {item['location']}"
            )

        user_prompt = self.perception_mapping_prompt.format(
            perception_types_list=perception_types_list,
            perception_items_list="\n".join(perception_items_list),
        )

        print("    正在发送Perception映射请求...")
        print(f"    完整请求内容: {user_prompt[:500]}...")

        start_time = time.time()
        try:
            response = self.llm.chat(
                system_prompt="你是一个专业的UI复杂度分类专家，专注于将具体的用户界面感知项映射到标准的UI复杂度分类体系中。只有当感知项确实与某个Perception Types匹配时才包含在结果中，不匹配的请忽略。",
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"    LLM调用失败: {e}")
            return {}

        end_time = time.time()

        print(f"    完整响应内容: {response}")
        print(f"    映射完成，耗时: {end_time - start_time:.2f}秒")

        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                mapping = json.loads(json_str)
                return mapping
            else:
                print("    无法解析JSON响应")
                return {}
        except json.JSONDecodeError as e:
            print(f"    JSON解析错误: {e}")
            return {}

    def _process_sitecard_directory(self, site_dir: Path) -> Tuple[str, Dict, Dict]:
        """处理单个sitecard目录，返回网站名、action映射和perception映射"""
        rel_parts = site_dir.relative_to(self.dataset.data_root_path).parts
        if len(rel_parts) < 2:
            return "", {}, {}

        category, site_name = rel_parts[0], "/".join(rel_parts[1:])

        print(f"处理网站: {site_name} (分类: {category})")

        action_csv = site_dir / "action_items.csv"
        perception_csv = site_dir / "perception_items.csv"

        action_items = []
        perception_items = []

        if action_csv.exists():
            print(f"  发现Action CSV: {action_csv.name}")
            encodings = ["utf-8", "gbk", "gb2312", "latin1"]
            rows = None

            for encoding in encodings:
                try:
                    with open(action_csv, "r", encoding=encoding, newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    print(f"    使用编码 {encoding} 成功读取Action CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"    使用编码 {encoding} 读取Action CSV失败: {e}")
                    continue

            if rows:
                for row in rows:
                    item_id = row.get("id", "")
                    category = row.get("组件类别", "")
                    component = row.get("组件名称", "")
                    action_name = row.get("操作名称", "")
                    description = row.get("复杂项描述", "")
                    location = row.get("页面位置/元素描述", "")

                    action_items.append(
                        {
                            "id": item_id,
                            "category": category,
                            "component": component,
                            "action": action_name,
                            "description": description,
                            "location": location,
                        }
                    )

        if perception_csv.exists():
            print(f"  发现Perception CSV: {perception_csv.name}")
            encodings = ["utf-8", "gbk", "gb2312", "latin1"]
            rows = None

            for encoding in encodings:
                try:
                    with open(perception_csv, "r", encoding=encoding, newline="") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    print(f"    使用编码 {encoding} 成功读取Perception CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"    使用编码 {encoding} 读取Perception CSV失败: {e}")
                    continue

            if rows:
                for row in rows:
                    item_id = row.get("id", "")
                    category = row.get("组件类别", "")
                    component = row.get("组件名称", "")
                    perception_name = row.get("感知名称", "")
                    description = row.get("复杂项描述", "")
                    location = row.get("页面位置/元素描述", "")

                    perception_items.append(
                        {
                            "id": item_id,
                            "category": category,
                            "component": component,
                            "perception": perception_name,
                            "description": description,
                            "location": location,
                        }
                    )

        action_mapping = {}
        if action_items:
            print(f"    发现 {len(action_items)} 个Action项，开始映射...")
            action_mapping = self._map_action_items(action_items, site_name)
            print(f"    Action映射完成，获得 {len(action_mapping)} 个映射")

        perception_mapping = {}
        if perception_items:
            print(f"    发现 {len(perception_items)} 个Perception项，开始映射...")
            perception_mapping = self._map_perception_items(perception_items, site_name)
            print(f"    Perception映射完成，获得 {len(perception_mapping)} 个映射")

        return site_name, action_mapping, perception_mapping

    def generate_mapping_for_site(self, site_dir: Path) -> bool:
        """为指定的sitecard目录生成映射表"""
        site_name, action_mapping, perception_mapping = self._process_sitecard_directory(
            site_dir
        )

        if not site_name:
            print("处理网站信息失败")
            return False

        self.output_path.mkdir(parents=True, exist_ok=True)
        mapping_file = self.output_path / f"{site_name}_mapping.json"

        mapping_data = {
            "site_name": site_name,
            "action_mapping": action_mapping,
            "perception_mapping": perception_mapping,
            "action_types_reference": self.action_types,
            "perception_types_reference": self.perception_types,
        }

        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)

        print(f"映射表已保存到: {mapping_file}")
        print(f"网站: {site_name}")
        print(f"Action映射数量: {len(action_mapping)}")
        print(f"Perception映射数量: {len(perception_mapping)}")

        return True

    def _load_action_types(self) -> Dict[str, Dict[str, str]]:
        ui_complexity_path = project_root() / "assets" / "ui_complexity" / "ui_complexity.md"
        content = ui_complexity_path.read_text(encoding="utf-8")

        action_types = {}
        lines = content.split("\n")
        in_action_section = False

        for line in lines:
            if line.strip() == "## Action Types (64 Items)":
                in_action_section = True
                continue
            if line.strip() == "## Perception Types (56 Items)":
                break

            if in_action_section and line.startswith("| ") and "ID" not in line:
                parts = [part.strip() for part in line.split("|")]
                if len(parts) >= 6:
                    at_id = parts[1]
                    category = parts[2]
                    component = parts[3]
                    action_type = parts[4]
                    description = parts[5]
                    if at_id and at_id != "----":
                        action_types[at_id] = {
                            "category": category,
                            "component": component,
                            "action_type": action_type,
                            "description": description,
                        }

        return action_types

    def _load_perception_types(self) -> Dict[str, Dict[str, str]]:
        ui_complexity_path = project_root() / "assets" / "ui_complexity" / "ui_complexity.md"
        content = ui_complexity_path.read_text(encoding="utf-8")

        perception_types = {}
        in_perception_section = False
        started = False

        for line in content.split("\n"):
            if line.strip() == "## Perception Types (56 Items)":
                started = True
                in_perception_section = True
                continue

            if in_perception_section and started:
                if line.startswith("| ") and "ID" not in line:
                    parts = [part.strip() for part in line.split("|")]
                    if len(parts) >= 6:
                        pt_id = parts[1]
                        category = parts[2]
                        component = parts[3]
                        perception_type = parts[4]
                        description = parts[5]
                        if pt_id and pt_id != "----":
                            perception_types[pt_id] = {
                                "category": category,
                                "component": component,
                                "perception_type": perception_type,
                                "description": description,
                            }

        return perception_types

    def _format_action_types_list(self) -> str:
        result = []
        for at_id, data in self.action_types.items():
            result.append(
                f"{at_id}: {data['category']} - {data['component']} - {data['action_type']} - {data['description']}"
            )
        return "\n".join(result)

    def _format_perception_types_list(self) -> str:
        result = []
        for pt_id, data in self.perception_types.items():
            result.append(
                f"{pt_id}: {data['category']} - {data['component']} - {data['perception_type']} - {data['description']}"
            )
        return "\n".join(result)

    def _parse_json_object(self, response: str) -> dict:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return {}
        json_str = response[start_idx : end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    def _map_action_items(self, action_items: List[Dict], site_name: str) -> Dict[str, str]:
        if not action_items:
            return {}

        action_types_list = self._format_action_types_list()
        action_items_list = []
        for item in action_items:
            action_items_list.append(
                f"ID: {site_name}:{item['id']}, "
                f"组件类别: {item['category']}, "
                f"组件名称: {item['component']}, "
                f"操作名称: {item['action']}, "
                f"复杂项描述: {item['description']}, "
                f"页面位置: {item['location']}"
            )

        user_prompt = self.action_mapping_prompt.format(
            action_types_list=action_types_list, action_items_list="\n".join(action_items_list)
        )

        print("    正在发送Action映射请求...")
        start_time = time.time()
        try:
            response = self.llm.chat(
                system_prompt="你是一个专业的UI复杂度分类专家，专注于将具体的用户界面操作映射到标准的UI复杂度分类体系中。只有当操作项确实与某个Action Types匹配时才包含在结果中，不匹配的请忽略。",
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"    LLM调用失败: {e}")
            return {}
        end_time = time.time()
        print(f"    映射完成，耗时: {end_time - start_time:.2f}秒")
        return self._parse_json_object(response)

    def _map_perception_items(self, perception_items: List[Dict], site_name: str) -> Dict[str, str]:
        if not perception_items:
            return {}

        perception_types_list = self._format_perception_types_list()
        perception_items_list = []
        for item in perception_items:
            perception_items_list.append(
                f"ID: {site_name}:{item['id']}, "
                f"组件类别: {item['category']}, "
                f"组件名称: {item['component']}, "
                f"感知名称: {item['perception']}, "
                f"复杂项描述: {item['description']}, "
                f"页面位置: {item['location']}"
            )

        user_prompt = self.perception_mapping_prompt.format(
            perception_types_list=perception_types_list,
            perception_items_list="\n".join(perception_items_list),
        )

        print("    正在发送Perception映射请求...")
        start_time = time.time()
        try:
            response = self.llm.chat(
                system_prompt="你是一个专业的UI复杂度分类专家，专注于将具体的用户界面感知项映射到标准的UI复杂度分类体系中。只有当感知项确实与某个Perception Types匹配时才包含在结果中，不匹配的请忽略。",
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"    LLM调用失败: {e}")
            return {}
        end_time = time.time()
        print(f"    映射完成，耗时: {end_time - start_time:.2f}秒")
        return self._parse_json_object(response)

    def run(self) -> None:
        root = project_root()
        sites_root = root / "assets" / "sitecard"
        mapping_dir = self.output_path
        mapping_dir.mkdir(parents=True, exist_ok=True)

        for site_dir in sites_root.glob("*/*"):
            if not site_dir.is_dir():
                continue

            rel = site_dir.relative_to(sites_root)
            if len(rel.parts) < 2:
                continue

            site_name = "/".join(rel.parts[1:])
            print(f"处理站点: {site_name}")

            actions = self.dataset._actions.get(site_name, [])
            perceptions = self.dataset._perceptions.get(site_name, [])

            action_mapping = self._map_action_items(actions, site_name)
            perception_mapping = self._map_perception_items(perceptions, site_name)

            data = {
                "action_mapping": action_mapping,
                "perception_mapping": perception_mapping,
                "action_types_reference": self.action_types,
                "perception_types_reference": self.perception_types,
            }

            out_file = mapping_dir / f"{site_name}_mapping.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    load_dotenv()
    output_dir = project_root / "output" / "map_table"

    generator = MappingTableGenerator(output_dir)

    all_site_dirs = []
    for site_dir in generator.dataset.data_root_path.rglob("*"):
        if not site_dir.is_dir():
            continue

        rel_parts = site_dir.relative_to(generator.dataset.data_root_path).parts
        if len(rel_parts) < 2:
            continue

        all_site_dirs.append(site_dir)

    if not all_site_dirs:
        print("未找到任何sitecard目录")
        return

    remaining_unmapped = []
    for site_dir in all_site_dirs:
        rel_parts = site_dir.relative_to(generator.dataset.data_root_path).parts
        if len(rel_parts) < 2:
            continue

        site_name = "/".join(rel_parts[1:])
        mapping_file = generator.output_path / f"{site_name}_mapping.json"

        if not mapping_file.exists():
            remaining_unmapped.append(site_dir)

    total_count = len(remaining_unmapped)
    print(f"总共找到 {total_count} 个待处理的网站")

    while remaining_unmapped:
        current_site_dir = random.choice(remaining_unmapped)

        rel_parts = current_site_dir.relative_to(generator.dataset.data_root_path).parts
        site_name = "/".join(rel_parts[1:])
        mapping_file = generator.output_path / f"{site_name}_mapping.json"

        if mapping_file.exists():
            print(f"跳过已处理网站: {site_name}")
            remaining_unmapped.remove(current_site_dir)
            continue

        print(
            f"正在处理 ({total_count - len(remaining_unmapped) + 1}/{total_count}): {site_name}"
        )
        file_start_time = time.time()
        if generator.generate_mapping_for_site(current_site_dir):
            file_end_time = time.time()
            print(f"  ✓ 已完成 (耗时: {file_end_time - file_start_time:.2f}秒)")

            rel_parts = current_site_dir.relative_to(generator.dataset.data_root_path).parts
            site_name = "/".join(rel_parts[1:])
            mapping_file = generator.output_path / f"{site_name}_mapping.json"
            if mapping_file.exists():
                remaining_unmapped.remove(current_site_dir)
        else:
            file_end_time = time.time()
            print(f"  ✗ 处理失败 (耗时: {file_end_time - file_start_time:.2f}秒)")
            remaining_unmapped.remove(current_site_dir)


if __name__ == "__main__":
    main()
