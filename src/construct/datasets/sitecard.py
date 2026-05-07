import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Callable

from construct.module_configs import ASSETS_DIR


class SiteCardDataset:
    """SiteCard 数据集

    提供网站功能点、操作项和感知项的加载与查询功能。

    Attributes:
        data_root_path: 数据根目录路径

    核心功能:
        1. get_clusters() - 获取所有功能集团
        2. get_cluster_functions(clusters) - 获取指定集团的功能点清单
        3. get_function_details(ids) - 获取功能点详细的操作项和感知项
    """

    _ASSETS_PATH = ASSETS_DIR
    _DATA_ROOT_PATH = _ASSETS_PATH / "sitecard"
    _ENCODING = "utf-8"

    def __init__(self, data_root_path: Optional[str] = None) -> None:
        """初始化数据集

        Args:
            data_root_path: 数据根目录路径，默认为 assets/sitecard

        Raises:
            FileNotFoundError: 当数据目录不存在时抛出
        """
        self.data_root_path = Path(data_root_path) if data_root_path else self._DATA_ROOT_PATH

        # 数据存储
        self._functions: Dict[str, Dict] = {}
        self._actions: Dict[str, List[Dict]] = defaultdict(list)
        self._perceptions: Dict[str, List[Dict]] = defaultdict(list)

        # 索引结构
        self._clusters: Set[str] = set()
        self._cluster_sites: Dict[str, Set[str]] = defaultdict(set)

        self._load_data()

    # ==================== 数据加载 ====================

    def _load_data(self) -> None:
        """加载所有数据并构建索引

        遍历数据目录，加载所有功能点、操作项和感知项数据，
        同时构建集团-网站的索引结构。

        Raises:
            FileNotFoundError: 当数据目录不存在时抛出
        """
        if not self.data_root_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_root_path}")

        for site_dir in self.data_root_path.rglob("*"):
            if not site_dir.is_dir():
                continue

            rel_parts = site_dir.relative_to(self.data_root_path).parts
            if len(rel_parts) < 2:
                continue

            category, site_name = rel_parts[0], "/".join(rel_parts[1:])

            # 更新索引
            self._clusters.add(category)
            self._cluster_sites[category].add(site_name)

            # 加载CSV文件
            if (f := site_dir / "functions.csv").exists():
                self._load_functions(f, category, site_name)
            if (f := site_dir / "action_items.csv").exists():
                self._load_items(f, site_name, self._actions, self._parse_action)
            if (f := site_dir / "perception_items.csv").exists():
                self._load_items(f, site_name, self._perceptions, self._parse_perception)

    def _load_functions(self, path: Path, category: str, site_name: str) -> None:
        """加载功能点CSV文件

        Args:
            path: CSV文件路径
            category: 功能集团名称
            site_name: 网站名称
        """
        # 尝试多种编码格式
        encodings = [self._ENCODING, 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    for row in csv.DictReader(f):
                        gid = f"{site_name}:{row['id']}"
                        self._functions[gid] = {
                            'global_id': gid,
                            'category': category,
                            'site': site_name,
                            'id': row['id'],
                            'label': row['label'],
                            'description': row['description']
                        }
                # 成功读取后跳出循环
                break
            except UnicodeDecodeError:
                # 如果不是最后一种编码，继续尝试下一种
                if encoding == encodings[-1]:
                    # 如果所有编码都尝试失败，则抛出异常
                    raise RuntimeError(f"无法使用常见编码读取文件 {path}")
                continue
            except Exception as e:
                raise Exception(f"读取文件 {path} 时出错: {e}") from e

    def _load_items(
            self,
            path: Path,
            site_name: str,
            storage: Dict[str, List[Dict]],
            parser: Callable[[Dict], Dict]
    ) -> None:
        """加载操作项或感知项CSV文件

        Args:
            path: CSV文件路径
            site_name: 网站名称
            storage: 存储字典（_actions 或 _perceptions）
            parser: 行解析函数
        """
        # 尝试多种编码格式
        encodings = [self._ENCODING, 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    for row in csv.DictReader(f):
                        for fid in row.get('对应功能点', '').split('|'):
                            if fid:
                                storage[f"{site_name}:{fid}"].append(parser(row))
                # 成功读取后跳出循环
                break
            except UnicodeDecodeError:
                # 如果不是最后一种编码，继续尝试下一种
                if encoding == encodings[-1]:
                    # 如果所有编码都尝试失败，则抛出异常
                    raise RuntimeError(f"无法使用常见编码读取文件 {path}")
                continue
            except Exception as e:
                raise Exception(f"读取文件 {path} 时出错: {e}") from e

    def _parse_action(self, row: Dict) -> Dict:
        """解析操作项行数据

        Args:
            row: CSV行字典

        Returns:
            解析后的操作项字典
        """
        return {
            'id': row['id'],
            'category': row['组件类别'],
            'component': row['组件名称'],
            'action': row['操作名称'],
            'description': row['复杂项描述'],
            'location': row['页面位置/元素描述'],
            'role': row['在功能点中的作用']
        }

    def _parse_perception(self, row: Dict) -> Dict:
        """解析感知项行数据

        Args:
            row: CSV行字典

        Returns:
            解析后的感知项字典
        """
        return {
            'id': row['id'],
            'category': row['组件类别'],
            'component': row['组件名称'],
            'perception': row['感知名称'],
            'description': row['复杂项描述'],
            'location': row['页面位置/元素描述'],
            'role': row['在功能点中的作用']
        }

    # ==================== 集团查询接口 ====================

    def get_clusters(self) -> List[str]:
        """获取所有功能集团列表

        Returns:
            集团名称列表，按字母序排列

        Example:
            >>> dataset.get_clusters()
            ['Academic Search', 'Code Hosting', 'E-commerce']
        """
        return sorted(self._clusters)

    def get_websites_by_cluster(self, cluster: str) -> List[str]:
        """获取指定集团下的网站列表

        Args:
            cluster: 集团名称

        Returns:
            网站名称列表，按字母序排列

        Example:
            >>> dataset.get_websites_by_cluster("Code Hosting")
            ['github.com', 'gitlab.com']
        """
        return sorted(self._cluster_sites.get(cluster, set()))

    def get_cluster_functions(
            self,
            clusters: List[str]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """获取指定集团的功能点清单（紧凑格式，供LLM选择）

        Args:
            clusters: 集团名称列表

        Returns:
            三层嵌套字典: {cluster: {site: {global_id: "label(description)"}}}

        Example:
            >>> dataset.get_cluster_functions(["Code Hosting"])
            {
                "Code Hosting": {
                    "github.com": {
                        "github.com:F1": "项目搜索(通过关键词搜索开源项目)"
                    }
                }
            }
        """
        result = defaultdict(lambda: defaultdict(dict))
        for gid, func in self._functions.items():
            if func['category'] in clusters:
                result[func['category']][func['site']][gid] = f"{func['label']}({func['description']})"
        return {k: dict(v) for k, v in result.items()}

    # ==================== 功能点详情接口 ====================

    def get_function_details(self, function_ids: List[str]) -> Dict:
        """根据功能点ID列表获取详细信息（含操作项和感知项）

        Args:
            function_ids: 功能点全局ID列表，如 ["github.com:F1", "arxiv.org:F2"]

        Returns:
            功能点详情字典:
            {
                global_id: {
                    function: "label(description)",
                    action: {action_id: "category,component,action,description,location,role"},
                    perception: {perception_id: "..."}
                }
            }

        Example:
            >>> dataset.get_function_details(["github.com:F1"])
            {
                "github.com:F1": {
                    "function": "项目搜索(通过关键词搜索开源项目)",
                    "action": {"A1": "导航组件,Pagination分页,翻页操作,..."},
                    "perception": {"P1": "数据展示组件,Card卡片,Star数识别,..."}
                }
            }
        """
        result = {}
        for gid in function_ids:
            if (func := self._functions.get(gid)) is None:
                continue

            result[gid] = {
                'function': f"{func['label']}({func['description']})",
                'action': {
                    a[
                        'id']: f"{a['category']},{a['component']},{a['action']},{a['description']},{a['location']},{a['role']}"
                    for a in self._actions.get(gid, [])
                },
                'perception': {
                    p[
                        'id']: f"{p['category']},{p['component']},{p['perception']},{p['description']},{p['location']},{p['role']}"
                    for p in self._perceptions.get(gid, [])
                }
            }
        return result

    def get_compact_hierarchical_functions(self) -> Dict:
        """获取所有功能点的紧凑型层级列表

        Returns:
            三层嵌套字典: {category: {site: {global_id: "label(description)"}}}
        """
        return self.get_cluster_functions(list(self._clusters))

    # ==================== 基本方法 ====================

    def __len__(self) -> int:
        """返回功能点总数"""
        return len(self._functions)

    def __repr__(self) -> str:
        """返回数据集的字符串表示"""
        # 计算所有网站的数量
        total_sites = sum(len(sites) for sites in self._cluster_sites.values())
        return f"SiteCardDataset(clusters={len(self._clusters)}, sites={total_sites}, functions={len(self)})"


if __name__ == '__main__':
    """SiteCard 数据集加载器

    加载网站功能点、操作项和感知项数据，支持层级浏览和功能点详情查询。

    数据结构:
        sitecard/
        └── {category}/          # 功能集团目录
            └── {site}/          # 网站目录
                ├── functions.csv
                ├── action_items.csv
                └── perception_items.csv

    """
    dataset = SiteCardDataset()
    clusters = dataset.get_clusters()
    functions = dataset.get_cluster_functions(["Code Hosting"])
    details = dataset.get_function_details(["github.com:F1"])

    print(dataset)
    print(clusters)
    print(functions)
    print(details)
