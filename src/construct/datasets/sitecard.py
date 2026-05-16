import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, Callable

from construct.module_configs import ASSETS_DIR


class SiteCardDataset:
    """SiteCard Dataset for website functionality annotation.

    Provides loading and querying capabilities for website functional points,
    action items, and perception items.

    Attributes:
        data_root_path: Root directory path for the dataset.

    Core Functionality:
        1. get_clusters() - Retrieve all functional clusters.
        2. get_cluster_functions(clusters) - Get function inventory for specified clusters.
        3. get_function_details(ids) - Retrieve detailed action and perception items for functions.
    """

    _ASSETS_PATH = ASSETS_DIR
    _DATA_ROOT_PATH = _ASSETS_PATH / "sitecard"
    _ENCODING = "utf-8"

    def __init__(self, data_root_path: Optional[str] = None) -> None:
        """Initialize the dataset.

        Args:
            data_root_path: Root directory path for data, defaults to assets/sitecard.

        Raises:
            FileNotFoundError: Raised when the data directory does not exist.
        """
        self.data_root_path = Path(data_root_path) if data_root_path else self._DATA_ROOT_PATH

        # Data storage
        self._functions: Dict[str, Dict] = {}
        self._actions: Dict[str, List[Dict]] = defaultdict(list)
        self._perceptions: Dict[str, List[Dict]] = defaultdict(list)

        # Index structures
        self._clusters: Set[str] = set()
        self._cluster_sites: Dict[str, Set[str]] = defaultdict(set)

        self._load_data()

    # ==================== Data Loading ====================

    def _load_data(self) -> None:
        """Load all data and build indices.

        Traverses the data directory to load all functional points, action items,
        and perception items, while constructing cluster-website index structures.

        Raises:
            FileNotFoundError: Raised when the data directory does not exist.
        """
        if not self.data_root_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_root_path}")

        for site_dir in self.data_root_path.rglob("*"):
            if not site_dir.is_dir():
                continue

            rel_parts = site_dir.relative_to(self.data_root_path).parts
            if len(rel_parts) < 2:
                continue

            category, site_name = rel_parts[0], "/".join(rel_parts[1:])

            # Update indices
            self._clusters.add(category)
            self._cluster_sites[category].add(site_name)

            # Load CSV files
            if (f := site_dir / "functions.csv").exists():
                self._load_functions(f, category, site_name)
            if (f := site_dir / "action_items.csv").exists():
                self._load_items(f, site_name, self._actions, self._parse_action)
            if (f := site_dir / "perception_items.csv").exists():
                self._load_items(f, site_name, self._perceptions, self._parse_perception)

    def _load_functions(self, path: Path, category: str, site_name: str) -> None:
        """Load functional points from CSV file.

        Args:
            path: Path to the CSV file.
            category: Name of the functional cluster.
            site_name: Name of the website.
        """
        # Try multiple encoding formats
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
                # Break loop after successful reading
                break
            except UnicodeDecodeError:
                # Continue trying next encoding if not the last one
                if encoding == encodings[-1]:
                    # Raise exception if all encodings fail
                    raise RuntimeError(f"Cannot read file {path} with common encodings")
                continue
            except Exception as e:
                raise Exception(f"Error reading file {path}: {e}") from e

    def _load_items(
            self,
            path: Path,
            site_name: str,
            storage: Dict[str, List[Dict]],
            parser: Callable[[Dict], Dict]
    ) -> None:
        """Load action or perception items from CSV file.

        Args:
            path: Path to the CSV file.
            site_name: Name of the website.
            storage: Storage dictionary (_actions or _perceptions).
            parser: Row parsing function.
        """
        # Try multiple encoding formats
        encodings = [self._ENCODING, 'gbk', 'gb2312', 'latin1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    for row in csv.DictReader(f):
                        for fid in row.get('对应功能点', '').split('|'):
                            if fid:
                                storage[f"{site_name}:{fid}"].append(parser(row))
                # Break loop after successful reading
                break
            except UnicodeDecodeError:
                # Continue trying next encoding if not the last one
                if encoding == encodings[-1]:
                    # Raise exception if all encodings fail
                    raise RuntimeError(f"Cannot read file {path} with common encodings")
                continue
            except Exception as e:
                raise Exception(f"Error reading file {path}: {e}") from e

    def _parse_action(self, row: Dict) -> Dict:
        """Parse action item row data.

        Args:
            row: CSV row dictionary.

        Returns:
            Parsed action item dictionary.
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
        """Parse perception item row data.

        Args:
            row: CSV row dictionary.

        Returns:
            Parsed perception item dictionary.
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

    # ==================== Cluster Query Interface ====================

    def get_clusters(self) -> List[str]:
        """Get list of all functional clusters.

        Returns:
            List of cluster names, sorted alphabetically.

        Example:
            >>> dataset.get_clusters()
            ['Academic Search', 'Code Hosting', 'E-commerce']
        """
        return sorted(self._clusters)

    def get_websites_by_cluster(self, cluster: str) -> List[str]:
        """Get list of websites under a specified cluster.

        Args:
            cluster: Name of the cluster.

        Returns:
            List of website names, sorted alphabetically.

        Example:
            >>> dataset.get_websites_by_cluster("Code Hosting")
            ['github.com', 'gitlab.com']
        """
        return sorted(self._cluster_sites.get(cluster, set()))

    def get_cluster_functions(
            self,
            clusters: List[str]
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Get functional point inventory for specified clusters (compact format for LLM selection).

        Args:
            clusters: List of cluster names.

        Returns:
            Three-level nested dictionary: {cluster: {site: {global_id: "label(description)"}}}

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

    # ==================== Function Details Interface ====================

    def get_function_details(self, function_ids: List[str]) -> Dict:
        """Get detailed information for function IDs (including action and perception items).

        Args:
            function_ids: List of function global IDs, e.g., ["github.com:F1", "arxiv.org:F2"].

        Returns:
            Function details dictionary:
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
        """Get compact hierarchical list of all functional points.

        Returns:
            Three-level nested dictionary: {category: {site: {global_id: "label(description)"}}}
        """
        return self.get_cluster_functions(list(self._clusters))

    # ==================== Basic Methods ====================

    def __len__(self) -> int:
        """Return total number of functional points."""
        return len(self._functions)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        # Calculate total number of websites
        total_sites = sum(len(sites) for sites in self._cluster_sites.values())
        return f"SiteCardDataset(clusters={len(self._clusters)}, sites={total_sites}, functions={len(self)})"


if __name__ == '__main__':
    """SiteCard Dataset Loader

    Loads website functional points, action items, and perception items data,
    supporting hierarchical browsing and function detail queries.

    Data Structure:
        sitecard/
        └── {category}/          # Functional cluster directory
            └── {site}/          # Website directory
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
