"""任务数据集

管理proposals和tasks的读写，提供历史统计信息供采样器使用。

数据结构:
    output/
    ├── proposals/           # 任务提案
    │   └── prop-{uuid}.json
    └── tasks/              # 最终任务
        └── task-{uuid}.json

Example:
    >>> dataset = TaskDataset()
    >>> dataset.save_proposal({"title": "测试任务"})
    'prop-a1b2c3'
    >>> dataset.get_cluster_usage()
    {'Code Hosting': 5, 'E-commerce': 3}
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from uuid import uuid4

from construct.module_configs import OUTPUT_DIR


class TaskDataset:
    """任务数据集
    
    管理两类数据:
        - proposals: 任务提案 (output/proposals/prop-{uuid}.json)
        - tasks: 最终任务 (output/tasks/task-{uuid}.json)
    
    Attributes:
        proposals_path: 提案存储目录
        tasks_path: 任务存储目录
    
    核心功能:
        1. save_proposal/get_proposal - 提案读写
        2. save_task/get_task - 任务读写  
        3. get_cluster_usage/get_combination_usage - 历史统计（供采样器使用）
    """

    _OUTPUT_PATH = OUTPUT_DIR
    _PROPOSALS_PATH = _OUTPUT_PATH / "proposals"
    _TASKS_PATH = _OUTPUT_PATH / "tasks"
    _TRASH_PATH = _OUTPUT_PATH / "trash"
    _ENCODING = "utf-8"

    def __init__(
        self, 
        proposals_path: Optional[str] = None,
        tasks_path: Optional[str] = None
    ) -> None:
        """初始化任务数据集
        
        Args:
            proposals_path: 提案目录路径，默认 output/proposals
            tasks_path: 任务目录路径，默认 output/tasks
        """
        self.proposals_path = Path(proposals_path) if proposals_path else self._PROPOSALS_PATH
        self.tasks_path = Path(tasks_path) if tasks_path else self._TASKS_PATH
        
        # 缓存
        self._proposals: Dict[str, Dict] = {}
        self._tasks: Dict[str, Dict] = {}
        self._loaded = False

    # ==================== 数据加载 ====================

    def _ensure_loaded(self) -> None:
        """确保数据已加载（懒加载机制）
        
        首次访问数据时自动触发加载，后续访问使用缓存。
        """
        if not self._loaded:
            self.load_all()

    def load_all(self) -> None:
        """加载所有proposals和tasks到内存
        
        扫描proposals和tasks目录，加载所有JSON文件。
        加载失败的文件会被静默跳过。
        """
        self._proposals.clear()
        self._tasks.clear()
        
        # 加载proposals
        if self.proposals_path.exists():
            for f in self.proposals_path.glob("prop-*.json"):
                try:
                    data = json.loads(f.read_text(encoding=self._ENCODING))
                    self._proposals[data.get("proposal_id", f.stem)] = data
                except (json.JSONDecodeError, IOError):
                    pass
        
        # 加载tasks
        if self.tasks_path.exists():
            for f in self.tasks_path.glob("task-*.json"):
                try:
                    data = json.loads(f.read_text(encoding=self._ENCODING))
                    self._tasks[data.get("task_id", f.stem)] = data
                except (json.JSONDecodeError, IOError):
                    pass
        
        self._loaded = True

    # ==================== Proposal操作 ====================

    def save_proposal(self, proposal: Dict[str, Any]) -> str:
        """保存proposal到文件
        
        Args:
            proposal: 提案字典，若无proposal_id则自动生成
            
        Returns:
            proposal_id，格式为 "prop-{6位uuid}"
            
        Example:
            >>> dataset.save_proposal({"title": "社区推荐验证"})
            'prop-a1b2c3'
        """
        proposal_id = proposal.get("proposal_id") or f"prop-{uuid4().hex[:6]}"
        proposal["proposal_id"] = proposal_id
        
        self.proposals_path.mkdir(parents=True, exist_ok=True)
        path = self.proposals_path / f"{proposal_id}.json"
        path.write_text(
            json.dumps(proposal, ensure_ascii=False, indent=2), 
            encoding=self._ENCODING
        )
        
        self._proposals[proposal_id] = proposal
        return proposal_id

    def get_proposal(self, proposal_id: str) -> Optional[Dict]:
        """获取指定proposal
        
        Args:
            proposal_id: 提案ID
            
        Returns:
            提案字典，不存在则返回None
        """
        self._ensure_loaded()
        return self._proposals.get(proposal_id)

    def list_proposals(self) -> List[str]:
        """列出所有proposal_id
        
        Returns:
            proposal_id列表
        """
        self._ensure_loaded()
        return list(self._proposals.keys())

    # ==================== Task操作 ====================

    def save_task(self, task: Dict[str, Any]) -> str:
        """保存task到文件
        
        Args:
            task: 任务字典，必须包含task_id字段
            
        Returns:
            task_id
            
        Raises:
            ValueError: 当task缺少task_id字段时抛出
            
        Example:
            >>> dataset.save_task({"task_id": "task-a1b2c3", "description": "..."})
            'task-a1b2c3'
        """
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError("task必须包含task_id字段")
        
        self.tasks_path.mkdir(parents=True, exist_ok=True)
        path = self.tasks_path / f"{task_id}.json"
        path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2), 
            encoding=self._ENCODING
        )
        
        self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict]:
        """获取指定task
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务字典，不存在则返回None
        """
        self._ensure_loaded()
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[str]:
        """列出所有task_id
        
        Returns:
            task_id列表
        """
        self._ensure_loaded()
        return list(self._tasks.keys())

    # ==================== 移动到回收站操作 ====================

    def move_proposal_to_trash(self, proposal_id: str) -> bool:
        """将proposal移动到回收站"""
        source = self.proposals_path / f"{proposal_id}.json"
        if not source.exists():
            return False

        trash_dir = self._TRASH_PATH / "proposals"
        trash_dir.mkdir(parents=True, exist_ok=True)

        target = trash_dir / source.name
        if target.exists():
            target.unlink()
        source.rename(target)

        self._proposals.pop(proposal_id, None)
        return True

    def move_task_to_trash(self, task_id: str) -> bool:
        """将task移动到回收站"""
        source = self.tasks_path / f"{task_id}.json"
        if not source.exists():
            return False

        trash_dir = self._TRASH_PATH / "tasks"
        trash_dir.mkdir(parents=True, exist_ok=True)

        target = trash_dir / source.name
        if target.exists():
            target.unlink()
        source.rename(target)

        self._tasks.pop(task_id, None)
        return True

    # ==================== 统计接口（供Sampler使用）====================

    def get_cluster_usage(self) -> Dict[str, int]:
        """统计各集团使用次数
        
        遍历所有已完成任务，统计每个集团被使用的次数。
        
        Returns:
            字典: {cluster_name: usage_count}
            
        Example:
            >>> dataset.get_cluster_usage()
            {'Code Hosting': 5, 'E-commerce': 3, 'Academic Search': 2}
        """
        self._ensure_loaded()
        usage: Dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            for cluster in task.get("metadata", {}).get("clusters_used", []):
                usage[cluster] += 1
        return dict(usage)

    def get_website_usage(self) -> Dict[str, int]:
        """统计各网站使用次数
        
        遍历所有已完成任务，统计每个网站被使用的次数。
        
        Returns:
            字典: {website_name: usage_count}
            
        Example:
            >>> dataset.get_website_usage()
            {'github.com': 8, 'amazon.com': 5, 'arxiv.org': 3}
        """
        self._ensure_loaded()
        usage: Dict[str, int] = defaultdict(int)
        for task in self._tasks.values():
            for website in task.get("metadata", {}).get("websites_involved", []):
                usage[website] += 1
        return dict(usage)

    def get_combination_usage(self, anchor_cluster: str) -> Dict:
        """获取指定锚定集团的组合使用频率
        
        统计以指定集团为锚定时，与其他集团的组合使用情况。
        
        Args:
            anchor_cluster: 锚定集团名称
            
        Returns:
            组合使用频率字典:
            {
                "order_0": int,  # 锚定集团单独使用次数
                "order_1": {other_cluster: count},  # 锚定+1个其他集团
                "order_2": {"cluster1 + cluster2": count}  # 锚定+2个其他集团
            }
            
        Example:
            >>> dataset.get_combination_usage("Code Hosting")
            {
                "order_0": 2,
                "order_1": {"Academic Search": 3, "E-commerce": 1},
                "order_2": {"Academic Search + E-commerce": 1}
            }
        """
        self._ensure_loaded()
        
        order_0 = 0
        order_1: Dict[str, int] = defaultdict(int)
        order_2: Dict[str, int] = defaultdict(int)
        
        for task in self._tasks.values():
            clusters = task.get("metadata", {}).get("clusters_used", [])
            if anchor_cluster not in clusters:
                continue
            
            others = sorted([c for c in clusters if c != anchor_cluster])
            
            if len(others) == 0:
                order_0 += 1
            elif len(others) == 1:
                order_1[others[0]] += 1
            elif len(others) >= 2:
                # 取前两个组成pair，按字母序连接
                key = f"{others[0]} + {others[1]}"
                order_2[key] += 1
        
        return {
            "order_0": order_0,
            "order_1": dict(order_1),
            "order_2": dict(order_2)
        }

    def get_points_usage_stats(self) -> Dict[str, int]:
        """统计各类points的去重计数
        
        遍历所有已完成任务，统计function_points_used、action_points_used、
        perception_points_used中各个点的去重计数。
        
        Returns:
            字典: {
                "unique_function_points": int,  # function_points_used去重计数
                "unique_action_points": int,    # action_points_used去重计数
                "unique_perception_points": int # perception_points_used去重计数
            }
        """
        self._ensure_loaded()
        
        function_points_set = set()
        action_points_set = set()
        perception_points_set = set()
        
        for task in self._tasks.values():
            metadata = task.get("metadata", {})
            function_points = metadata.get("function_points_used", [])
            action_points = metadata.get("action_points_used", [])
            perception_points = metadata.get("perception_points_used", [])
            
            function_points_set.update(function_points)
            action_points_set.update(action_points)
            perception_points_set.update(perception_points)
        
        return {
            "unique_function_points": len(function_points_set),
            "unique_action_points": len(action_points_set),
            "unique_perception_points": len(perception_points_set)
        }

    # ==================== 基本方法 ====================

    def __len__(self) -> int:
        """返回任务总数"""
        self._ensure_loaded()
        return len(self._tasks)

    def __repr__(self) -> str:
        """返回数据集的字符串表示"""
        self._ensure_loaded()
        return f"TaskDataset(proposals={len(self._proposals)}, tasks={len(self._tasks)})"


if __name__ == '__main__':
    dataset = TaskDataset()
    stats = dataset.get_points_usage_stats()
    print("各类型points的去重计数:")
    print(f"function_points_used 去重种类数: {stats['unique_function_points']}")
    print(f"action_points_used 去重种类数: {stats['unique_action_points']}")
    print(f"perception_points_used 去重种类数: {stats['unique_perception_points']}")
