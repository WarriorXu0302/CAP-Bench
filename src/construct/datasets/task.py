"""Task Dataset

Manages reading and writing of proposals and tasks, providing historical statistics for samplers.

Data Structure:
    output/
    ├── proposals/           # Task proposals
    │   └── prop-{uuid}.json
    └── tasks/              # Final tasks
        └── task-{uuid}.json

Example:
    >>> dataset = TaskDataset()
    >>> dataset.save_proposal({"title": "Test Task"})
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
    """Task Dataset
    
    Manages two types of data:
        - proposals: Task proposals (output/proposals/prop-{uuid}.json)
        - tasks: Final tasks (output/tasks/task-{uuid}.json)
    
    Attributes:
        proposals_path: Proposal storage directory
        tasks_path: Task storage directory
    
    Core Functionality:
        1. save_proposal/get_proposal - Proposal read/write operations
        2. save_task/get_task - Task read/write operations  
        3. get_cluster_usage/get_combination_usage - Historical statistics (for sampler use)
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
        """Initialize the task dataset.
        
        Args:
            proposals_path: Proposal directory path, defaults to output/proposals
            tasks_path: Task directory path, defaults to output/tasks
        """
        self.proposals_path = Path(proposals_path) if proposals_path else self._PROPOSALS_PATH
        self.tasks_path = Path(tasks_path) if tasks_path else self._TASKS_PATH
        
        # Cache
        self._proposals: Dict[str, Dict] = {}
        self._tasks: Dict[str, Dict] = {}
        self._loaded = False

    # ==================== Data Loading ====================

    def _ensure_loaded(self) -> None:
        """Ensure data is loaded (lazy loading mechanism).
        
        Automatically triggers loading on first data access, uses cache for subsequent accesses.
        """
        if not self._loaded:
            self.load_all()

    def load_all(self) -> None:
        """Load all proposals and tasks into memory.
        
        Scans proposals and tasks directories, loads all JSON files.
        Files that fail to load are silently skipped.
        """
        self._proposals.clear()
        self._tasks.clear()
        
        # Load proposals
        if self.proposals_path.exists():
            for f in self.proposals_path.glob("prop-*.json"):
                try:
                    data = json.loads(f.read_text(encoding=self._ENCODING))
                    self._proposals[data.get("proposal_id", f.stem)] = data
                except (json.JSONDecodeError, IOError):
                    pass
        
        # Load tasks
        if self.tasks_path.exists():
            for f in self.tasks_path.glob("task-*.json"):
                try:
                    data = json.loads(f.read_text(encoding=self._ENCODING))
                    self._tasks[data.get("task_id", f.stem)] = data
                except (json.JSONDecodeError, IOError):
                    pass
        
        self._loaded = True

    # ==================== Proposal Operations ====================

    def save_proposal(self, proposal: Dict[str, Any]) -> str:
        """Save proposal to file.
        
        Args:
            proposal: Proposal dictionary, generates proposal_id automatically if not present
            
        Returns:
            proposal_id, format is "prop-{6-digit uuid}"
            
        Example:
            >>> dataset.save_proposal({"title": "Community Recommendation Verification"})
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
        """Get specified proposal.
        
        Args:
            proposal_id: Proposal ID
            
        Returns:
            Proposal dictionary, returns None if not exists
        """
        self._ensure_loaded()
        return self._proposals.get(proposal_id)

    def list_proposals(self) -> List[str]:
        """List all proposal_ids.
        
        Returns:
            List of proposal_ids
        """
        self._ensure_loaded()
        return list(self._proposals.keys())

    # ==================== Task Operations ====================

    def save_task(self, task: Dict[str, Any]) -> str:
        """Save task to file.
        
        Args:
            task: Task dictionary, must contain task_id field
            
        Returns:
            task_id
            
        Raises:
            ValueError: Raised when task lacks task_id field
            
        Example:
            >>> dataset.save_task({"task_id": "task-a1b2c3", "description": "..."})
            'task-a1b2c3'
        """
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError("Task must contain task_id field")
        
        self.tasks_path.mkdir(parents=True, exist_ok=True)
        path = self.tasks_path / f"{task_id}.json"
        path.write_text(
            json.dumps(task, ensure_ascii=False, indent=2), 
            encoding=self._ENCODING
        )
        
        self._tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get specified task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task dictionary, returns None if not exists
        """
        self._ensure_loaded()
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[str]:
        """List all task_ids.
        
        Returns:
            List of task_ids
        """
        self._ensure_loaded()
        return list(self._tasks.keys())

    # ==================== Move to Trash Operations ====================

    def move_proposal_to_trash(self, proposal_id: str) -> bool:
        """Move proposal to trash."""
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
        """Move task to trash."""
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

    # ==================== Statistics Interface (for Sampler use) ====================

    def get_cluster_usage(self) -> Dict[str, int]:
        """Count usage frequency of each cluster.
        
        Iterates through all completed tasks and counts how many times each cluster is used.
        
        Returns:
            Dictionary: {cluster_name: usage_count}
            
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
        """Count usage frequency of each website.
        
        Iterates through all completed tasks and counts how many times each website is used.
        
        Returns:
            Dictionary: {website_name: usage_count}
            
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
        """Get combination usage frequency for specified anchor cluster.
        
        Counts how often the specified cluster is combined with other clusters.
        
        Args:
            anchor_cluster: Name of the anchor cluster
            
        Returns:
            Combination usage frequency dictionary:
            {
                "order_0": int,  # Number of times anchor cluster used alone
                "order_1": {other_cluster: count},  # Anchor + 1 other cluster
                "order_2": {"cluster1 + cluster2": count}  # Anchor + 2 other clusters
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
                # Take first two to form pair, connect in alphabetical order
                key = f"{others[0]} + {others[1]}"
                order_2[key] += 1
        
        return {
            "order_0": order_0,
            "order_1": dict(order_1),
            "order_2": dict(order_2)
        }

    def get_points_usage_stats(self) -> Dict[str, int]:
        """Count unique occurrences of various point types.
        
        Iterates through all completed tasks and counts unique occurrences
        of points in function_points_used, action_points_used, and perception_points_used.
        
        Returns:
            Dictionary: {
                "unique_function_points": int,  # Unique count of function_points_used
                "unique_action_points": int,    # Unique count of action_points_used
                "unique_perception_points": int # Unique count of perception_points_used
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

    # ==================== Basic Methods ====================

    def __len__(self) -> int:
        """Return total number of tasks."""
        self._ensure_loaded()
        return len(self._tasks)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        self._ensure_loaded()
        return f"TaskDataset(proposals={len(self._proposals)}, tasks={len(self._tasks)})"


if __name__ == '__main__':
    dataset = TaskDataset()
    stats = dataset.get_points_usage_stats()
    print("Unique counts for each point type:")
    print(f"Unique function_points_used: {stats['unique_function_points']}")
    print(f"Unique action_points_used: {stats['unique_action_points']}")
    print(f"Unique perception_points_used: {stats['unique_perception_points']}")
