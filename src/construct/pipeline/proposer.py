import json
import re
from pathlib import Path
from typing import Dict, List, Any
from uuid import uuid4

from loguru import logger

from construct.datasets import SiteCardDataset, TaskDataset
from construct.utils import LLMService


class Proposer:
    _PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts"

    def __init__(self, sitecard: SiteCardDataset, task_dataset: TaskDataset) -> None:
        self._sitecard = sitecard
        self._tasks = task_dataset

    def run(self, sampling_result: Dict[str, Any], num_proposals: int = 5) -> List[str]:
        selected_clusters = sampling_result.get("selected_clusters", [])
        cluster_functions = self._sitecard.get_cluster_functions(selected_clusters)

        prompt_path = self._PROMPTS_PATH / "proposing.md"
        prompt_template = prompt_path.read_text(encoding="utf-8")

        input_section = f"""
        ## 输入材料

        ### 采样结果

        ```json
        {json.dumps(sampling_result, ensure_ascii=False, indent=2)}
        ```

        ### 功能点清单

        ```json
        {json.dumps(cluster_functions, ensure_ascii=False, indent=2)}
        ```

        请生成 {num_proposals} 个任务提案。
        """
        user_prompt = prompt_template + input_section
        logger.debug(f"任务提议阶段提示词（输入部分）: {input_section}")

        llm = LLMService()
        response = llm.chat(
            system_prompt="你是AI浏览器Benchmark的任务提议者。",
            user_prompt=user_prompt,
        )

        proposals = self._parse_json_response(response)
        if not isinstance(proposals, list):
            proposals = [proposals]

        proposal_ids: List[str] = []
        for proposal in proposals:
            short_uuid = uuid4().hex[:6]
            proposal["proposal_id"] = f"prop-{short_uuid}"
            pid = self._tasks.save_proposal(proposal)
            proposal_ids.append(pid)

        return proposal_ids

    def _parse_json_response(self, response: str) -> Any:
        response = response.strip()

        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        response = re.sub(r",(\s*[}\]])", r"\1", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
            logger.debug(f"原始响应内容: {response}")
            raise
