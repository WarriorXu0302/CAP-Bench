import json
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from construct.datasets import SiteCardDataset, TaskDataset
from construct.utils.llm_util import LLMService


class Refiner:
    _PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts"

    def __init__(self, sitecard: SiteCardDataset, task_dataset: TaskDataset) -> None:
        self._sitecard = sitecard
        self._tasks = task_dataset

    def run(self, proposal_id: str) -> str:
        proposal = self._tasks.get_proposal(proposal_id)
        if proposal is None:
            raise FileNotFoundError(f"Proposal does not exist: {proposal_id}")

        function_points = proposal.get("metadata", {}).get("function_points", [])
        action_perception = self._sitecard.get_function_details(function_points)

        prompt_path = self._PROMPTS_PATH / "refining.md"
        prompt_template = prompt_path.read_text(encoding="utf-8")

        input_section = f"""
        ## Input Materials

        ### Task Proposal

        ```json
        {json.dumps(proposal, ensure_ascii=False, indent=2)}
        ```

        ### Action and Perception Items List

        ```json
        {json.dumps(action_perception, ensure_ascii=False, indent=2)}
        ```
        """
        user_prompt = prompt_template + input_section
        logger.debug(f"Task refinement stage prompt (input section): {input_section}")

        llm = LLMService()
        response = llm.chat(
            system_prompt="You are a task refiner for AI Browser Benchmark.",
            user_prompt=user_prompt,
        )

        task = self._parse_json_response(response)

        short_uuid = proposal_id.replace("prop-", "")
        task["task_id"] = f"task-{short_uuid}"
        task["source_proposal"] = proposal_id

        return self._tasks.save_task(task)

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        response = response.strip()

        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        return json.loads(response)
