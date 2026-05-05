import json
from pathlib import Path
from typing import Dict, Optional, Any

from loguru import logger

from construct.datasets import SiteCardDataset, TaskDataset
from construct.utils.llm_util import LLMService


class Sampler:
    _PROMPTS_PATH = Path(__file__).resolve().parents[1] / "prompts"

    def __init__(self, sitecard: SiteCardDataset, task_dataset: TaskDataset) -> None:
        self._sitecard = sitecard
        self._tasks = task_dataset

    def select_anchor_cluster(self) -> str:
        clusters = self._sitecard.get_clusters()
        usage = self._tasks.get_cluster_usage()
        return min(clusters, key=lambda c: usage.get(c, 0))

    def prepare_sampling_input(self, anchor_cluster: Optional[str] = None) -> Dict[str, Any]:
        all_clusters = self._sitecard.get_clusters()
        anchor = anchor_cluster or self.select_anchor_cluster()
        other_clusters = [c for c in all_clusters if c != anchor]

        combo_usage = self._tasks.get_combination_usage(anchor)

        for c in other_clusters:
            if c not in combo_usage["order_1"]:
                combo_usage["order_1"][c] = 0

        for i, c1 in enumerate(sorted(other_clusters)):
            for c2 in sorted(other_clusters)[i + 1:]:
                key = f"{c1} + {c2}"
                if key not in combo_usage["order_2"]:
                    combo_usage["order_2"][key] = 0

        website_usage = self._tasks.get_website_usage()
        function_clusters: Dict[str, Dict[str, int]] = {}
        for cluster in all_clusters:
            websites = self._sitecard.get_websites_by_cluster(cluster)
            function_clusters[cluster] = {w: website_usage.get(w, 0) for w in websites}

        return {
            "anchor_cluster": anchor,
            "combination_usage": combo_usage,
            "function_clusters": function_clusters,
        }

    def run(self, anchor_cluster: Optional[str] = None) -> Dict[str, Any]:
        sampling_input = self.prepare_sampling_input(anchor_cluster)

        prompt_path = self._PROMPTS_PATH / "sampling.md"
        prompt_template = prompt_path.read_text(encoding="utf-8")

        user_prompt = (
            prompt_template
            + "\n\n## Input Materials\n\n```json\n"
            + json.dumps(sampling_input, ensure_ascii=False, indent=2)
            + "\n```"
        )
        logger.debug(f"Smart sampling stage prompt (input section): {sampling_input}")

        llm = LLMService()
        response = llm.chat(
            system_prompt="You are a sampling planner for the AI Browser Benchmark.",
            user_prompt=user_prompt,
        )

        result = self._parse_json_response(response)
        result["anchor_cluster"] = sampling_input["anchor_cluster"]
        return result

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
