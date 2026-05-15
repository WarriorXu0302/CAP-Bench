"""Generate a per-task evaluation script from a task JSON using an LLM.

Inputs:
  --json_path        Path to a task JSON containing task_id, task_description,
                     information_flow_summary, covered_points
  --reference_script Path to a reference Python script the LLM should imitate.
                     Defaults to eval_scripts/<eval_version>/<task-1533cc>.py
                     under the cap-eval project root.
  --output_dir       Where to write the generated <task_id>.py
  --eval_version     Version subfolder under eval_scripts/ (default: 2026_03_13)
  --model            LLM model id used to synthesise the script
                     (default: gpt-4o, override via OPENAI_MODEL env var)

Environment:
  OPENAI_API_KEY     Required.
  OPENAI_BASE_URL    Optional. Set this to point at any OpenAI-compatible
                     endpoint (Azure, vLLM, LiteLLM proxy, etc.).
  OPENAI_MODEL       Optional default model id.
"""

import argparse
import json
import os
from pathlib import Path

import openai


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_VERSION = "2026_03_13"
DEFAULT_REFERENCE = REPO_ROOT / "eval_scripts" / DEFAULT_EVAL_VERSION / "task-1533cc.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a CAP-Eval evaluation script for a single task.",
    )
    parser.add_argument(
        "--json_path", type=Path, required=True,
        help="Path to the input task JSON file",
    )
    parser.add_argument(
        "--reference_script", type=Path, default=None,
        help=f"Reference Python template (default: {DEFAULT_REFERENCE})",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None,
        help=f"Output directory (default: eval_scripts/<eval_version>/)",
    )
    parser.add_argument(
        "--eval_version", default=DEFAULT_EVAL_VERSION,
        help=f"Evaluation script version subfolder (default: {DEFAULT_EVAL_VERSION})",
    )
    parser.add_argument(
        "--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"),
        help="LLM model id (default: $OPENAI_MODEL or gpt-4o)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    reference_script = args.reference_script or DEFAULT_REFERENCE
    output_dir = args.output_dir or (REPO_ROOT / "eval_scripts" / args.eval_version)

    with open(args.json_path, "r", encoding="utf-8") as f:
        full_json_data = json.load(f)

    context = {
        "task_id": full_json_data.get("task_id"),
        "task_description": full_json_data.get("task_description"),
        "information_flow_summary": full_json_data.get("information_flow_summary"),
        "covered_points": full_json_data.get("covered_points"),
    }

    task_id = context.get("task_id") or "generated_task_default"

    with open(reference_script, "r", encoding="utf-8") as f:
        reference_code = f.read()

    prompt = f"""You are a skilled evaluation annotator. Please write a Python evaluation script based
on the CAP-Eval framework for a new task, using the following task information
and reference code.
### New Task Information
{json.dumps(context, indent=2, ensure_ascii=False)}
### Core Requirements
1. Focus on covered_points & Node Naming: Design evaluation points based on the covered_points field.
- ONLY add the "[Action Node]" or "[Perception Node]" prefix to the `desc` field if the node directly corresponds to a specific point_id in `covered_points`.
- Other regular nodes, container nodes, or general validation nodes MUST NOT have these prefixes.
- point_id examples: "ikea.com:F2:A8" (Action) or "ikea.com:F9:P6" (Perception).
2. Minimize Critical Nodes (Allow Partial Scoring):
- Cause of Error: The framework stipulates that if a parent node is critical=True, all its child nodes must also be critical=True, which easily causes the entire evaluation tree to fail early.
- Design Recommendation: Minimize the use of `critical=True`. Set almost all nodes (both container and leaf nodes) to `critical=False` to maximize partial credit. Only set a node to `critical=True` if it is an absolute showstopper (e.g., the core destination address is completely missing).
3. Do not deviate from the task: Do not add irrelevant evaluation points not
mentioned in the task description.
4. Limit LLM Verification (evaluator.verify): You MUST restrict the usage of `evaluator.verify(...)` to AT MOST 1 time per evaluation task. For all other assertions, use `evaluator.add_custom_node` with basic, lightweight Python logic to save LLM calls.
5. Lenient Evaluation Logic: Lower the strictness of the evaluation code to make it slightly easier for the model to pass. Use fault-tolerant logic, such as case-insensitive substring matching (e.g., checking if "1 bed" or "one bedroom" is in the text), safe type casting, and handling `None` gracefully without throwing exceptions.
### Code Writing Specifications (API Whitelist)
You must strictly adhere to the following APIs; do not invent non-existent methods:
1. Allowed Methods:
- evaluator.initialize(...): Initialize, returns the root node.
- evaluator.add_sequential(...): Can be used (Note: not add_sequence). Used to
add sequential execution container nodes.
- evaluator.add_parallel(...): Used to add parallel execution container nodes.
- evaluator.add_leaf(...): Add a leaf node.
- evaluator.add_custom_node(...): Pass in a boolean result node directly.
- evaluator.extract(...): Extract information.
- evaluator.verify(...): Verification logic.
- evaluator.get_summary(): Return the result.
2. Advanced Technique (verify with node=None):
- If you need to obtain a verification result (True/False) in the Python code to
determine subsequent logic, but do not want to draw this node on the evaluation
tree, use await evaluator.verify(node=None, ...).
3. Strictly Prohibit Hallucinations (Negative Constraints):
- Prohibited Usage: add_sequence (missing 'ial'), add_serial, update_node_result.
- Prohibited Attribute Access: The Evaluator object does not have .logger or
.answer attributes.
- Correct Practice: logger and answer are arguments of the evaluate_answer
function; use the argument variables directly (e.g., logger.info(...)).
4. Data Structures:
- Use pydantic.BaseModel to define extraction structures.
- The evaluate_answer function signature must remain complete (containing all
arguments: client, answer, cache, semaphore, logger, model etc.).
### Reference Code Template
Please imitate the structure, import style, and verification logic of the following
code:
---
{reference_code}
---
Please output the Python code directly. Do not include Markdown code block markers,
and ensure the code is directly executable. """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running.")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    print(f"Calling {args.model} to generate evaluation script for {task_id}...")
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
    )
    generated_code = response.choices[0].message.content

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{task_id}.py"
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    print(f"Saved generated evaluation script to {output_file_path.resolve()}")


if __name__ == "__main__":
    main()
