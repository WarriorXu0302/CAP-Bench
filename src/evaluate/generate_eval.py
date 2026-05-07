import argparse
import json
import openai
from pathlib import Path

def main():
    # 1. & 3. 使用 add_argument 传参，新增 output_dir 用于保存文件
    parser = argparse.ArgumentParser(description="Generate evaluation script based on JSON task info and a reference Python template.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--py_path", type=str, default="/home/devuser/guyongtong/cap-bench-pipeline/cap-bench-pipeline-main/src/evaluate/eval_scripts/task-1533cc.py", help="Path to the reference Python file")
    parser.add_argument("--output_dir", type=str, default="/home/yongtong/seu/cap-eval/eval_scripts/2026_03_13", help="Directory to save the generated Python file")
    args = parser.parse_args()

    # 2. 解析 JSON 文件并提取指定字段
    with open(args.json_path, 'r', encoding='utf-8') as f:
        full_json_data = json.load(f)
        
    context = {
        "task_id": full_json_data.get("task_id"),
        "task_description": full_json_data.get("task_description"),
        "information_flow_summary": full_json_data.get("information_flow_summary"),
        "covered_points": full_json_data.get("covered_points")
    }

    # 获取 task_id 用于后续给新文件命名
    task_id = context.get("task_id")
    if not task_id:
        # 容错处理：如果 JSON 中没有 task_id 字段
        task_id = "generated_task_default"
        print(f"⚠️ 警告: JSON 中未找到 'task_id'，将使用默认名称 '{task_id}'")

    # 读取 py 参考代码
    with open(args.py_path, 'r', encoding='utf-8') as f:
        reference_code = f.read()

    # 拼接 Prompt (完全基于你提供的模板，不做任何改动)
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
- Prohibited Usage: add_sequence (missing ’ial’), add_serial, update_node_result.
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
—
{reference_code}
—
Please output the Python code directly. Do not include Markdown code block markers,
and ensure the code is directly executable. """

    # 4. 调用大模型
    client = openai.OpenAI(
        api_key="......",
        base_url="https://litellm.fellou.ai" 
    )

    print("🚀 正在调用大模型生成代码，请稍候...")
    response = client.chat.completions.create(
        model='openai/gpt-5', 
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 提取生成的文本内容
    generated_code = response.choices[0].message.content
    
    # 5. 处理保存路径与写入文件
    output_folder = Path(args.output_dir)
    # 如果输出文件夹不存在，则自动创建 (包括父目录)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 构建最终的输出文件路径
    output_file_path = output_folder / f"{task_id}.py"
    
    # 将生成的代码写入该文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(generated_code)

    print(f"\n✅ 生成完毕！代码已成功保存至: {output_file_path.resolve()}")

if __name__ == "__main__":
    main()