import os
import json
from pathlib import Path

def calculate_eval_metrics(base_dir, target_file_pattern="*.json", filename_keyword=""):
    """Calculate evaluation metrics from result JSON files.

    Args:
        base_dir: Root directory to search for result files.
        target_file_pattern: Glob pattern for file matching (e.g. "*_result.json").
        filename_keyword: Only process files whose name contains this substring.
                          Empty string disables this filter.
    """
    json_files = list(Path(base_dir).rglob(target_file_pattern))
    
    if not json_files:
        print(f"No files matching '{target_file_pattern}' found in {base_dir}.")
        return

    total_files = 0
    total_partial_score = 0.0
    success_count = 0
    
    action_node_scores = []
    perception_node_scores = []

    def traverse_tree(node):
        if not isinstance(node, dict):
            return

        desc = node.get("desc", "")
        score = node.get("score", 0.0)
        
        if "[Action Node]" in desc:
            action_node_scores.append(score)
        if "[Perception Node]" in desc:
            perception_node_scores.append(score)
            
        for child in node.get("children", []):
            traverse_tree(child)

    for file_path in json_files:
        if filename_keyword and filename_keyword not in file_path.name:
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                continue
                
            eval_breakdown = data.get("eval_breakdown", [])
            if not eval_breakdown or not isinstance(eval_breakdown, list):
                continue
                
            if not isinstance(eval_breakdown[0], dict):
                continue
                
            tree = eval_breakdown[0].get("verification_tree")
            if not tree or not isinstance(tree, dict):
                continue
            
            total_files += 1
            
            root_score = tree.get("score", 0.0)
            total_partial_score += root_score
            
            if root_score == 1.0:
                success_count += 1
                
            traverse_tree(tree)
            
        except json.JSONDecodeError:
            print(f"Warning: invalid JSON in {file_path}")
        except Exception as e:
            print(f"Warning: error processing {file_path} - {e}")

    if total_files == 0:
        print("No valid result files with verification_tree found. Check filter settings.")
        return

    partial_completion = total_partial_score / total_files
    success_rate = success_count / total_files
    complex_a = sum(action_node_scores) / len(action_node_scores) if action_node_scores else 0.0
    complex_p = sum(perception_node_scores) / len(perception_node_scores) if perception_node_scores else 0.0

    print("-" * 40)
    print(f"Valid files:          {total_files}")
    print(f"Action nodes:         {len(action_node_scores)}")
    print(f"Perception nodes:     {len(perception_node_scores)}")
    print("-" * 40)
    print(f"Partial Completion:   {partial_completion:.4f}")
    print(f"Success Rate:         {success_rate:.4f} ({success_rate * 100:.2f}%)")
    print(f"Complex A:            {complex_a:.4f}")
    print(f"Complex P:            {complex_p:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    folder_path = "./cap"
    
    calculate_eval_metrics(
        base_dir=folder_path, 
        target_file_pattern="*.json",
        filename_keyword="comet"
    )
