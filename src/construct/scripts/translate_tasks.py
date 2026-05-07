import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from construct.datasets.task import TaskDataset
from construct.utils.llm_util import LLMService
from construct.module_configs import PROJECT_ROOT


class TaskTranslator:
    def __init__(self, tasks_path: Path, output_path: Path):
        self.tasks_path = tasks_path
        self.output_path = output_path
        self.dataset = TaskDataset()
        self.llm = LLMService()

        self.translation_prompt = """你是一个专业的翻译助手，负责将AI浏览器Benchmark任务从中文翻译成英文。

翻译要求：
1. 保持任务的专业性和准确性
2. 确保任务描述的自然合理性
3. 如果任务涉及到预约的时间，你就把时间动态模糊化处理。例如，如果是我打算在2025年12月12日到12月14日这个周末去度个短假，找3家该城市的酒店。要求：入住时间对应我的旅行日期。这种任务很有可能执行的时候已经过了时间，所有应该是改成下个星期几-星期几

以下是需要翻译的任务描述，请仅返回翻译后的英文内容："""

    def translate_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return text

        user_prompt = f"{self.translation_prompt}\n\n{text}"
        print(f"    正在发送翻译请求...")
        print(f"    完整请求内容: {user_prompt}")

        start_time = time.time()
        response = self.llm.chat(
            system_prompt="你是一个专业的翻译助手，专注于将AI浏览器Benchmark任务从中文翻译成英文。",
            user_prompt=user_prompt,
        )
        end_time = time.time()

        print(f"    完整响应内容: {response}")
        print(f"    翻译完成，耗时: {end_time - start_time:.2f}秒")
        return response.strip()

    def translate_task_description(self, task_file: Path) -> bool:
        try:
            print(f"  读取任务文件: {task_file.name}")
            content = task_file.read_text(encoding="utf-8")
            task_data = json.loads(content)

            task_description = task_data.get("task_description", "")
            if not task_description:
                print(f"  警告: 文件中没有找到task_description字段，跳过: {task_file.name}")
                return False

            print("  提取task_description字段进行翻译...")
            translated_description = self.translate_text(task_description)
            task_data["task_description"] = translated_description

            self.output_path.mkdir(parents=True, exist_ok=True)

            output_file = self.output_path / task_file.name
            output_file.write_text(
                json.dumps(task_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"  保存翻译结果到: {output_file}")
            return True
        except json.JSONDecodeError:
            print(f"错误: 无法解析JSON文件 {task_file}")
            return False
        except UnicodeDecodeError as e:
            print(f"错误: 无法解码文件 {task_file}, 编码问题: {e}")
            return False
        except Exception as e:
            print(f"翻译任务文件失败 {task_file}: {e}")
            return False

    def translate_all_tasks(self) -> None:
        if not self.tasks_path.exists():
            print(f"警告: 任务目录不存在: {self.tasks_path}")
            return

        task_files = list(self.tasks_path.glob("task-*.json"))
        if not task_files:
            print("没有找到任务文件")
            return

        print(f"找到 {len(task_files)} 个任务文件")

        if self.output_path.exists():
            translated_files = {f.name for f in self.output_path.glob("task-*.json")}
        else:
            translated_files = set()

        untranslated_files = [f for f in task_files if f.name not in translated_files]

        print(f"已翻译: {len(translated_files)} 个任务文件")
        print(f"待翻译: {len(untranslated_files)} 个任务文件")

        if not untranslated_files:
            print("所有任务文件都已翻译完成")
            return

        success_count = 0
        total_count = len(untranslated_files)

        start_time = time.time()
        remaining_untranslated = untranslated_files.copy()

        while remaining_untranslated:
            current_task_file = random.choice(remaining_untranslated)

            if self.output_path.exists():
                translated_files = {f.name for f in self.output_path.glob("task-*.json")}
                if current_task_file.name in translated_files:
                    print(f"跳过已翻译文件: {current_task_file.name}")
                    remaining_untranslated.remove(current_task_file)
                    continue

            print(
                f"正在翻译 ({total_count - len(remaining_untranslated) + 1}/{total_count}): {current_task_file.name}"
            )
            file_start_time = time.time()
            if self.translate_task_description(current_task_file):
                success_count += 1
                file_end_time = time.time()
                print(f"  ✓ 已完成 (耗时: {file_end_time - file_start_time:.2f}秒)")
                remaining_untranslated.remove(current_task_file)
            else:
                file_end_time = time.time()
                print(f"  ✗ 翻译失败 (耗时: {file_end_time - file_start_time:.2f}秒)")
                remaining_untranslated.remove(current_task_file)

        total_end_time = time.time()
        print(f"\n翻译完成: {success_count}/{total_count} 个任务文件已成功翻译")
        print(f"总耗时: {total_end_time - start_time:.2f}秒")

        final_task_files = list(self.tasks_path.glob("task-*.json"))
        final_untranslated = [
            f
            for f in final_task_files
            if f.name not in {f.name for f in self.output_path.glob('task-*.json')}
        ]
        if final_untranslated:
            print(f"仍有 {len(final_untranslated)} 个任务文件待翻译")
        else:
            print("所有任务文件均已翻译完成")


def main():
    load_dotenv()
    tasks_dir = PROJECT_ROOT / "output" / "tasks"
    output_dir = PROJECT_ROOT / "output" / "translate"

    translator = TaskTranslator(tasks_dir, output_dir)

    start_time = time.time()
    print("正在翻译任务文件...")
    translator.translate_all_tasks()
    end_time = time.time()
    print(f"总执行时间: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    main()
