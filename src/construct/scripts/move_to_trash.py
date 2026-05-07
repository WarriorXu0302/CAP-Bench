import re

from construct.datasets.task import TaskDataset


def get_ids_from_input():
    print("请粘贴ID列表（每行一个，以空行结束）:")
    print("示例: task-002f04 或 prop-002f04\n")

    ids = set()
    pattern = re.compile(r"^(?:task|prop)-([a-zA-Z0-9]+)$")

    while True:
        try:
            line = input().strip()
            if not line:
                break
            match = pattern.match(line)
            if match:
                ids.add(match.group(1))
            else:
                print(f"警告: '{line}' 格式无效，已跳过")
        except EOFError:
            break

    return list(ids)


def move_ids_to_trash(dataset, ids):
    success, failed = 0, 0

    for id_ in ids:
        task_ok = dataset.move_task_to_trash(f"task-{id_}")
        prop_ok = dataset.move_proposal_to_trash(f"prop-{id_}")

        if task_ok or prop_ok:
            print(
                f"✓ {id_}: task={'✓' if task_ok else '✗'}, proposal={'✓' if prop_ok else '✗'}"
            )
            success += 1
        else:
            print(f"✗ {id_}: 未找到任何文件")
            failed += 1

    print(f"\n完成！成功: {success}, 失败: {failed}")


def main():
    print("FellouBench 任务文件移动工具")
    print("-" * 40)

    ids = get_ids_from_input()
    if not ids:
        print("没有输入有效的ID")
        return

    print(f"\n找到 {len(ids)} 个ID，开始移动...")
    move_ids_to_trash(TaskDataset(), ids)


if __name__ == "__main__":
    main()
