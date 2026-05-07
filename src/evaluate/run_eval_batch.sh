#!/bin/bash

# 1. 定义路径变量
ANSWERS_DIR="/home/yongtong/seu/cap-eval/answers/cap"
JSON_BASE_DIR="/home/yongtong/seu/cap-eval/eval/tasks_final_en"

# 设置最大并发度
MAX_JOBS=7

# 当前正在运行的后台任务数量
current_jobs=0

# 检查 answers 目录是否存在
if [ ! -d "$ANSWERS_DIR" ]; then
    echo "❌ 错误: 找不到目录 $ANSWERS_DIR"
    exit 1
fi

echo "🚀 开始遍历任务，设置并发度为: $MAX_JOBS"

# 2. 遍历该路径下的所有 task-* 文件夹
for dir in "$ANSWERS_DIR"/task-*/; do
    # 提取纯文件夹名字 (例如 task-0aae15)
    task_id=$(basename "$dir")
    
    # 拼接对应的 JSON 文件路径
    json_path="$JSON_BASE_DIR/${task_id}.json"

    # (可选) 检查对应的 JSON 文件是否存在，防止 Python 报错退出
    if [ ! -f "$json_path" ]; then
        echo "⚠️  跳过: 找不到对应的 JSON 文件 $json_path"
        continue
    fi

    echo "🏃 启动任务: $task_id"

    # 3. 运行 Python 脚本并放入后台执行 (&)
    python generate_eval.py --json_path "$json_path" &

    # 后台任务数 +1
    ((current_jobs++))

    # 4. 控制并发度：如果当前运行的任务数达到了 5 个，就等待其中任意一个完成
    if (( current_jobs >= MAX_JOBS )); then
        wait -n          # 等待最快结束的一个后台任务
        ((current_jobs--)) # 任务结束后，腾出一个空位，计数减 1
    fi
done

# 循环结束后，等待所有剩余的后台任务全部完成
wait
echo "✅ 所有任务执行完毕！"