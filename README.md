## cap-bench-pipeline

本仓库用于 CAP Benchmark 的流水线开发，目前包含：

- `construct`：Benchmark 任务构建流水线
- `evaluate`：Benchmark 结果评估流水线

### 目录结构

```text
src/
  construct/          构建模块（自包含：datasets/utils/prompts）
  evaluate/           评估模块
  common/             预留公共包（当前为空）
  main_construct.py   任务构建入口
```