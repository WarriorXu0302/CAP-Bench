from pathlib import Path

# =============================================================================
# 主要配置
# =============================================================================

PROJECT_NAME = "cap-bench"

# =============================================================================
# 路径与派生配置
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
