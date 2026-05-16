from pathlib import Path

# =============================================================================
# Project configuration
# =============================================================================

PROJECT_NAME = "cap-bench-pipeline"

# =============================================================================
# Paths and derived configuration
# =============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
