from configs import (
    ASSETS_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
)

# =============================================================================
# 主要配置
# =============================================================================

MODULE_NAME = "construct"

# =============================================================================
# 路径与派生配置
# =============================================================================

MODULE_LOGS_DIR = LOGS_DIR / MODULE_NAME
MODULE_OUTPUT_DIR = OUTPUT_DIR / MODULE_NAME
MODULE_ASSETS_DIR = ASSETS_DIR / MODULE_NAME
