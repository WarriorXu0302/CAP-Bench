from configs import (
    ASSETS_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
)

# =============================================================================
# Main Configuration
# =============================================================================

MODULE_NAME = "construct"

# =============================================================================
# Paths and Derived Configuration
# =============================================================================

MODULE_LOGS_DIR = LOGS_DIR / MODULE_NAME
MODULE_OUTPUT_DIR = OUTPUT_DIR / MODULE_NAME
MODULE_ASSETS_DIR = ASSETS_DIR / MODULE_NAME
