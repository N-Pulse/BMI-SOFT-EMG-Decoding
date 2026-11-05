# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.utils import Registry as BaseRegistry

from .download_strategy import DownloadStrategy


# ================================================================
# 1. Section: Class definition
# ================================================================
class Registry(BaseRegistry[DownloadStrategy]): ...
