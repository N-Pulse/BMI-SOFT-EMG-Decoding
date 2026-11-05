# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from ..domain import DownloadSpec, Registry


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class Downloader:
    _registry: Registry = field(default_factory=Registry)

    def run(self, spec: DownloadSpec) -> None:
        # 1. Select strategy based on source
        source = spec.source
        strategy = self._registry.get(source)

        # 2. Check the source is reachable, then fetch from it
        strategy.validate(spec)
        strategy.fetch(spec)
