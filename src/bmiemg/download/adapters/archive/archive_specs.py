# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from urllib.parse import quote
from dataclasses import dataclass, field

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class ArchiveSpecs(DownloadSpec):
    source: ClassVar[str] = "archive-raw"

    base_url: ClassVar[str] = "https://make-archives.epfl.ch"
    body: ClassVar[str] = """<?xml version="1.0"?>
    <d:propfind xmlns:d="DAV:">
      <d:prop>
        <d:displayname />
        <d:resourcetype />
        <d:getcontentlength />
        <d:getlastmodified />
      </d:prop>
    </d:propfind>
    """
    namespaces: ClassVar[dict[str, str]] = {"d": "DAV:"}

    url: str
    username: str
    password: str
    dav_user: str
    filename: str | None = None
    extra: dict = field(default_factory=dict)

    dav_root: str = field(init=False)
    dav_url: str = field(init=False)

    def __post_init__(self) -> None:
        self.dav_root = f"{self.base_url}/remote.php/dav/files/{self.dav_user}"

        remote_path = self.url.strip("/")
        if remote_path:
            self.dav_url = f"{self.dav_root}/{quote(remote_path)}"
        else:
            self.dav_url = self.dav_root

    def header(self, depth: int | str = 1) -> dict[str, str]:
        """PROPFIND headers for the given depth."""
        return {
            "Depth": str(depth),
            "Content-Type": "application/xml",
        }
