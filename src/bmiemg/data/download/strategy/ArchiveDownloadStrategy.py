# ================================================================
# 0. Section: IMPORTS
# ================================================================
import requests

from dataclasses import dataclass, field
from requests import Response
from requests.auth import HTTPBasicAuth

from .DownloadStrategy import DownloadStrategy
from ..Credentials import Credentials
from ..Request import Request
from .archive_utils import (
    dav_url
)
from .archive_vars import (
    BODY,
    HEADER,
    BASE_URL
)



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ArchiveDownloadStrategy(DownloadStrategy):
    _auth: HTTPBasicAuth = field(init=False)
    _dav_root: str = ""
    _dav_user: str = ""

    def __post_init__(self):
        self._auth = HTTPBasicAuth("", "")

    def authenticate(self, credentials: Credentials) -> Response:
        self._dav_user = credentials.key
        self._dav_root = dav_url(BASE_URL, credentials.key)
        self._auth = HTTPBasicAuth(credentials.username, credentials.password)

        response = requests.request(
            method = "PROPFIND",
            url = self._dav_root,
            headers = HEADER,
            data = BODY,
            auth=self._auth,
            allow_redirects=False,
        )

        print("status:", response.status_code)
        print("final_url:", response.url)

        return response

    def download(self, request: Request) -> Path:
        return super().download(request)

    def cleanup(self) -> None:
        return super().cleanup()



def extract_path_of_interest(rel_path: str, server_path: str) -> str:
    def split_path(path: str) -> list[str]:
        return [part for part in path.strip("/").split("/") if part]

    rel_parts = split_path(rel_path)
    server_parts = split_path(server_path)

    if not server_parts:
        return "/".join(rel_parts)

    # Try to find the deepest matching part of server_path inside rel_path.
    # It prefers longer contiguous matches first.
    for end in range(len(server_parts), 0, -1):
        for length in range(end, 0, -1):
            candidate = server_parts[end - length:end]

            for start in range(len(rel_parts) - length, -1, -1):
                if rel_parts[start:start + length] == candidate:
                    return "/".join(rel_parts[start + length:])

    raise ValueError(
        f"No folder from server_path was found in rel_path:\n"
        f"rel_path: {rel_path}\n"
        f"server_path: {server_path}"
    )
