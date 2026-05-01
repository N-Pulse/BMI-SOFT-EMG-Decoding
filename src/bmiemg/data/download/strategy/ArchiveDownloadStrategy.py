# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlparse
from tqdm import tqdm
from requests import Response
from requests.auth import HTTPBasicAuth
from pathlib import Path

from .DownloadStrategy import DownloadStrategy
from ..Credentials import Credentials
from ..Request import Request
from .archive_utils import (
    dav_url,
    propfind,
    parse_propfind,
    remote_relative_path,
    download_file
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
        self._dav_root = f"{BASE_URL}/remote.php/dav/files/{credentials.key}"
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
        os.makedirs(request.out_path, exist_ok=True)

        server_path = dav_url(self._dav_root, request.url)
        xml_text = propfind(server_path, self._auth, depth=3)
        items = parse_propfind(xml_text)

        for item in tqdm(items):
            rel_path = remote_relative_path(self._dav_user, item["href"])

            # Skip the folder itself (PROPFIND returns the folder as the first entry too)
            if rel_path == request.url:
                continue

            if item["is_dir"]:
                remote_folder = extract_path_of_interest(rel_path, server_path)
                folder = os.path.join(request.out_path, remote_folder)
                os.makedirs(folder, exist_ok=True)
            else:
                remote_folder = extract_path_of_interest(rel_path, server_path)
                local_path = os.path.join(request.out_path, remote_folder)
                download_file(self._auth, self._dav_root, rel_path, local_path)

        return request.out_path

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

def normalize_path(path: str) -> str:
    # If path is a full URL, extract only the path part
    parsed = urlparse(path)
    path = parsed.path if parsed.scheme else path

    # Decode %20 etc.
    path = unquote(path)

    return path.strip("/")
