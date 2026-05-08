# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from dataclasses import dataclass, field
from tqdm import tqdm
from requests import Response
from requests.auth import HTTPBasicAuth
from pathlib import Path, PurePosixPath

from .DownloadStrategy import DownloadStrategy
from ..Credentials import Credentials
from ..Request import Request
from .archive_utils import (
    dav_url,
    propfind,
    parse_propfind,
    remote_relative_path,
    download_file,
    extract_path_of_interest
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
        # 1. Makes sure we have a folder to download to
        os.makedirs(request.out_path, exist_ok=True)

        # 2. Gets everyting inside the remote folder
        server_path = dav_url(self._dav_root, request.url)
        xml_text = propfind(server_path, self._auth, depth="infinity")
        items = parse_propfind(xml_text)

        # 3. Iterates over every file
        ignore = set(request.extra.get("ignore", []))
        for item in tqdm(items, desc="Downloading files", unit="file"):
            # 3.1 Get the relative path
            rel_path = remote_relative_path(self._dav_user, item["href"])

            # 3.2 Skip the folder itself (propfin returns the folder as the first entry too)
            if rel_path == request.url:
                continue

            rel_parts = PurePosixPath(rel_path).parts
            if any(part in ignore for part in rel_parts):
                continue

            # 3.3 Makes sub-folders if needed and downloads the files
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
