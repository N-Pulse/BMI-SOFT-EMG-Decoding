# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from tqdm import tqdm
from dataclasses import dataclass
from pathlib import PurePosixPath
import xml.etree.ElementTree as ET
from requests.auth import HTTPBasicAuth
from urllib.parse import quote, unquote, urlparse

from ...domain import DownloadStrategy, Registry
from .archive_specs import ArchiveSpecs

# ================================================================
# 1. Section: Constants
# ================================================================
CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = (10, 300)


# ================================================================
# 2. Section: Class definition
# ================================================================
@Registry.register(ArchiveSpecs.source)
@dataclass
class ArchiveStrategy(DownloadStrategy[ArchiveSpecs]):
    def validate(self, spec: ArchiveSpecs) -> None:
        response = requests.request(
            method="PROPFIND",
            url=spec.dav_root,
            headers=spec.header(),
            data=spec.body,
            auth=_auth(spec),
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 207:
            raise ValueError(
                f"Unexpected status code {response.status_code} from {spec.dav_root!r}, "
                "expected 207"
            )

    def fetch(self, spec: ArchiveSpecs) -> None:
        auth = _auth(spec)

        # 1. Makes sure we have a folder to download to
        os.makedirs(spec.dest, exist_ok=True)

        # 2. Gets everyting inside the remote folder
        xml_text = _propfind(spec, auth, depth="infinity")
        items = _parse_propfind(spec, xml_text)

        # 3. Iterates over every file
        ignore = set(spec.extra.get("ignore", []))
        for item in tqdm(items, desc="Downloading files", unit="file"):
            # 3.1 Get the relative path
            rel_path = _remote_relative_path(spec.dav_user, item["href"])

            # 3.2 Skip the folder itself (propfin returns the folder as the first entry too)
            if rel_path == spec.url:
                continue

            rel_parts = PurePosixPath(rel_path).parts
            if any(part in ignore for part in rel_parts):
                continue

            # 3.3 Makes sub-folders if needed and downloads the files
            remote_folder = _extract_path_of_interest(rel_path, spec.dav_url)
            local_path = os.path.join(spec.dest, remote_folder)
            if item["is_dir"]:
                os.makedirs(local_path, exist_ok=True)
                continue

            # Skip files that were already downloaded in a previous run
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                continue

            remote_url = f"{spec.base_url}{quote(item['href'])}"
            _download_file(auth, remote_url, local_path)


# ──────────────────────────────────────────────────────
# 2.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _auth(spec: ArchiveSpecs) -> HTTPBasicAuth:
    return HTTPBasicAuth(spec.username, spec.password)


def _propfind(spec: ArchiveSpecs, auth: HTTPBasicAuth, depth: int | str = 1) -> str:
    response = requests.request(
        "PROPFIND",
        spec.dav_url,
        headers=spec.header(depth),
        data=spec.body,
        auth=auth,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def _parse_propfind(spec: ArchiveSpecs, xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    items = []

    for response in root.findall("d:response", spec.namespaces):
        href = response.find("d:href", spec.namespaces)
        propstat = response.find("d:propstat", spec.namespaces)
        if href is None or propstat is None:
            continue

        prop = propstat.find("d:prop", spec.namespaces)
        if prop is None:
            continue

        resourcetype = prop.find("d:resourcetype", spec.namespaces)
        is_dir = (
            resourcetype is not None
            and resourcetype.find("d:collection", spec.namespaces) is not None
        )

        items.append({"href": unquote(str(href.text)), "is_dir": is_dir})

    return items


def _remote_relative_path(dav_user: str, href: str) -> str:
    path = urlparse(href).path

    prefix = f"/remote.php/dav/files/{dav_user}"
    if path == prefix:
        return ""
    if path.startswith(f"{prefix}/"):
        return path[len(prefix) + 1 :].strip("/")

    raise ValueError(f"Unexpected href {href!r}, expected it under {prefix!r}")


def _extract_path_of_interest(rel_path: str, server_path: str) -> str:
    rel_parts = [part for part in rel_path.split("/") if part]
    server_parts = [part for part in server_path.split("/") if part]

    if not server_parts:
        return "/".join(rel_parts)

    # Try to find the deepest matching part of server_path inside rel_path.
    # It prefers longer contiguous matches first.
    for end in range(len(server_parts), 0, -1):
        for length in range(end, 0, -1):
            candidate = server_parts[end - length : end]

            for start in range(len(rel_parts) - length, -1, -1):
                if rel_parts[start : start + length] == candidate:
                    return "/".join(rel_parts[start + length :])

    raise ValueError(
        f"No folder of server_path {server_path!r} was found in rel_path {rel_path!r}"
    )


def _download_file(auth: HTTPBasicAuth, remote_url: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with requests.get(
        remote_url, auth=auth, stream=True, timeout=REQUEST_TIMEOUT
    ) as response:
        response.raise_for_status()
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
