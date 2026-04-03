"""
download_folder(
    remote_folder="EPFL N-pulse (2)/Quality Management System/BMI/Old_format/sub-P008",
    local_folder="./sub-P008"
)
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

import xml.etree.ElementTree as ET

from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote, unquote, urlparse
from requests.auth import HTTPBasicAuth

ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".venv")

USERNAME = str(os.getenv("MAKER_USERNAME"))
PASSWORD = str(os.getenv("MAKER_APP_PASSWORD"))
DAV_USER = str(os.getenv("MAKER_DAV_USER"))

BASE_URL = "https://make-archives.epfl.ch"
AUTH = HTTPBasicAuth(USERNAME, PASSWORD)
DAV_ROOT = f"{BASE_URL}/remote.php/dav/files/{DAV_USER}"
NAMESPACES = {
    "d": "DAV:",
}



# ================================================================
# 1. Section: Main Functions
# ================================================================
def dav_url(remote_path: str) -> str:
    remote_path = remote_path.strip("/")
    if remote_path:
        return f"{DAV_ROOT}/{quote(remote_path)}"
    return DAV_ROOT

def propfind(remote_path: str, depth: int = 1):
    url = dav_url(remote_path)
    body = """<?xml version="1.0"?>
    <d:propfind xmlns:d="DAV:">
      <d:prop>
        <d:displayname />
        <d:resourcetype />
        <d:getcontentlength />
        <d:getlastmodified />
      </d:prop>
    </d:propfind>
    """

    r = requests.request(
        "PROPFIND",
        url,
        headers={
            "Depth": str(depth),
            "Content-Type": "application/xml",
        },
        data=body,
        auth=AUTH,
    )
    r.raise_for_status()
    return r.text

def parse_propfind(xml_text: str):
    root = ET.fromstring(xml_text)
    items = []

    for response in root.findall("d:response", NAMESPACES):
        href = response.find("d:href", NAMESPACES)
        propstat = response.find("d:propstat", NAMESPACES)
        if href is None or propstat is None:
            continue

        prop = propstat.find("d:prop", NAMESPACES)
        if prop is None:
            continue

        resourcetype = prop.find("d:resourcetype", NAMESPACES)
        is_dir = resourcetype is not None and resourcetype.find("d:collection", NAMESPACES) is not None

        items.append({
            "href": unquote(str(href.text)),
            "is_dir": is_dir,
        })

    return items

def remote_rel_path_from_href(href: str) -> str:
    parsed = urlparse(href)
    path = parsed.path

    prefix = f"/remote.php/dav/files/{DAV_USER}/"
    if path.startswith(prefix):
        return path[len(prefix):].strip("/")

    prefix_no_slash = f"/remote.php/dav/files/{DAV_USER}"
    if path == prefix_no_slash:
        return ""

    raise ValueError(f"Unexpected href: {href}")

def download_file(remote_path: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = dav_url(remote_path)

    with requests.get(url, auth=AUTH, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded file: {remote_path} -> {local_path}")

def download_folder(remote_folder: str, local_folder: str):
    os.makedirs(local_folder, exist_ok=True)

    xml_text = propfind(remote_folder, depth=1)
    items = parse_propfind(xml_text)

    remote_folder_clean = remote_folder.strip("/")

    for item in items:
        rel_path = remote_rel_path_from_href(item["href"])

        # Skip the folder itself (PROPFIND returns the folder as the first entry too)
        if rel_path == remote_folder_clean:
            continue

        if item["is_dir"]:
            sub_local = os.path.join(local_folder, os.path.basename(rel_path))
            download_folder(rel_path, sub_local)
        else:
            filename = os.path.basename(rel_path)
            local_path = os.path.join(local_folder, filename)
            download_file(rel_path, local_path)

if __name__ == '__main__':
    #https://make-archives.epfl.ch/apps/files/files/222832?dir=/EPFL%20N-pulse/Quality%20Management%20System/BMI/bids/2025-11-12/sub-05/ses-01
    download_folder(
        remote_folder="EPFL N-pulse/Quality Management System/BMI/bids/2025-11-12/sub-05/ses-01",
        local_folder=".data/bids/sub-05"
    )
