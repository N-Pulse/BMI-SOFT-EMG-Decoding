# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from urllib.parse import quote, unquote, urlparse
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET

from .archive_vars import (
    BODY,
    NAMESPACES,
    header,
)



# ================================================================
# 1. Section: Path/Content Functions
# ================================================================
def dav_url(dav_root: str, remote_path: str) -> str:
    remote_path = remote_path.strip("/")
    if remote_path:
        return f"{dav_root}/{quote(remote_path)}"
    return dav_root

def propfind(url: str, auth: HTTPBasicAuth, depth: int = 1) -> str:
    r = requests.request(
        "PROPFIND",
        url,
        headers=header(depth),
        data=BODY,
        auth=auth,
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

def remote_relative_path(dav_user: str, href: str) -> str:
    parsed = urlparse(href)
    path = parsed.path

    prefix = f"/remote.php/dav/files/{dav_user}/"
    if path.startswith(prefix):
        return path[len(prefix):].strip("/")

    prefix_no_slash = f"/remote.php/dav/files/{dav_user}"
    if path == prefix_no_slash:
        return ""

    raise ValueError(f"Unexpected href: {href}")



# ================================================================
# 2. Section: Download Functions
# ================================================================
def download_file(auth: HTTPBasicAuth, dav_root: str, remote_path: str, local_path: str):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = dav_url(dav_root, remote_path)

    with requests.get(url, auth=auth, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded file: {remote_path} -> {local_path}")
