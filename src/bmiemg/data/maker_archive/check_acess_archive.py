# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import requests

from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".venv")

USERNAME = str(os.getenv("MAKER_USERNAME"))
PASSWORD = str(os.getenv("MAKER_APP_PASSWORD"))
DAV_USER = str(os.getenv("MAKER_DAV_USER"))

BASE_URL = "https://make-archives.epfl.ch"
DAV_ROOT = f"{BASE_URL}/remote.php/dav/files/{DAV_USER}"

HEADER = {
    "Depth": "1",
    "Content-Type": "application/xml",
}
BODY = """<?xml version="1.0"?>
<d:propfind xmlns:d="DAV:">
  <d:prop>
    <d:displayname />
    <d:resourcetype />
    <d:getcontentlength />
    <d:getlastmodified />
  </d:prop>
</d:propfind>
"""



# ================================================================
# 1. Section: Mian Function
# ================================================================
def check_archive_access():
    r = requests.request(
        "PROPFIND",
        DAV_ROOT,
        headers=HEADER,
        data=BODY,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        allow_redirects=False,
    )

    print("status:", r.status_code)
    print("headers:", dict(r.headers))
    print("body:", r.text[:1000])
    print("final_url:", r.url)

#check_archive_access()
