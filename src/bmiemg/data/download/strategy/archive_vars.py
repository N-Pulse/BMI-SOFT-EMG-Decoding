# ================================================================
# 0. Section: INPUTS
# ================================================================
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
BASE_URL = "https://make-archives.epfl.ch"
NAMESPACES = {
    "d": "DAV:",
}



# ================================================================
# 1. Section: Functions
# ================================================================
def header(depth: int | str = 1) -> dict:
    return {
        "Depth": str(depth),
        "Content-Type": "application/xml",
    }
