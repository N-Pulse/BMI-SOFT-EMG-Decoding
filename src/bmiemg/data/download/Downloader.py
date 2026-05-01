# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from multiprocessing import AuthenticationError


from .Credentials import Credentials
from .Request import Request
from .strategy import DownloadStrategy



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Downloader:
    strategy: DownloadStrategy
    credentials: Credentials

    def run(self, request: Request):

        # 1. Always authenticate first
        response = self.strategy.authenticate(self.credentials)

        # 1.A Makes sure you can authenticate
        if response.status_code != 207:
            raise AuthenticationError(
                "The login was not sucessfull: "
                f"{response.status_code}"
            )

        # 2. Download the data
        self.strategy.download(request)
