# Download

Layers: `service` (`Downloader`) → `domain` (specs, strategies, registry) → `adapters` (one per source). `Registry` subclasses the shared `bmiemg.utils.Registry`.

To add a source, create `adapters/<source>/` with `<source>_specs.py` (a `DownloadSpec` dataclass with `source: ClassVar[str] = "<source-name>"`) and `<source>_strategy.py` (a `DownloadStrategy[<Source>Specs]` decorated with `@Registry.register(<Source>Specs.source)`), re-export both from the folder's and `adapters/__init__.py` so the strategy registers on import, and add an entry script in `scripts/00_download/`.

## Zenodo source

Downloads one file of a Zenodo record, checks its size and MD5, then optionally extracts it as a zip. A valid file already on disk is not downloaded again. No credentials needed. Dataset values (record and file) live in the entry script, not the adapter.

| Parameter | Default | Meaning |
|---|---|---|
| `dest` | required | Local folder for the downloaded file |
| `record_id` | required | Zenodo record |
| `filename` | required | File of the record to fetch |
| `extract` | `True` | Unzip the file after download |
| `extract_to` | `dest/<file stem>` | Folder to extract into |

`scripts/00_download/download_epn612.py` fetches EMG-EPN-612 (record `4421500`, `EMG-EPN612 Dataset.zip`) into `data/raw/emg_epn_612`, the folder prune reads:

```bash
poetry run python scripts/00_download/download_epn612.py
```

## Archive source

Pulls files from EPFL's WebDAV archive (`make-archives.epfl.ch`).

1. Install dependencies: `poetry install`.
2. Copy `.env.example` to `.env` and fill in your credentials (never commit them):

   ```env
   MAKER_USERNAME=<your username>
   MAKER_APP_PASSWORD=<your app password>
   MAKER_DAV_USER=<your dav user>
   ```

3. Set the remote `url` and local `dest` in `scripts/00_download/download_archive.py`, then run it (it loads `.env` automatically):

   ```bash
   poetry run python scripts/00_download/download_archive.py
   ```

### Getting the credentials

- **`MAKER_USERNAME`** — Is your EPFL username
- **`MAKER_APP_PASSWORD`** — Go to Profile Settings (top right) on the Archive MAKE Drive > Security > Create New App Password
- **`MAKER_DAV_USER`** — Go to Files Settings (bottom left) on the Archive MAKE drive > WebDAV > Copy the URL and past the code in the URL that comes after https://make-archives.epfl.ch/remote.php/dav/files/
