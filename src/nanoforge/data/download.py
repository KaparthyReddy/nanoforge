from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(dest="data/tinyshakespeare.txt", url=TINY_SHAKESPEARE_URL, force=False):
    path = Path(dest)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())
    return path
