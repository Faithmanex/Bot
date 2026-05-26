"""
Client for sharing backtest replay HTML files to a remote sharing server.

Usage (in GUI):
    from currency.backtest_sharing import upload_replay, get_server_url, set_server_url
    share_url, error = upload_replay(html_path, symbol="EURUSD", timeframe="M5")
"""
import os
import json
import uuid
from urllib.request import Request, urlopen
from urllib.error import URLError

DEFAULT_SERVER_URL = "http://localhost:8080"
SETTINGS_KEY = "_sharing"


def _load_settings():
    from .settings import get_path
    path = get_path("currency/settings.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings):
    from .settings import get_path
    path = get_path("currency/settings.json")
    with open(path, "w") as f:
        json.dump(settings, f, indent=4)


def get_server_url():
    s = _load_settings()
    return s.get(SETTINGS_KEY, {}).get("server_url", DEFAULT_SERVER_URL)


def set_server_url(url):
    s = _load_settings()
    s.setdefault(SETTINGS_KEY, {})["server_url"] = url
    _save_settings(s)


def upload_replay(html_path, symbol="", timeframe="", name=""):
    """Upload an HTML replay file to the sharing server.

    Returns (share_url_or_None, error_or_None).
    """
    if not os.path.exists(html_path):
        return None, f"File not found: {html_path}"

    server_url = get_server_url()
    boundary = f"----Boundary{uuid.uuid4().hex}"

    with open(html_path, "rb") as f:
        file_data = f.read()

    filename = os.path.basename(html_path)

    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    )
    parts.append(b"Content-Type: text/html\r\n\r\n")
    parts.append(file_data)
    parts.append(f"\r\n--{boundary}\r\n".encode())
    for field_name, val in [
        ("symbol", symbol),
        ("timeframe", timeframe),
        ("name", name),
    ]:
        parts.append(
            f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n{val}\r\n'.encode()
        )
        parts.append(f"--{boundary}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)

    try:
        req = Request(
            f"{server_url.rstrip('/')}/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                full_url = f"{server_url.rstrip('/')}{result['url']}"
                return full_url, None
            return None, result.get("error", "Unknown server error")
    except URLError as e:
        return None, f"Cannot reach server at {server_url}: {e.reason}"
    except Exception as e:
        return None, str(e)


def test_connection(server_url=None):
    """Test connectivity to the sharing server by hitting its root endpoint."""
    url = (server_url or get_server_url()).rstrip("/")
    try:
        with urlopen(f"{url}/api/list", timeout=10) as resp:
            return resp.status == 200, None
    except Exception as e:
        return False, str(e)
