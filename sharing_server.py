"""
Standalone HTTP server for hosting and sharing backtest replay HTML files.

Run locally (port 8080):
    python sharing_server.py

Run on custom port:
    python sharing_server.py 9090

Once running, configure the URL in the app's GUI under Share → Configure Server.
Or set it directly: http://localhost:8080

For public sharing, deploy this server on a VPS / cloud platform and configure
the public URL in the app. Alternatively, use a tunnel like ngrok:
    ngrok http 8080
"""

import os
import sys
import json
import uuid
import re
import html as htmlmod
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from urllib.parse import urlparse

STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_replays")
os.makedirs(STORAGE_DIR, exist_ok=True)
META_FILE = os.path.join(STORAGE_DIR, "_index.json")

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


def _load_index():
    try:
        with open(META_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_index(index):
    with open(META_FILE, "w") as f:
        json.dump(index, f, indent=2)


class SharingHandler(BaseHTTPRequestHandler):

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, mime="text/html"):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)
        else:
            self._html("<h1>404 Not Found</h1>", 404)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/" or path == "":
            self._serve_index()
        elif path.startswith("/view/"):
            fid = path[len("/view/"):]
            fpath = os.path.join(STORAGE_DIR, fid)
            if os.path.isfile(fpath) and fid.endswith(".html"):
                self._file(fpath)
            else:
                self._html("<h1>404 Replay Not Found</h1>", 404)
        elif path == "/api/list":
            self._json({"ok": True, "replays": _load_index()})
        elif path.startswith("/api/share/"):
            sid = path[len("/api/share/"):]
            idx = _load_index()
            entry = next((e for e in idx if e["id"] == sid), None)
            if entry:
                self._json({"ok": True, "entry": entry})
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        else:
            self._html("404", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/upload":
            self._handle_upload()
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _handle_upload(self):
        ct = self.headers.get("Content-Type", "")
        cl = int(self.headers.get("Content-Length", 0))
        if "multipart/form-data" not in ct:
            self._json({"ok": False, "error": "Expected multipart/form-data"}, 400)
            return

        body = self.rfile.read(cl)
        boundary = ct.split("boundary=")[1].strip().encode()
        parts = body.split(b"--" + boundary)

        file_data = None
        symbol = timeframe = name = ""

        for part in parts:
            if b"Content-Disposition" not in part:
                continue
            hdr_end = part.find(b"\r\n\r\n")
            if hdr_end == -1:
                continue
            hdrs = part[:hdr_end].decode("utf-8", errors="replace")
            data = part[hdr_end + 4:].rstrip(b"\r\n-")
            if 'name="file"' in hdrs:
                file_data = data
            elif 'name="symbol"' in hdrs:
                symbol = data.decode("utf-8", errors="replace").strip()
            elif 'name="timeframe"' in hdrs:
                timeframe = data.decode("utf-8", errors="replace").strip()
            elif 'name="name"' in hdrs:
                name = data.decode("utf-8", errors="replace").strip()

        if not file_data:
            self._json({"ok": False, "error": "No file data received"}, 400)
            return

        uid = uuid.uuid4().hex[:12]
        safe = re.sub(r"[^\w\s-]", "", name or "replay")
        safe = re.sub(r"\s+", "_", safe.strip())[:40]
        fname = f"replay_{uid}_{safe}.html"
        fpath = os.path.join(STORAGE_DIR, fname)

        with open(fpath, "wb") as f:
            f.write(file_data)

        idx = _load_index()
        entry = {
            "id": uid,
            "filename": fname,
            "symbol": symbol,
            "timeframe": timeframe,
            "name": name or f"{symbol} {timeframe}".strip(),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "size": len(file_data),
        }
        idx.insert(0, entry)
        _save_index(idx)

        self._json({"ok": True, "url": f"/view/{fname}", "id": uid})

    def _serve_index(self):
        idx = _load_index()
        rows = ""
        for e in idx:
            safe_name = htmlmod.escape(e.get("name", "Unnamed"))
            safe_sym = htmlmod.escape(e.get("symbol", "?"))
            safe_tf = htmlmod.escape(e.get("timeframe", "?"))
            safe_date = htmlmod.escape(str(e.get("uploaded_at", ""))[:10])
            rows += f"""
            <div class="e" onclick="location='/view/{htmlmod.escape(e["filename"])}'">
              <span class="t">{safe_name}</span>
              <span class="m">{safe_sym}  ·  {safe_tf}  ·  {safe_date}</span>
            </div>"""
        if not rows:
            rows = '<p class="e" style="color:#555;text-align:center;padding:40px;cursor:default">No shared replays yet.</p>'

        self._html(f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Replays</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0D0D0D;color:#EEE;font-family:system-ui,sans-serif;min-height:100vh}}
.b{{background:#111;border-bottom:1px solid #1E1E1E;padding:20px 40px}}
.b h1{{color:#00ADB5;font-size:20px;font-weight:700}}
.b p{{color:#555;font-size:12px;margin-top:4px}}
.c{{max-width:720px;margin:0 auto;padding:30px 20px}}
.e{{background:#141414;border:1px solid #1E1E1E;border-radius:6px;padding:14px 18px;margin-bottom:10px;cursor:pointer;transition:border-color .15s}}
.e:hover{{border-color:#00ADB5}}
.e .t{{color:#EEE;font-size:14px;font-weight:600;display:block}}
.e .m{{color:#555;font-size:11px;margin-top:3px;display:block}}
</style></head>
<body>
<div class="b"><h1>◈ Backtest Replays</h1><p>Shared by Trading Bot</p></div>
<div class="c">{rows}</div>
</body>
</html>""")


def main():
    server = HTTPServer(("0.0.0.0", PORT), SharingHandler)
    print(f"[OK] Sharing server running at http://0.0.0.0:{PORT}")
    print(f"     Storage: {STORAGE_DIR}")
    print(f"     Upload:  POST http://localhost:{PORT}/upload")
    print(f"     View:    http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
