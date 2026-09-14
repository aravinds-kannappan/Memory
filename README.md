# Browser Memory

**Close tabs without losing your place.** Ask for anything you were working on, find it, and pick up where you left off.

This is a **local agent that starts when you log in** — not a Chrome extension. On install it registers as a login item (macOS LaunchAgent / Linux autostart), serves a small UI at `http://127.0.0.1:8765`, and keeps an encrypted memory database on disk.

## 60-second demo

```bash
cd agent
python3 -m pip install -e .
browser-memory serve --open --seed-demo
```

1. The UI opens automatically.
2. On **Ask**, run: *What was that supplier with the minimum order of 500?*
3. Open the Northwind result (historical snapshot + excerpt).
4. On **Workspaces**, **Resume** “Supplier comparison” and see the saved next step: *Tomorrow, compare delivery times.*

## What it remembers (kept separate)

| Kind | Example | Source |
|---|---|---|
| Observed activity | Viewed a supplier page last night | Browser history / live tabs |
| Saved content | “Minimum order: 500 units.” | Page snapshot |
| User intent | “Compare delivery times tomorrow.” | Explicit next-step field |

## Architecture

- **Local agent** (Python / FastAPI) — capture, retrieval, workspaces, UI
- **SQLite + FTS5** — pages, visits, snapshots, workspaces
- **Fernet field encryption** — snapshot text encrypted at rest
- **Local embeddings** — hashed semantic vectors (no cloud required)
- **Chrome history sync** — import visited URLs without an extension
- **Optional Chrome DevTools (port 9222)** — list / open / close live tabs
- **Login auto-start** — `browser-memory install`

```text
Login → agent starts → UI opens → sync history → ask / save / resume
```

## Install (Mac & Linux)

```bash
cd agent
python3 -m pip install -e .
browser-memory install          # start on login
browser-memory serve --open     # run now + open UI
```

Optional live-tab control:

```bash
../scripts/chrome-debug.sh      # Chrome with --remote-debugging-port=9222
```

## Three screens

1. **Ask** — natural-language recall with source excerpts  
2. **Workspaces** — Save my place, next steps, Resume  
3. **Memory** — capture status, exclusions, auto-start, delete  

## Privacy defaults

- Built-in exclusions for login/banking hosts  
- Sensitive query params stripped before storage  
- Forms stripped from extracted HTML  
- Excluded sites never enter storage or model inputs  
- “Delete all history” wipes the local DB  

## Develop

```bash
cd agent
python3 -m pip install -e ".[dev]"
pytest -q
MEMORY_DATA_DIR=/tmp/bm-test browser-memory serve --open --seed-demo
```

## Limits (honest MVP)

- Restores **links and context**, not unsaved forms or arbitrary web-app state  
- Live close/open needs Chrome remote debugging; otherwise Resume opens URLs in your browser  
- Embeddings are local and lightweight; swap in a stronger model later via the same store  
