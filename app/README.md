# AutoVUS demo

Two processes: the pages (Next.js, `web/`) and the Python side (FastAPI, `api/`) that reads the benchmark and computes live.

```bash
# from the repo root, two terminals
conda run -n autovus uvicorn main:app --app-dir app/api --port 8765 --reload
cd app/web && npm install && npm run dev        # http://localhost:3000 (PORT=3100 npm run dev for another port)
```

`AUTOVUS_API` changes where the web app looks for the Python side (default `http://127.0.0.1:8765`).

Pages: Intro (`/`), Rankings, Cases, Series, Speed. Shared state lives in the URL (`?b=tsb-ad`). Every page has a "?" popup (`web/lib/help.ts`).

## The tables

The Rankings, Cases and Speed pages read `app/data/<benchmark>/`, written by the export. It saves every 10 series, short series first, so the pages fill up while it runs. Keep it to a few cores:

```bash
taskset -c 0-3 nice -n 19 conda run -n autovus python app/export.py --phase 1 --workers 4   # every metric, every pair
taskset -c 0-3 nice -n 19 conda run -n autovus python app/export.py --phase 3              # FastVUS GPU time per series
taskset -c 0-3 nice -n 19 conda run -n autovus python app/export.py --phase 2 --subset 2   # the slow baselines, on a subset
```

Stop it with `pkill -f "export[.]py"`.
