# Deploying Harvest Atlas

The app is a plain Flask server. It needs a host that runs a process 24/7 — the
public URL keeps working after you shut your laptop down, because the code runs
on the host, not on your machine.

| File | Purpose |
| --- | --- |
| `wsgi.py` | Production entry point (`wsgi:app`). Imports the app without starting the terminal chat. |
| `render.yaml` | Render Blueprint: free web service, Python 3.13, gunicorn. |
| `Procfile` | Same start command for Railway / Heroku-style hosts. |
| `Dockerfile` | Portable image for Fly.io, Google Cloud Run, Hugging Face Spaces, a VPS. |
| `requirements.txt` | Now includes gunicorn for the production server. |

`python main.py` still works locally exactly as before.

---

## Option A — Render (recommended, free)

**1. Put the code on GitHub** (once, in this folder):

```powershell
git init
git add .
git commit -m "Harvest Atlas: deployable Flask app"
# create an empty repo on github.com named harvest-atlas, then:
git remote add origin https://github.com/<your-username>/harvest-atlas.git
git branch -M main
git push -u origin main
```

`.env` is git-ignored on purpose — your keys must not go into git.

**2. Create the Render service**

1. Sign up at <https://render.com> (GitHub sign-in is fastest).
2. **New +** → **Blueprint** → pick the repo you just pushed.
3. Render reads `render.yaml` and asks for three secrets:
   - `GROQ_API_KEY` — from <https://console.groq.com/keys>
   - `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — optional, only for the
     AppEEARS satellite samples. Leave empty and the app reports that gap.
4. Click **Apply**. First build takes ~2–3 minutes.

You get a public URL like `https://harvest-atlas.onrender.com`.

**Free-tier behaviour:** the instance sleeps after ~15 minutes with no visitors,
so the first visit afterwards takes ~50 seconds to wake. The site is never
deleted — it just naps. Upgrade to Starter ($7/mo) if you need it always warm.

---

## Option B — Railway

1. `npm i -g @railway/cli`, then `railway login` (opens your browser).
2. In this folder: `railway init` → `railway up`.
3. Add the same three env vars under **Variables**, then **Settings → Generate
   Domain**.

Railway includes a small free credit and does not sleep.

---

## Option C — Docker anywhere

```powershell
docker build -t harvest-atlas .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key harvest-atlas
```

Then open <http://localhost:7860>. The same image deploys to Fly.io, Cloud Run
or a VPS. On Hugging Face Spaces the default `PORT=7860` already matches what
Spaces expects.

---

## Environment variables

| Variable | Required | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | AI scenarios, stress check, plant diagnosis. |
| `GROQ_API_KEY_2/_3`, `GROQ_API_KEYS` | no | Fallbacks when a key is rate-limited. |
| `GROQ_MODEL` | no | Default `openai/gpt-oss-120b`. |
| `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` | no | AppEEARS MODIS/SMAP point samples. |

---

## Operational notes

- **Keep one worker.** The selected field, environment cache and AppEEARS task
  live in module memory, so `--workers 2` would make the browser and the server
  disagree about which field is selected. `--threads 8` gives concurrency
  without that split.
- **Long requests.** `/select` waits on NASA POWER, soil and terrain (up to
  ~70 s), which is why gunicorn runs with `--timeout 180`.
- **No database.** Everything is in-memory: a restart clears the selected field
  and cached lookups, nothing is lost permanently.
- **Open access.** As requested, there is no login. Anyone with the URL can call
  `/recommend`, `/stress` and `/disease`, which spend your Groq credits. If that
  becomes a problem, say the word and a per-IP rate limit can be added to those
  three routes only (the map and NASA data stay unmetered).
