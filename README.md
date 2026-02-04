# AdChecker

## Overview
AdChecker analyzes ad copy and optional ad images for Google Ads compliance. It runs a hard‑rules check, then AI compliance analysis, then marketing recommendations. The backend and the web UI are demo-only to visualize the workflow; the core logic lives in the analysis flow.

## What we did so far
1. Implemented hard‑rules checks for ad length and banned words.
2. Added AI compliance analysis using Google Gemini (text and optional image).
3. Added AI marketing recommendations.
4. Added guideline chunking to reduce prompt size and improve coverage.
5. Added image normalization to reduce resolution while preserving information.
6. Built a simple demo UI and FastAPI endpoint to show results visually.

## Tools to install
- Python 3.12+
- uv (https://github.com/astral-sh/uv) - fast Python package and project manager
- Project dependencies from pyproject.toml:
  - fastapi
  - google-generativeai
  - pillow
  - python-dotenv
  - python-multipart
  - requests
  - uvicorn

## Dependency management (uv)
This project uses **uv** for dependency management. All dependencies are tracked in `pyproject.toml` and `uv.lock`.

### Common commands:
```bash
# Sync dependencies (install from lockfile)
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Run a command in the uv environment
uv run python hybrid_checker.py
uv run uvicorn backend:app --reload

# Update dependencies
uv lock --upgrade
```

### Important notes:
- **Always use `uv add`** to add new packages, not `pip install`.
- `uv sync` will create/update `.venv` automatically.
- Do not mix pip and uv to avoid dependency conflicts.

## Setup steps
1. Install uv if you haven't already: `pip install uv` or follow https://github.com/astral-sh/uv
2. Sync dependencies: `uv sync`
3. Create a .env file with GEMINI_API_KEY.
4. Update inputs in these files if you want default data:
   - ad_copy.txt
   - business_info.txt
   - guidelines.txt
   - recommendation.txt

## How the workflow runs (step‑by‑step)
1. Load configuration from rules_config.json.
2. Load ad copy, business info, guidelines, and recommendations.
3. Optional: load ad image if it exists.
4. Step 1: Hard rules validation (length + banned words).
5. Step 1.5: If an image is provided, run image compliance analysis.
6. Step 2: Run compliance check against chunked guidelines.
7. Step 3: Generate marketing recommendations.
8. Aggregate results into a final PASS/FAIL outcome.

## Core logic: what the code is doing
### Hard rules (logic-based)
- Enforces max ad length.
- Flags banned words based on rules_config.json.

### Guideline chunking (LLM prompt reduction)
- Guidelines are split by paragraph boundaries (\n\n).
- Chunks are limited to ~1200 characters.
- Each chunk is analyzed separately by the LLM.
- Results are merged into a single compliance summary.

### Image normalization (resolution reduction without losing much info)
- Converts to RGB and flattens alpha.
- Resizes the image so the longest edge is 1024px.
- Uses LANCZOS resampling to preserve visual quality.

## Demo UI vs core logic
- Demo UI: web/index.html (visualizes results only).
- Demo backend: backend.py (API wrapper + data flow).
- Core analysis flow: the compliance and recommendation steps used by both the backend and the CLI.

## How to run
### CLI (core logic demo)
- Run hybrid_checker.py to execute the step‑by‑step analysis in the terminal.

### API + UI (demo)
- Start the FastAPI server in backend.py and open the UI in the browser.
- The UI is only for presentation and can be changed anytime.

## Files and roles
- backend.py: Demo API server + orchestration.
- hybrid_checker.py: CLI workflow (core flow mirror).
- web/index.html: Demo UI only.
- rules_config.json: Hard rules configuration.
- ad_copy.txt / business_info.txt / guidelines.txt / recommendation.txt: default inputs for quick testing.

## Notes
- If any chunk returns FAIL, the compliance step fails.
- If any chunk errors, the compliance step is marked ERROR.
- Overall status is FAIL if any step fails or errors.
