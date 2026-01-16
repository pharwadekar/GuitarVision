# GuitarVision

Lightweight prototype for recognizing guitar chords with computer vision and turning the output into a mini rhythm game.

## Quick start
1. Create a virtual environment and install deps:
	- `python -m venv .venv`
	- `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
	- `pip install -r backend/requirements.txt`
2. Run the API: `uvicorn backend.app.main:app --reload`
3. Visit `http://localhost:8000/health` to confirm it is running.

## Repo layout
- backend/app: FastAPI service (currently just health check)
- ml/src: Stubs for data capture, feature engineering, training, evaluation
- ml/data: Placeholder folders for raw and processed datasets
- docs/version-control.md: Short note on the git workflow being used

## Working style
- Keep branches short-lived and merge through pull requests.
- Commit small, focused changes with clear messages.
- Add tests or simple scripts alongside new functionality when possible.
