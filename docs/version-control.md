# Simple Git Workflow

- `main`: stable builds you would demo.
- `develop`: active work; branch from here.
- `feature/<something>`: short-lived branches for tasks.

Workflow in practice:
1. Pull `develop`.
2. `git checkout -b feature/<task>`.
3. Commit small changes with a short message (e.g., `feat: add chord capture stub`).
4. Push and open a pull request back to `develop`.
5. Merge when checks (lint/tests) pass and the change is reviewed.

If something is broken in production, branch from `main`, fix it, then merge into both `main` and `develop`.
