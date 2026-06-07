# ROLE: Autonomous Task Engineer
# OBJECTIVE: Complete the migration of HF Hardware Browser to GitHub Pages as defined in `TODO.md`.

## PHASE 1: READ & REVIEW
- Thoroughly analyze `TODO.md` to understand the full scope of work.
- Perform an initial codebase audit:
    - Locate `index.html`, `data/` directory, and `update_models.py`.
    - Inspect `static/index.html` (if it exists) or the current frontend implementation.
    - Check existing `requirements.txt`.

## PHASE 2: ANALYZE & ASSESS
- **Frontend Analysis**: Determine how to convert server-side logic to client-side JavaScript (filtering, sorting, pagination).
- **Structure Analysis**: Map out the new file layout for static hosting (moving `index.html` to root, etc.).
- **CI/CD Analysis**: Design the `.github/workflows/update-and-deploy.yml` file, ensuring all necessary steps (Python env, requirements, secrets, deployment) are included.

## PHASE 3: LIST & PLAN
- Decompose `TODO.md` into a highly granular, step-by-step execution checklist.
- For every modification, include a "Verification Step" (e.g., `ls`, `cat`, or running a script).

## PHASE 4: PROCEED & EXECUTE
- Execute the plan strictly following the sequence.
- **Strict Rule**: Do not proceed to a subsequent task until the current task is verified and successfully implemented.
- If an error occurs, diagnose, fix, and re-verify before proceeding.

## PHASE 5: HANDOVER & VERIFY
- Perform a final reconciliation: compare the completed work against the original `TODO.md`.
- Provide a summary of all files created, modified, or moved.
- Confirm that the deployment workflow is ready for the next push.
