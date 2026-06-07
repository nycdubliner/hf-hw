# TODO: Migrate HF Hardware Browser to GitHub Pages

## 1. Refactor for Static Hosting
- [ ] **Move Frontend logic**: Update `static/index.html` to:
    - [ ] Fetch `data/gpus.json` and `data/models.json` via relative paths.
    - [ ] Implement all filtering (GPU VRAM, context, search) and sorting (trending, downloads, etc.) in JavaScript.
    - [ ] Implement client-side pagination to handle the model list.
- [ ] **Restructure File Layout**: 
    - [ ] Move `index.html` to the root directory.
    - [ ] Ensure `data/` directory is correctly referenced by the updated `index.html`.

## 2. Automate with GitHub Actions
- [ ] **Create Workflow File**: Create `.github/workflows/update-and-deploy.yml`.
- [ ] **Define Schedule**: Set `on: schedule: - cron: '0 * * * *'` (hourly).
- [ ] **Define Triggers**: Set `on: push: branches: [main]`.
- [ ] **Implement Build Job**:
    - [ ] Setup Python environment.
    - [ ] Install `requirements.txt`.
    - [ ] Run `python3 update_models.py` (using `HF_TOKEN` from GitHub Secrets).
- [ ] **Implement Deploy Job**:
    - [ ] Use `peaceiris/actions-gh-pages` or similar to push `index.html` and `data/` to the `gh-pages` branch.

## 3. Verification
- [ ] **Local Test**: Run `python3 update_models.py` and open `index.html` locally to ensure the UI works with the generated JSON.
- [ ] **CI/CD Test**: Push to `main` and verify the GitHub Action runs successfully and the site appears at `https://nycdubliner.github.io/hf-hw/`.
