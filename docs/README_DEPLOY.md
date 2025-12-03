This `docs/` directory contains the demo HTML exported by `scripts/export_demo.py` and any other hosting assets.

Automation:
- The GitHub Actions workflow `.github/workflows/publish_docs.yml` automatically runs `scripts/export_demo.py` and deploys `./docs` to the `gh-pages` branch.
- To publish manually: run `scripts/release.sh` and then push the `docs/` contents to `gh-pages` or run the GitHub tooling.

Notes:
- Ensure `plotly` is installed for export; otherwise, `scripts/export_demo.py` will skip export.
