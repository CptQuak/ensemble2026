# Project Guidelines

- **Dependency Management**: Use `uv` for managing dependencies (e.g., `uv add <package>`, `uv run <script>`).
- **Logging**: Use `loguru` for all logging instead of the standard `print` or `logging` modules. Ensure logs provide sufficient context.
- **Module Structure**: Keep the logic modular inside the `src/` directory. Use `main.py` purely as an entry point.
- **Artifacts**: Store all generated files, such as prediction CSVs, plots, and log files, in the `artifacts/` folder (preferably in timestamped subfolders or clearly named files).
