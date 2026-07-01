"""Advanced dependency readiness checks for the learning repository.

This file intentionally avoids hard imports because many of these libraries are
optional and may not be available in every learner's environment.
"""

from __future__ import annotations

import importlib.util


ADVANCED_LIBS: list[tuple[str, str]] = [
    ("Numba", "numba"),
    ("CuPy", "cupy"),
    ("Bottleneck", "bottleneck"),
    ("Optuna", "optuna"),
    ("PyMC", "pymc"),
    ("Pyro", "pyro"),
    ("JAX", "jax"),
    ("LightGBM", "lightgbm"),
    ("XGBoost", "xgboost"),
    ("CatBoost", "catboost"),
    ("Plotly", "plotly"),
    ("Dash", "dash"),
    ("Bokeh", "bokeh"),
    ("NetworkX", "networkx"),
    ("GeoPandas", "geopandas"),
    ("Rasterio", "rasterio"),
    ("Cartopy", "cartopy"),
    ("Statsmodels", "statsmodels"),
    ("Pyro", "pyro"),
    ("Sage", "sageall") ,  # package alias varies; keep graceful fallback
    ("Prophet", "prophet"),
    ("Darts", "darts"),
    ("Neural Prophet", "neuralprophet"),
    ("OpenCV", "cv2"),
    ("Scikit-Image", "skimage"),
    ("LibROSA", "librosa"),
    ("MoviePy", "moviepy"),
    ("Playwright", "playwright"),
    ("Aiohttp", "aiohttp"),
    ("Websockets", "websockets"),
    ("Scrapy", "scrapy"),
    ("Ray", "ray"),
    ("Polars", "polars"),
    ("Vaex", "vaex"),
    ("PyTorch Lightning", "pytorch_lightning"),
    ("Hugging Face Diffusers", "diffusers"),
    ("Transformers", "transformers"),
]


def check_import(module_name: str) -> bool:
    """Return True when a module can be resolved without raising ImportError."""
    return importlib.util.find_spec(module_name) is not None


def report_installation_readiness(modules: list[tuple[str, str]]) -> dict[str, bool]:
    """Build a deterministic readiness report for optional dependencies."""
    return {name: check_import(import_name) for name, import_name in modules}


def run_cli() -> None:
    """CLI-style entrypoint used by onboarding scripts and CI checks."""
    statuses = report_installation_readiness(ADVANCED_LIBS)
    available = sum(statuses.values())
    total = len(statuses)
    print(f"[OK] Available optional libraries: {available}/{total}\n")
    for label, imported in sorted(statuses.items()):
        marker = "✓" if imported else "✗"
        print(f"{marker} {label}")


if __name__ == "__main__":
    run_cli()
