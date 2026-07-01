"""Specialized optional imports catalog used by advanced learning tracks."""

from __future__ import annotations

import importlib.util
from typing import Iterable


ADVANCED_LIBS: list[tuple[str, str]] = [
    ("Qiskit", "qiskit"),
    ("Cirq", "cirq"),
    ("PennyLane", "pennylane"),
    ("PyTorch", "torch"),
    ("TensorNetwork", "tensornetwork"),
    ("Gymnasium", "gymnasium"),
    ("Stable-Baselines3", "stable_baselines3"),
    ("TensorFlow Probability", "tensorflow_probability"),
    ("Scikit-Optimize", "skopt"),
    ("PyMC", "pymc"),
    ("Bambi", "bambi"),
    ("NetworkX", "networkx"),
    ("PyVIS", "pyvis"),
    ("GUDHI", "gudhi"),
    ("Ripser", "ripser"),
    ("Persim", "persim"),
    ("H3", "h3"),
    ("Sphinx", "sphinx"),
    ("Napari", "napari"),
    ("Ipyvolume", "ipyvolume"),
    ("PyDeck", "pydeck"),
    ("Geopandas", "geopandas"),
    ("PyArrow", "pyarrow"),
    ("Xarray", "xarray"),
    ("DuckDB", "duckdb"),
    ("Dask", "dask"),
    ("Streamz", "streamz"),
    ("Prefect", "prefect"),
    ("Dagster", "dagster"),
    ("Luigi", "luigi"),
    ("Apache Beam", "apache_beam"),
    ("Kedro", "kedro"),
    ("Rich", "rich"),
    ("Typer", "typer"),
    ("Loguru", "loguru"),
    ("FastAPI", "fastapi"),
    ("uvicorn", "uvicorn"),
]


def available(module_name: str) -> bool:
    """Return True when the optional module import target can be resolved."""
    return importlib.util.find_spec(module_name) is not None


def check(modules: Iterable[tuple[str, str]]) -> dict[str, bool]:
    """Resolve a mapping of human-friendly names to import availability."""
    return {name: available(import_name) for name, import_name in modules}


def print_report(modules: Iterable[tuple[str, str]]) -> None:
    """Print a clean, deterministic report used by docs and onboarding."""
    statuses = check(modules)
    print("Advanced library readiness:")
    for name, is_available in sorted(statuses.items()):
        status = "available" if is_available else "missing"
        print(f"- {name}: {status}")


if __name__ == "__main__":
    print_report(ADVANCED_LIBS)
