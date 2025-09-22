"""
Ultimate Python Environment Installer
Author: Cazzy
Purpose: Automatically install all packages from three advanced Python import reference files.
Includes error handling, logging, requirements.txt generation, and optional GPU detection.
"""

import subprocess
import sys
import importlib
import logging
import platform

# =========================
# Logging setup
# =========================
logging.basicConfig(
    filename="ultimate_env_install_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Detect Python version and system
# =========================
python_version = sys.version.split()[0]
system_name = platform.system()
architecture = platform.architecture()[0]

print(f"Python version detected: {python_version}")
print(f"Operating system: {system_name}")
print(f"System architecture: {architecture}")

# =========================
# GPU detection (CUDA-enabled packages)
# =========================
gpu_available = False
try:
    import torch
    gpu_available = torch.cuda.is_available()
except ImportError:
    gpu_available = False

if gpu_available:
    print("CUDA-enabled GPU detected.")
else:
    print("No CUDA-enabled GPU detected or PyTorch not installed.")

# =========================
# Combined list of packages from all three files
# =========================
# Only include pip-installable packages
all_packages = [
    # Advanced numerical computation
    "jax", "jaxlib", "cupy", "torch", "torchvision", "torch_sparse", "torch_scatter", 
    "torch_geometric", "mpi4py", "numba", "pytensor", "aesara", "pytorch-lightning",
    "functorch", "cupyx", "tensorly", "tensornetwork", "einops", 

    # Advanced ML/AI
    "diffusers", "accelerate", "transformers", "timm", "detectron2", "monai", 
    "lightning", "skorch", "autogluon", "flaml", "nevergrad", "optuna", "ray", 
    "rllib", "rl_games", "stable-baselines3", "gymnasium", "gym", "jaxopt",
    "tensorflow-probability", "pyro-ppl", "numpyro", "pymc3", "bambi", "scikit-optimize",

    # Graphs, networks, and topology
    "networkx", "graph-tool", "python-igraph", "pyvis", "stellargraph", "karateclub",
    "gudhi", "giotto-tda", "ripser", "persim", "pyflagser", "hypernetx", "h3",
    "scikit-network",

    # Advanced visualization
    "plotly", "holoviews", "panel", "bokeh", "datashader", "pyvista", "vedo",
    "mayavi", "dash", "dash-core-components", "dash-html-components",
    "napari", "ipyvolume", "pythreejs", "vtk", "matplotlib", "seaborn",

    # Geospatial
    "geopandas", "fiona", "rioxarray", "salem", "movingpandas", "pyproj", 
    "rasterio", "regionmask", "skgstat", "pysal", "osmnx", "folium", "mapclassify", "pydeck",

    # Time series, forecasting, streaming
    "sktime", "darts", "neuralprophet", "tbats", "prophet", "kats", "statsforecast", 
    "pmdarima", "pyflux", "river", "creme", "apache-beam", "faust", "trino",

    # Bioinformatics / scientific computing
    "biopython", "pybiomart", "pysam", "scikit-bio", "scanpy", "anndata", "squidpy",
    "loompy", "cellrank", "mudata", "scanorama",

    # Quantum computing / simulation
    "qiskit", "cirq", "braket.aws", "qulacs", "pennylane", "openfermion", "openfermioncirq",
    "projectq", "pyquil", "tensorflow-quantum",

    # Optimization / control / physics
    "cvxpy", "cvxopt", "gekko", "pyomo", "casadi", "do-mpc", "pybullet", "mujoco-py", "bullet3",

    # File formats / pipelines
    "h5py", "zarr", "netCDF4", "pyarrow", "parquet", "feather-format", "fastparquet", "tables",
    "odfpy", "pdfplumber", "PyPDF2", "python-docx", "xlwings", "snakemake", "luigi", "prefect",
    "dagster", "kedro", "airflow", "streamz", "dask", "dask[distributed]",

    # Other tools
    "rich", "click", "typer", "loguru", "icecream", "tqdm", "alive-progress",
    "memory-profiler", "line-profiler", "pyinstrument", "watchgod", "fastapi",
    "uvicorn", "starlette", "pydantic"
]

# =========================
# Function to install a package
# =========================
def install_package(package_name):
    """
    Attempt to import a package; if missing, install via pip.
    Logs success or failure.
    """
    try:
        importlib.import_module(package_name.replace("-", "_"))
        logging.info(f"{package_name} already installed.")
        print(f"{package_name} already installed.")
    except ImportError:
        try:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logging.info(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as error:
            logging.error(f"Failed to install {package_name}: {error}")
            print(f"Failed to install {package_name}, check log.")

# =========================
# Function to create requirements.txt
# =========================
def create_requirements_file(package_list):
    """
    Generates a requirements.txt file with installed package versions.
    """
    with open("requirements.txt", "w", encoding="utf-8") as req_file:
        for package in package_list:
            try:
                module = importlib.import_module(package.replace("-", "_"))
                version = getattr(module, "__version__", "unknown")
                req_file.write(f"{package}=={version}\n")
            except Exception:
                req_file.write(f"{package}\n")
    print("requirements.txt created with package versions.")

# =========================
# Main installer
# =========================
def main():
    """
    Install all packages and generate requirements.txt
    """
    print("\nStarting Ultimate Python Environment Installation...\n")
    for package in all_packages:
        install_package(package)
    create_requirements_file(all_packages)
    print("\nInstallation complete. Check ultimate_env_install_log.txt for details.")

# Run installer
if __name__ == "__main__":
    main()