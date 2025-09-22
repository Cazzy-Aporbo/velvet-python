"""
Ultimate Python Environment Installer and Tester
Author: Cazzy
Purpose: Install and verify functionality of advanced Python packages for portfolio demonstration.
Includes tests for core, ML/AI, GPU, visualization, and distributed computing libraries.
"""

import subprocess
import sys
import importlib
import logging
import platform
import warnings

# =========================
# Logging setup
# =========================
logging.basicConfig(
    filename="ultimate_env_test_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# System info
# =========================
python_version = sys.version.split()[0]
system_name = platform.system()
architecture = platform.architecture()[0]

print(f"Python version: {python_version}")
print(f"OS: {system_name}")
print(f"Architecture: {architecture}")

# =========================
# Packages to install
# =========================
all_packages = [
    "numpy", "pandas", "jax", "jaxlib", "cupy", "torch", "torchvision", "torch-geometric",
    "tensorflow", "tensorflow-probability", "pytorch-lightning", "functorch",
    "matplotlib", "seaborn", "plotly", "holoviews", "panel", "pyvista", "vedo", "mayavi",
    "dash", "bokeh", "napari", "ipyvolume", "pythreejs",
    "dask", "ray", "prefect", "luigi", "airflow",
    "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "nevergrad",
    "networkx", "graph-tool", "igraph", "gudhi", "giotto-tda", "torch-geometric",
    "qiskit", "cirq", "pennylane",
    "cvxpy", "pyomo", "gekko", "casadi"
]

# =========================
# Helper functions
# =========================
def install_package(pkg):
    try:
        importlib.import_module(pkg.replace("-", "_"))
        logging.info(f"{pkg} already installed")
        print(f"{pkg} already installed")
    except ImportError:
        try:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            logging.info(f"Successfully installed {pkg}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {pkg}: {e}")
            print(f"Failed to install {pkg}, check log")

def test_numpy_pandas():
    try:
        import numpy as np
        import pandas as pd
        arr = np.arange(10)
        df = pd.DataFrame(arr, columns=["numbers"])
        logging.info("Numpy and Pandas test passed")
        print("Numpy and Pandas test passed")
    except Exception as e:
        logging.error(f"Numpy/Pandas test failed: {e}")
        print("Numpy/Pandas test failed")

def test_torch_cuda():
    try:
        import torch
        gpu = torch.cuda.is_available()
        logging.info(f"Torch test passed, CUDA available: {gpu}")
        print(f"Torch test passed, CUDA available: {gpu}")
    except Exception as e:
        logging.error(f"Torch test failed: {e}")
        print("Torch test failed")

def test_jax_cuda():
    try:
        import jax
        import jax.numpy as jnp
        x = jnp.array([1.0, 2.0, 3.0])
        gpu = jax.devices()[0].platform
        logging.info(f"JAX test passed, platform: {gpu}")
        print(f"JAX test passed, platform: {gpu}")
    except Exception as e:
        logging.error(f"JAX test failed: {e}")
        print("JAX test failed")

def test_visualization():
    try:
        import matplotlib.pyplot as plt
        import plotly.express as px
        import pyvista as pv
        # Matplotlib test
        plt.plot([0,1], [0,1])
        plt.close()
        # Plotly test
        fig = px.line(x=[0,1], y=[0,1])
        # PyVista test
        cube = pv.Cube()
        logging.info("Visualization libraries test passed")
        print("Visualization libraries test passed")
    except Exception as e:
        logging.error(f"Visualization test failed: {e}")
        print("Visualization test failed")

def test_distributed():
    try:
        import dask.array as da
        import ray
        ray.init(ignore_reinit_error=True)
        x = da.ones((1000,1000), chunks=(100,100))
        x.sum().compute()
        logging.info("Distributed computing libraries test passed")
        print("Distributed computing libraries test passed")
        ray.shutdown()
    except Exception as e:
        logging.error(f"Distributed computing test failed: {e}")
        print("Distributed computing test failed")

# =========================
# Create requirements.txt
# =========================
def create_requirements(packages):
    import pkg_resources
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for pkg in packages:
            try:
                version = pkg_resources.get_distribution(pkg).version
                f.write(f"{pkg}=={version}\n")
            except Exception:
                f.write(f"{pkg}\n")
    print("requirements.txt created")

# =========================
# Main installer and tester
# =========================
def main():
    print("\nStarting installation and testing of ultimate Python environment...\n")
    
    for pkg in all_packages:
        install_package(pkg)
    
    print("\nRunning tests...\n")
    test_numpy_pandas()
    test_torch_cuda()
    test_jax_cuda()
    test_visualization()
    test_distributed()
    
    create_requirements(all_packages)
    print("\nAll done. Check ultimate_env_test_log.txt for details.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()