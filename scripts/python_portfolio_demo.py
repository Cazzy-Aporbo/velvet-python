"""
Ultimate Python Portfolio Demo Extended
Author: Cazzy
Purpose: Install, test, and run illustrative demos for advanced Python packages.
Includes multiple examples per domain 
"""

import subprocess
import sys
import importlib
import logging
import platform
import warnings

# Logging setup
logging.basicConfig(
    filename="ultimate_portfolio_demo_extended_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# System info
python_version = sys.version.split()[0]
system_name = platform.system()
architecture = platform.architecture()[0]

print(f"Python version: {python_version}")
print(f"Operating system: {system_name}")
print(f"Architecture: {architecture}")

# Packages list for demo
all_packages = [
    "numpy", "pandas", "scikit-learn", "torch", "jax", "jaxlib", "pytorch-lightning",
    "matplotlib", "plotly", "pyvista", "networkx", "ray", "dask",
    "qiskit", "biopython"
]

def install_package(package):
    try:
        importlib.import_module(package.replace("-", "_"))
        logging.info(f"{package} already installed")
        print(f"{package} already installed")
    except ImportError:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logging.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to install {package}: {e}")
            print(f"Failed to install {package}, check log")

# Portfolio demos with multiple examples
def demo_numpy_pandas():
    import numpy as np
    import pandas as pd
    arr = np.arange(10)
    df = pd.DataFrame({"numbers": arr, "squared": arr**2, "sqrt": np.sqrt(arr)})
    print("\nNumpy and Pandas demo:")
    print(df.head())
    logging.info("Numpy/Pandas demo ran successfully")

def demo_scikit_learn():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    pred = model.predict(poly.transform([[6]]))
    print("\nScikit-learn demo prediction for 6 (polynomial regression):", pred)
    logging.info("Scikit-learn demo ran successfully")

def demo_torch_jax():
    import torch
    import jax
    import jax.numpy as jnp
    x_torch = torch.tensor([1.0, 2.0, 3.0])
    x_jax = jnp.array([1.0, 2.0, 3.0])
    print("\nTorch tensor:", x_torch, "CUDA available:", torch.cuda.is_available())
    print("JAX tensor:", x_jax, "Device platform:", jax.devices()[0].platform)
    logging.info("Torch/JAX demo ran successfully")

def demo_visualization():
    import matplotlib.pyplot as plt
    import plotly.express as px
    import pyvista as pv

    # Matplotlib example
    plt.scatter([0, 1, 2], [0, 1, 4], c=["red", "green", "blue"])
    plt.close()

    # Plotly example
    fig = px.bar(x=["A", "B", "C"], y=[5, 10, 7])
    
    # PyVista example
    cube = pv.Cube()
    sphere = pv.Sphere()
    
    print("\nVisualization demo executed for Matplotlib, Plotly, and PyVista")
    logging.info("Visualization demo ran successfully")

def demo_network_graph():
    import networkx as nx
    G = nx.karate_club_graph()
    degree_dict = dict(G.degree())
    top_node = max(degree_dict, key=degree_dict.get)
    print("\nNetwork demo: Karate Club graph nodes:", G.number_of_nodes())
    print("Node with highest degree:", top_node)
    logging.info("Network demo ran successfully")

def demo_distributed():
    import dask.array as da
    import ray
    ray.init(ignore_reinit_error=True)
    x = da.random.random((1000, 1000), chunks=(100, 100))
    result = x.sum().compute()
    print("\nDistributed computing demo result sum:", result)
    logging.info("Distributed computing demo ran successfully")
    ray.shutdown()

def demo_quantum():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    print("\nQuantum demo circuit:")
    print(qc)
    logging.info("Quantum computing demo ran successfully")

def demo_bioinformatics():
    from Bio.Seq import Seq
    my_seq = Seq("AGTACACTGGT")
    print("\nBioinformatics demo sequence complement:", my_seq.complement())
    print("Bioinformatics demo sequence reverse complement:", my_seq.reverse_complement())
    logging.info("Bioinformatics demo ran successfully")

def create_requirements(packages):
    import pkg_resources
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for pkg in packages:
            try:
                version = pkg_resources.get_distribution(pkg).version
                f.write(f"{pkg}=={version}\n")
            except Exception:
                f.write(f"{pkg}\n")
    print("\nrequirements.txt created")

def main():
    print("\nStarting Ultimate Portfolio Extended Demo Installation...\n")
    
    for pkg in all_packages:
        install_package(pkg)
    
    print("\nRunning portfolio demos with multiple examples...\n")
    demo_numpy_pandas()
    demo_scikit_learn()
    demo_torch_jax()
    demo_visualization()
    demo_network_graph()
    demo_distributed()
    demo_quantum()
    demo_bioinformatics()
    
    create_requirements(all_packages)
    print("\nAll demos complete. Check ultimate_portfolio_demo_extended_log.txt for details.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()