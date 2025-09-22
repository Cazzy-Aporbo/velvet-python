"""
Ultimate Python Environment Builder Pro
Author: Cazzy
Purpose: Install and manage a comprehensive set of Python packages for data science, AI,
visualization, geospatial, scientific computing, web, and system utilities.
Designed for maximum professionalism and reproducibility.
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
    filename="environment_build_log.txt",
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
# Package categories
# =========================
package_categories = {
    "core_utilities": [
        "os", "sys", "time", "datetime", "math", "random", "re", "logging", "functools", "itertools",
        "collections", "operator", "string", "copy", "pathlib", "json", "csv", "pickle", "shutil", "tempfile",
        "glob", "hashlib", "uuid", "subprocess", "threading", "multiprocessing", "queue", "signal",
        "argparse", "configparser", "types", "warnings", "inspect"
    ],
    "data_manipulation": [
        "numpy", "pandas", "dask", "vaex", "modin", "polars", "pyarrow", "tables", "h5py",
        "dataset", "tinydb", "redis", "pymongo", "sqlalchemy", "sqlite3"
    ],
    "machine_learning": [
        "scikit-learn", "torch", "torchvision", "tensorflow", "keras", "pytorch-lightning",
        "fastai", "nltk", "spacy", "gensim", "transformers"
    ],
    "visualization": [
        "matplotlib", "seaborn", "plotly", "bokeh", "altair", "geopandas", "folium",
        "cartopy", "networkx"
    ],
    "geospatial": [
        "shapely", "fiona", "rasterio", "pyproj", "geopy", "osmnx", "contextily"
    ],
    "web_and_network": [
        "requests", "urllib3", "aiohttp", "selenium", "beautifulsoup4", "scrapy", "websockets"
    ],
    "time_series_and_finance": [
        "statsmodels", "arch", "yfinance", "quandl", "prophet"
    ],
    "image_audio_video": [
        "opencv-python", "pillow", "imageio", "scikit-image", "librosa", "soundfile", "moviepy"
    ],
    "scientific_computing": [
        "scipy", "sympy", "numba", "cupy", "pynvml"
    ],
    "parallel_and_distributed": [
        "ray", "joblib"
    ],
    "databases_and_big_data": [
        "pymysql", "psycopg2-binary", "pyodbc", "cassandra-driver", "influxdb", "happybase"
    ],
    "file_formats_and_documentation": [
        "openpyxl", "xlrd", "xlwt", "netCDF4", "pdfplumber", "PyPDF2", "python-docx", "odfpy"
    ],
    "advanced_tools": [
        "regex", "fuzzywuzzy", "rapidfuzz", "multipledispatch", "click", "typer", "rich"
    ]
}

# =========================
# Function to install package
# =========================
def install_package(package_name):
    """
    Attempt to import a package; if missing, install via pip.
    Logs success or failure.
    """
    try:
        importlib.import_module(package_name)
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
# Interactive category selection
# =========================
def select_categories():
    """
    Let user choose which categories to install.
    Returns list of selected packages.
    """
    print("\nAvailable package categories:")
    for index, category in enumerate(package_categories.keys(), start=1):
        print(f"{index}. {category}")
    print("Enter numbers separated by commas for categories you want to install, or 'all' for everything.")
    
    user_input = input("Your selection: ").strip().lower()
    selected_packages = []
    
    if user_input == "all":
        for packages in package_categories.values():
            selected_packages.extend(packages)
    else:
        indices = [int(i) for i in user_input.split(",") if i.isdigit()]
        keys = list(package_categories.keys())
        for i in indices:
            if 1 <= i <= len(keys):
                selected_packages.extend(package_categories[keys[i-1]])
    
    return selected_packages

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
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "unknown")
                req_file.write(f"{package}=={version}\n")
            except Exception:
                req_file.write(f"{package}\n")
    print("requirements.txt created with package versions.")

# =========================
# Main function
# =========================
def main():
    """
    Build Python environment interactively and generate requirements.txt.
    """
    selected_packages = select_categories()
    print("\nStarting installation process...\n")
    
    for package in selected_packages:
        install_package(package)
    
    create_requirements_file(selected_packages)
    print("\nEnvironment build complete. Check environment_build_log.txt for details.")

# Run program
if __name__ == "__main__":
    main()