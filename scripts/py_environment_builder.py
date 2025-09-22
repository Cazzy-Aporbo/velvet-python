"""
Ultimate Python Environment Builder
Author: Cazzy
Purpose: Automatically install a huge variety of Python packages covering
data science, AI, visualization, geospatial, ML, NLP, finance, system utilities, and more.
Designed for maximum breadth and portfolio impact.
"""

import subprocess
import sys
import importlib
import logging

# =========================
# Setup logging
# =========================
logging.basicConfig(
    filename="package_installation_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# List of packages
# =========================
all_packages = [
    # Basic utilities
    "numpy", "pandas", "dask", "vaex", "modin", "polars", "pyarrow", "tables", "h5py",
    "sqlite3", "sqlalchemy", "dataset", "tinydb", "redis", "pymongo",
    
    # ML and AI
    "scikit-learn", "torch", "torchvision", "tensorflow", "keras", "pytorch-lightning",
    "fastai", "nltk", "spacy", "gensim", "transformers",
    
    # Visualization
    "matplotlib", "seaborn", "plotly", "bokeh", "altair", "geopandas", "folium", "cartopy",
    "networkx",
    
    # Geospatial
    "shapely", "fiona", "rasterio", "pyproj", "geopy", "osmnx", "contextily",
    
    # Web & networking
    "requests", "urllib3", "aiohttp", "selenium", "beautifulsoup4", "scrapy", "websockets",
    
    # Time series & finance
    "statsmodels", "arch", "yfinance", "quandl", "prophet",
    
    # Image, audio, video
    "opencv-python", "pillow", "imageio", "scikit-image", "librosa", "soundfile", "moviepy",
    
    # Scientific computing
    "scipy", "sympy", "numba", "cupy", "pynvml",
    
    # Parallel & distributed
    "ray", "joblib",
    
    # Database & big data
    "pymysql", "psycopg2-binary", "pyodbc", "cassandra-driver", "influxdb", "happybase",
    
    # File formats
    "openpyxl", "xlrd", "xlwt", "netCDF4", "pdfplumber", "PyPDF2", "python-docx", "odfpy",
    
    # Other advanced tools
    "regex", "fuzzywuzzy", "rapidfuzz", "multipledispatch", "click", "typer", "rich"
]

# =========================
# Function to install a package
# =========================
def install_package(package_name):
    """
    Try importing a package, if it fails, attempt pip install.
    Logs the result in a file.
    """
    try:
        importlib.import_module(package_name)
        logging.info(f"{package_name} is already installed.")
        print(f"{package_name} is already installed.")
    except ImportError:
        try:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logging.info(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as error:
            logging.error(f"Failed to install {package_name}: {error}")
            print(f"Failed to install {package_name}, check log.")

# =========================
# Main installer loop
# =========================
def main():
    """
    Iterate through all packages and ensure they are installed.
    """
    print("Starting Ultimate Python Environment Build...\n")
    for package in all_packages:
        install_package(package)
    print("\n✅ Environment build complete. Check package_installation_log.txt for details.")

# Run the program
if __name__ == "__main__":
    main()