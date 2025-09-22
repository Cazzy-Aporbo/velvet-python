"""
Python Import Encyclopedia
Author: Cazzy
Purpose: A single file containing a vast collection of commonly used and advanced import statements
for data science, AI, visualization, geospatial work, web scraping, system utilities, and more.
Uncomment the ones you need in your project.
"""

# ------------------------
# Basic Python Utilities
# ------------------------
import os
import sys
import time
import datetime
import math
import random
import re
import logging
import functools
import itertools
import collections
import operator
import string
import copy
import pathlib
import json
import csv
import pickle
import pprint
import shutil
import tempfile
import glob
import hashlib
import uuid
import subprocess
import threading
import multiprocessing
import queue
import signal
import argparse
import configparser
import types
import warnings
import inspect

# ------------------------
# Data Manipulation
# ------------------------
import numpy as np
import pandas as pd
import dask.dataframe as dd
import vaex
import modin.pandas as mpd
import polars as pl
import pyarrow as pa
import tables
import h5py
import sqlite3
import sqlalchemy
import dataset
import tinydb
import redis
import pymongo

# ------------------------
# Machine Learning & AI
# ------------------------
# Classical ML
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, RFE

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, preprocessing
import pytorch_lightning as pl_lightning
import fastai

# NLP
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy import displacy
import gensim
from gensim.models import Word2Vec, LdaModel, FastText
import transformers
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel

# ------------------------
# Data Visualization
# ------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
import bokeh
from bokeh.plotting import figure, show, output_file
import altair as alt
import geopandas as gpd
import folium
import cartopy.crs as ccrs
import networkx as nx

# ------------------------
# Geospatial & Maps
# ------------------------
import shapely
from shapely.geometry import Point, Polygon, LineString
import fiona
import rasterio
import pyproj
import geopy
from geopy.geocoders import Nominatim
import osmnx as ox
import folium
import contextily

# ------------------------
# Web & Networking
# ------------------------
import requests
import urllib
import urllib.request
import urllib.parse
import urllib.error
import http.client
import socket
import paramiko
import ftplib
import websockets
import aiohttp
import selenium
from selenium import webdriver
from bs4 import BeautifulSoup
import scrapy

# ------------------------
# Time Series & Finance
# ------------------------
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import arch
import yfinance as yf
import quandl
import prophet

# ------------------------
# Image, Audio & Video
# ------------------------
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFilter
import imageio
import skimage
from skimage import io, filters, feature, color
import librosa
import soundfile as sf
import moviepy.editor as mp

# ------------------------
# Scientific Computing
# ------------------------
import scipy
from scipy import stats, optimize, signal, linalg, spatial
import sympy
import numba
import cupy
import pynvml

# ------------------------
# Parallel & Distributed
# ------------------------
import concurrent.futures
import ray
import dask
import joblib

# ------------------------
# Database & Big Data
# ------------------------
import pymysql
import psycopg2
import pyodbc
import cassandra
from cassandra.cluster import Cluster
import influxdb
import happybase

# ------------------------
# File Formats
# ------------------------
import openpyxl
import xlrd
import xlwt
import h5py
import netCDF4
import pdfplumber
import PyPDF2
import docx
import odf

# ------------------------
# Other Advanced Tools
# ------------------------
import regex
import fuzzywuzzy
import rapidfuzz
import symengine
import multipledispatch
import click
import typer
import rich
from rich.console import Console
from rich.table import Table

# End of Import Encyclopedia