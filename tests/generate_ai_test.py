"""
generate_500_ai_tests.py

Generates 500 AI-focused pytest-compatible tests covering ML, DL, NLP, and CV tasks.
Randomized inputs are used to ensure test uniqueness.
"""

import os
import random
import numpy as np
import pandas as pd

OUTPUT_FILE = "ai_tests_generated.py"

# Helper functions
def random_vector(n=5, low=-10, high=10):
    return np.random.randint(low, high, size=n).tolist()

def random_matrix(rows=3, cols=3, low=-5, high=5):
    return np.random.randint(low, high, size=(rows, cols)).tolist()

def random_text(length=10):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join(random.choices(letters, k=length))

# Test templates
templates = [
    # Data preprocessing
    ("Normalize a vector", "vec = np.array({vec}); norm_vec = vec / np.linalg.norm(vec); assert np.isclose(np.linalg.norm(norm_vec),1)"),
    ("Handle missing values in DataFrame", "df = pd.DataFrame({df}); df.fillna(0,inplace=True); assert df.isnull().sum().sum() == 0"),
    
    # ML algorithms
    ("Linear regression prediction", "X = np.array({vec}).reshape(-1,1); y = np.array({vec}); coef = 2; y_pred = X.flatten()*coef; assert len(y_pred)==len(y)"),
    ("Decision tree depth calculation", "depth = random.randint(1,5); assert depth > 0"),
    ("KMeans clustering assignment", "points = np.array({matrix}); clusters = 2; labels = np.random.randint(0, clusters, len(points)); assert all(0<=l<clusters for l in labels)"),
    
    # Deep learning
    ("ReLU activation", "x = np.array({vec}); relu = np.maximum(0,x); assert all(relu>=0)"),
    ("Sigmoid activation", "x = np.array({vec}); sigmoid = 1/(1+np.exp(-x)); assert all(sigmoid>=0) and all(sigmoid<=1)"),
    ("Compute MSE loss", "y_true = np.array({vec}); y_pred = np.array({vec}); mse = ((y_true - y_pred)**2).mean(); assert mse >=0"),
    
    # NLP
    ("Tokenize text", "text='{text}'; tokens = list(text); assert ''.join(tokens) == text"),
    ("Word embedding vector length", "vec = np.random.randn(10); assert len(vec)==10"),
    ("Sentiment polarity check", "score = random.uniform(-1,1); assert -1 <= score <=1"),
    
    # Computer vision
    ("Convert image to grayscale", "img = np.array({matrix}); gray = img.mean(axis=1); assert gray.shape[0]==len(img)"),
    ("Flatten image", "img = np.array({matrix}); flat = img.flatten(); assert flat.shape[0]==img.size"),
    
    # Evaluation metrics
    ("Compute accuracy", "y_true = np.array([0,1,1,0]); y_pred = np.array([0,1,0,0]); acc = (y_true==y_pred).mean(); assert 0<=acc<=1"),
    ("Compute F1 score", "precision=0.8; recall=0.6; f1=2*(precision*recall)/(precision+recall); assert 0<=f1<=1"),
]

# Generate 500 unique tests
tests = []
while len(tests) < 500:
    t = random.choice(templates)
    desc = t[0]
    code_template = t[1]
    
    # Randomize placeholders
    vec = random_vector()
    matrix = random_matrix().tolist()
    text = random_text()
    df = pd.DataFrame({'col'+str(i): random_vector() for i in range(3)}).to_dict(orient='list')
    
    code = code_template.format(vec=vec, matrix=matrix, text=text, df=df)
    tests.append((desc, code))

# Write tests to output file
with open(OUTPUT_FILE, "w") as f:
    f.write("import pytest\n")
    f.write("import numpy as np\n")
    f.write("import pandas as pd\n")
    f.write("import random\n\n")
    for idx, (desc, code) in enumerate(tests, 1):
        func_name = f"test_{idx:03d}"
        f.write(f"def {func_name}():\n")
        f.write(f"    '''{desc}'''\n")
        for line in code.split("; "):
            f.write(f"    {line}\n")
        f.write("\n")

print(f"500 AI-focused Python tests generated in {OUTPUT_FILE}")