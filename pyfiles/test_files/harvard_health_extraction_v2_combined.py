#!/usr/bin/env python3

#!/usr/bin/env python3

import os
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

HARVARD_BASE_URL = "https://www.health.harvard.edu"
HARVARD_BLOG_URL = "https://www.health.harvard.edu/blog"

additional_womens_health_concerns = [
    "sepsis", "meningitis", "tuberculosis", "shingles", "mono", "lyme disease",
    "zika virus", "syphilis", "trichomoniasis", "bacterial pneumonia",
    "bartholinitis", "cervicitis", "pelvic inflammatory disease", "toxic shock syndrome",
    "postpartum hemorrhage", "ectopic pregnancy", "placenta previa", "placental abruption",
    "polydipsia", "polyuria", "hypoglycemia", "hyperthyroidism", "hypothyroidism",
    "pituitary adenoma", "Cushing's syndrome", "Addison's disease",
    "angina", "peripartum cardiomyopathy", "deep vein thrombosis", "pulmonary embolism",
    "varicose veins", "Raynaud's phenomenon", "vascular disease",
    "gastritis", "peptic ulcer", "helicobacter pylori", "biliary colic", "hepatomegaly",
    "nausea in pregnancy", "hyperemesis gravidarum", "bronchitis", "pleurisy",
    "respiratory syncytial virus", "chronic sinusitis", "preeclampsia-related seizure",
    "postpartum psychosis", "catatonia", "body dysmorphia", "perinatal OCD",
    "perimenopausal depression", "cognitive impairment", "hidradenitis suppurativa",
    "rosacea", "dermatitis", "onychomycosis", "hirsutism", "hydronephrosis", "nephritis",
    "kidney stones", "renal failure", "pyelonephritis", "myalgia", "costochondritis",
    "plantar fasciitis", "bursitis", "frozen shoulder", "dry mouth", "oral thrush",
    "burning mouth syndrome", "oral lichen planus", "pericoronitis", "dry eye syndrome",
    "conjunctivitis", "glaucoma", "macular degeneration", "retinal detachment",
    "frailty syndrome", "compression fractures", "falls in elderly", "incontinence in aging",
    "hearing loss", "visual impairment", "Rh incompatibility", "amniotic fluid embolism",
    "intrauterine growth restriction (IUGR)", "gestational hypertension",
    "cholestasis of pregnancy", "stillbirth", "Paget’s disease of the breast",
    "inflammatory breast cancer", "molar pregnancy", "choriocarcinoma", "vulvar cancer",
    "fallopian tube cancer", "Turner syndrome", "fragile X syndrome", "Marfan syndrome",
    "Ehlers-Danlos syndrome", "Kallmann syndrome", "lipoedema", "mitochondrial disease"
]

def fetch_webpage(url):
    headers = {
        'User-Agent': 'Mozilla/5.0'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')

def get_article_links():
    soup = fetch_webpage(HARVARD_BLOG_URL)
    articles = soup.select("a[href^='/blog/']")
    return list({urljoin(HARVARD_BASE_URL, a['href']) for a in articles})

def extract_article_text(url):
    soup = fetch_webpage(url)
    article = soup.find("div", class_="article__body")
    return article.get_text(separator=" ").strip() if article else ""

def keyword_in_text(text, keywords):
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

def process_articles():
    article_links = get_article_links()
    seen_urls = set()
    extracted_data = []

    for i, link in enumerate(article_links):
        if link in seen_urls:
            continue
        seen_urls.add(link)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing article {i+1}/{len(article_links)}: {link}")
        text = extract_article_text(link)
        if keyword_in_text(text, additional_womens_health_concerns):
            extracted_data.append({
                "url": link,
                "content": text[:1000] + "..." if len(text) > 1000 else text
            })
        time.sleep(1.5)

    return pd.DataFrame(extracted_data)

def main():
    df = process_articles()
    raw_path = os.path.join(RAW_DATA_DIR, "harvard_health_v2_raw.csv")
    processed_path = os.path.join(PROCESSED_DATA_DIR, "harvard_health_v2_processed.csv")

    if os.path.exists(raw_path):
        try:
            df_existing = pd.read_csv(raw_path)
        except pd.errors.EmptyDataError:
            df_existing = pd.DataFrame()
    else:
        df_existing = pd.DataFrame()

    df = pd.concat([df_existing, df]).drop_duplicates(subset=["url"])
    df.to_csv(raw_path, index=False)
    df.to_csv(processed_path, index=False)

    print(f"""Saved {len(df)} relevant articles to:
  - {raw_path}
  - {processed_path}""")

if __name__ == "__main__":
    main()

