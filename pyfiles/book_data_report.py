import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def fetch_google_books(query="python", max_results=40):
    url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
    response = requests.get(url).json()
    books = []
    for item in response.get("items", []):
        info = item.get("volumeInfo", {})
        books.append({
            "title": info.get("title"),
            "authors": ", ".join(info.get("authors", [])),
            "publisher": info.get("publisher"),
            "published_date": info.get("publishedDate"),
            "categories": ", ".join(info.get("categories", [])),
            "average_rating": info.get("averageRating", 0),
            "ratings_count": info.get("ratingsCount", 0)
        })
    df = pd.DataFrame(books)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce")
    return df

def fetch_open_library(query="python", limit=40):
    url = f"https://openlibrary.org/search.json?q={query}&limit={limit}"
    response = requests.get(url).json()
    books = []
    for doc in response.get("docs", []):
        books.append({
            "title": doc.get("title"),
            "authors": ", ".join(doc.get("author_name", [])),
            "publisher": ", ".join(doc.get("publisher", [])) if doc.get("publisher") else None,
            "published_date": doc.get("first_publish_year"),
            "categories": ", ".join(doc.get("subject", [])) if doc.get("subject") else "Unknown"
        })
    df = pd.DataFrame(books)
    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", format="%Y")
    return df

def clean_books_data(df):
    df["categories"].fillna("Unknown", inplace=True)
    df["average_rating"].fillna(0, inplace=True)
    df["ratings_count"].fillna(0, inplace=True)
    df["year"] = df["published_date"].dt.year
    return df

def plot_top(df, column, top_n=10, filename=None, title=None):
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(10,5))
    counts.plot.barh()
    plt.title(title if title else f"Top {column}")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def plot_time_series(df, column, filename=None, title=None):
    counts = df[column].value_counts().sort_index()
    plt.figure(figsize=(10,5))
    counts.plot()
    plt.title(title if title else f"{column} over Time")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.close()

def generate_html_report(df, images, filename="books_data_report.html"):
    html_content = f"""
    <html>
    <head><title>Books Data Report</title></head>
    <body>
    <h1>Books Data Report</h1>
    <p>Report generated: {datetime.now()}</p>
    """
    for caption, img in images.items():
        html_content += f"<h2>{caption}</h2><img src='{img}' width='600'><br>"
    html_content += "<h2>Sample Books</h2>"
    html_content += df.head(20).to_html()
    html_content += "</body></html>"
    with open(filename, "w") as f:
        f.write(html_content)
    print(f"Report generated: {filename}")

def main():
    df_google = fetch_google_books()
    df_ol = fetch_open_library()
    df_books = pd.concat([df_google, df_ol], ignore_index=True)
    df_books = clean_books_data(df_books)

    plot_top(df_books, "authors", filename="top_authors.png", title="Top Authors")
    plot_time_series(df_books, "year", filename="books_by_year.png", title="Books Published Over Years")
    plot_top(df_books, "categories", filename="top_categories.png", title="Top Categories")

    images = {
        "Top Authors": "top_authors.png",
        "Books Published Over Years": "books_by_year.png",
        "Top Categories": "top_categories.png"
    }
    generate_html_report(df_books, images)

if __name__ == "__main__":
    main()