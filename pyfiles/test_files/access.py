import requests
from bs4 import BeautifulSoup

# URL to scrape
url = 'http://openinsider.com/insider-purchases'

# Send a HTTP request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Now you need to find the data you are interested in.
    # This usually involves inspecting the HTML structure of the website
    # and identifying the tags, classes, or IDs that contain the data.
    # For example, to find all elements with the class 'data', you could use:
    # data_elements = soup.find_all(class_='data')
    
    # Process your data_elements here
    
    print("Data extracted successfully.")
else:
    print(f"Failed to retrieve data, status code: {response.status_code}")
