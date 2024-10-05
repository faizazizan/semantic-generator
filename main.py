import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd

# Function to ensure necessary NLTK datasets are downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download NLTK data
download_nltk_data()

# Function to get SERP results
def get_serp_results(query):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

# Function to analyze SERP and extract related topics
def topical_mapping(query):
    soup = get_serp_results(query)
    search_results = soup.find_all('div', class_='tF2Cxc')
    related_topics = []

    for result in search_results:
        link = result.find('a')['href']
        response = requests.get(link)
        page_soup = BeautifulSoup(response.text, 'html.parser')
        page_text = page_soup.get_text()
        tokens = word_tokenize(page_text.lower())
        words = [word for word in tokens if word.isalnum()]
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        word_freq = Counter(filtered_words)
        related_topics.extend(word_freq.most_common(10))

    return Counter(dict(related_topics)).most_common(10)

# Function to generate semantic keywords
def generate_semantic_keywords(query):
    related_texts = [result[0] for result in topical_mapping(query)]
    
    if not related_texts:
        return []
    
    semantic_keywords = set()
    for text in related_texts:
        tokens = word_tokenize(text.lower())
        words = [word for word in tokens if word.isalnum()]
        semantic_keywords.update(words)
    return list(semantic_keywords)

# Streamlit app
st.title("Keyword Analyzer for SEO")

query = st.text_input("Enter a keyword to analyze:", "")

if query:
    st.write(f"Analyzing '{query}'...")

    st.write("### Semantic Keywords")
    semantic_keywords = generate_semantic_keywords(query)
    semantic_keywords_df = pd.DataFrame(semantic_keywords, columns=['Semantic Keywords'])
    st.table(semantic_keywords_df)
