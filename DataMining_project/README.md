# Decoding E-Commerce: Mining eMAG and Amazon Product Reviews

**Data Mining Course Project -- 2025/2026**
**Politehnica University of Timisoara**

---

## Project Overview

This project applies comprehensive Data Mining and Natural Language Processing (NLP) techniques to analyze thousands of product reviews from two major e-commerce platforms: **eMAG** (Romanian market) and **Amazon** (Global market). By examining consumer feedback across four major electronics categories (phones, laptops, headphones, and tablets), the analysis aims to uncover the primary drivers of consumer satisfaction, predict product ratings from text, and discover latent behavioral themes.

The project is structured to meet and exceed the course requirements, strictly following the 6 core Data Mining requirements across multiple datasets.

## Research Objectives

1. **Satisfaction Analysis**: What specific factors drive positive (5-star) versus negative (1-star) reviews across different markets?
2. **Predictive Modeling**: Can a supervised Machine Learning model accurately predict a product's star rating based entirely on the raw text of the review?
3. **Latent Theme Discovery**: What underlying topics dominate consumer feedback in both Romanian and English markets (e.g., battery life, delivery logistics, customer service)?

## Technical Pipeline & Core Requirements Covered

This project implements a complete, end-to-end data mining pipeline, fulfilling at least 6 core academic requirements by integrating multilingual transformer models and multi-category data processing:

1. **Data Collection (Requirement 1)**
   - Custom Python scraper for eMAG using `requests` and `BeautifulSoup4` with dual extraction strategy (JSON-LD and HTML parsing).
   - Automated downloader script for the Amazon 2023 Electronics dataset (via HuggingFace caching).

2. **Data Cleaning & Preprocessing (Requirement 2)**
   - Handling missing values and removing absolute duplicates.
   - Normalization of Romanian and English text: regex filtering, HTML tag removal, and Unicode diacritic normalization.
   - Tokenization and removal of custom stop-words.
   - Feature engineering (e.g., word count, review length).

3. **Exploratory Data Analysis (EDA) (Requirement 3)**
   - Statistical visualizations of rating distributions using `matplotlib` and `seaborn`.
   - Cross-category rating comparisons.
   - Analysis of review length vs. sentiment polarity.
   - Polarity-specific word clouds.

4. **TF-IDF Analysis (Requirement 4)**
   - Term Frequency-Inverse Document Frequency vectorization.
   - Extraction of the most definitive terms separating high-satisfaction from high-dissatisfaction reviews.

5. **Topic Modelling / Unsupervised Learning (Requirement 5)**
   - Unsupervised clustering using Latent Dirichlet Allocation (LDA).
   - Identification of 5 dominant conversational topics per language (e.g., Delivery & Returns, Build Quality, Performance).

6. **Supervised Classification (Requirement 6)**
   - Binary sentiment classification (Positive >= 4 stars; Negative <= 2 stars).
   - Training and evaluating multiple models: Logistic Regression, Random Forest Classifier, and Support Vector Machines (SVM).
   - Feature Importance extraction to validate TF-IDF findings.

7. **Bonus: Sentiment Analysis via Transformers**
   - Implementation of HuggingFace's `nlptown/bert-base-multilingual-uncased-sentiment`.
   - Zero-shot sentiment prediction directly on native texts.

## Project Structure

```text
DataMining_project/
|-- data/
|   |-- raw/                        # Raw scraped CSV data
|   |-- processed/                  # Cleaned and tokenized datasets
|-- notebooks/
|   |-- emag_analysis.ipynb         # Jupyter Notebook with full eMAG pipeline
|   |-- amazon_analysis.ipynb       # Jupyter Notebook with full Amazon pipeline
|-- src/
|   |-- scraper.py                  # Standalone eMAG web scraper and data generator
|   |-- download_amazon.py          # Script to download Amazon reviews via HF caching
|   |-- generate_amazon_notebook.py # Script to autogenerate the Amazon notebook
|-- requirements.txt                # Python dependencies
|-- .gitignore
|-- README.md                       # Project documentation
|-- PROJECT_DOCUMENTATION.md        # Technical design and rationale
```

## Setup and Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository_url>
   cd DataMining_project
   ```

2. **Create a virtual environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### 1. Data Collection
To run the scrapers manually to collect live data or generate a fresh dataset:
```bash
# For eMAG (Romanian):
python src/scraper.py

# For Amazon (English):
python src/download_amazon.py
```
*Note: The eMAG script falls back to generating a realistic sample dataset if live scraping is restricted.*

### 2. Execution and Analysis
The core analysis is housed in two Jupyter Notebooks.
```bash
jupyter notebook notebooks/emag_analysis.ipynb
jupyter notebook notebooks/amazon_analysis.ipynb
```
Execute the cells sequentially in each notebook. 
*Note: The Multilingual BERT section is computationally intensive. Running this specific section on Google Colab with GPU acceleration is recommended for large datasets.*

## Team Members
- [Toma Ionut-Adrian]
- [Boros Fabian]

## License
This project is created for academic purposes for the Data Mining course at the Politehnica University of Timisoara.
