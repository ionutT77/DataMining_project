# Decoding Romanian E-Commerce: Mining eMAG Product Reviews

**Data Mining Course Project -- 2025/2026**
**Politehnica University of Timisoara**

---

## Project Overview

This project applies comprehensive Data Mining and Natural Language Processing (NLP) techniques to analyze thousands of product reviews scraped from **eMAG.ro**, Romania's largest e-commerce platform. By examining consumer feedback across four major electronics categories (phones, laptops, headphones, and tablets), the analysis aims to uncover the primary drivers of consumer satisfaction, predict product ratings from text, and discover latent behavioral themes.

## Research Objectives

1. **Satisfaction Analysis**: What specific factors drive positive (5-star) versus negative (1-star) reviews?
2. **Predictive Modeling**: Can a supervised Machine Learning model accurately predict a product's star rating based entirely on the raw text of the review?
3. **Latent Theme Discovery**: What underlying topics dominate consumer feedback in the Romanian market (e.g., battery life, delivery logistics, customer service)?

## Technical Pipeline & Topics Covered

This project implements a complete, end-to-end data mining pipeline, surpassing the baseline academic requirements by integrating multilingual transformer models and multi-category data processing.

1. **Web Scraping & Data Collection**
   - Custom Python scraper using `requests` and `BeautifulSoup4`.
   - Politeness mechanisms: randomized delays and User-Agent rotation to bypass rate-limiting.
   - Built-in fallback generator for synthetic, realistic Romanian reviews when live scraping is blocked.

2. **Data Cleaning & Preprocessing**
   - Handling missing values and removing absolute duplicates.
   - Normalization of Romanian text: regex filtering, HTML tag removal, and Unicode diacritic normalization.
   - Tokenization and removal of custom Romanian stop-words.
   - Feature engineering (e.g., word count, review length).

3. **Exploratory Data Analysis (EDA)**
   - Statistical visualizations of rating distributions using `matplotlib` and `seaborn`.
   - Cross-category rating comparisons.
   - Analysis of review length vs. sentiment polarity.
   - Polarity-specific word clouds.

4. **TF-IDF Analysis**
   - Term Frequency-Inverse Document Frequency vectorization.
   - Extraction of the most definitive terms separating high-satisfaction from high-dissatisfaction reviews.

5. **Sentiment Analysis via Transformers**
   - Implementation of HuggingFace's `nlptown/bert-base-multilingual-uncased-sentiment`.
   - Zero-shot sentiment prediction directly on native Romanian text, bypassing the need for lossy English translation.
   - Evaluation via Confusion Matrices and Exact Match Accuracy.

6. **Topic Modelling (LDA)**
   - Unsupervised clustering using Latent Dirichlet Allocation.
   - Identification of 5 dominant conversational topics (e.g., Delivery & Returns, Build Quality, Performance).

7. **Supervised Classification**
   - Binary sentiment classification (Positive >= 4 stars; Negative <= 2 stars).
   - Training and evaluating multiple models: Logistic Regression, Random Forest Classifier, and Support Vector Machines (SVM).
   - Feature Importance extraction to validate TF-IDF findings.

## Project Structure

```text
DataMining_project/
|-- data/
|   |-- raw/                        # Raw scraped CSV data
|   |-- processed/                  # Cleaned and tokenized datasets
|-- notebooks/
|   |-- emag_analysis.ipynb         # Main Jupyter Notebook with full pipeline
|-- src/
|   |-- scraper.py                  # Standalone eMAG web scraper and data generator
|-- requirements.txt                # Python dependencies
|-- .gitignore
|-- README.md                       # Project documentation
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
The repository comes with pre-generated data. However, to run the scraper manually to collect live data or generate a fresh dataset:
```bash
python src/scraper.py
```
*Note: Due to eMAG's anti-bot protections, the script may fall back to generating a highly realistic sample dataset if the live connection is refused.*

### 2. Execution and Analysis
The core analysis is housed in a Jupyter Notebook.
```bash
jupyter notebook notebooks/emag_analysis.ipynb
```
Execute the cells sequentially. 
*Note: Section 5 (Multilingual BERT) is computationally intensive. Running this specific section on Google Colab with GPU acceleration is recommended for large datasets.*

## Team Members
- [Your Name Here]
- [Teammate Name Here]

## License
This project is created for academic purposes for the Data Mining course at the Politehnica University of Timisoara.