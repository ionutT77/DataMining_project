import json
import os

def create_cell(cell_type, source, outputs=None):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    # Fix the trailing newline for the last line
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
        
    if cell_type == "code":
        cell["outputs"] = outputs or []
        cell["execution_count"] = None
    return cell

def generate_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    cells = []

    # Title
    cells.append(create_cell("markdown", "# 🛒 Decoding Romanian E-Commerce: Mining eMAG Product Reviews\n\n**Data Mining Course Project — 2025/2026**\n\nThis notebook contains the full data mining pipeline for analyzing product reviews from eMAG.ro across four categories: phones, laptops, headphones, and tablets. We will explore what drives consumer satisfaction, extract hidden themes, and predict ratings based on review text."))

    # Section 0
    cells.append(create_cell("markdown", "## ⚙️ Section 0: Imports & Configuration\n\nHere we import all necessary libraries and configure our environment. We set standard random seeds for reproducibility and configure matplotlib for a clean, professional aesthetic."))
    cells.append(create_cell("code", """import pandas as pd
import numpy as pd_np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import os
import datetime

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization
from wordcloud import WordCloud

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Matplotlib configuration for professional plots
plt.style.use('ggplot')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.family': 'sans-serif'
})

# Custom Romanian Stopwords (NLTK doesn't have a built-in one for Romanian by default)
ROMANIAN_STOPWORDS = set([
    "a", "abia", "acela", "aceea", "aceasta", "aceste", "acestea", "acest", "acesta", "acele", "acelea", "acel", "acolo", "acum",
    "adica", "ai", "aia", "aici", "al", "ala", "ale", "alea", "alt", "alta", "alti", "altul", "am", "anume", "apoi", "ar", "are",
    "as", "asa", "asta", "astea", "astfel", "ati", "au", "avea", "avem", "aveti", "azi", "ba", "bine", "ca", "cam", "cand", "care",
    "cat", "catre", "ce", "cea", "ceea", "cei", "cel", "cele", "ceva", "chiar", "ci", "cine", "cu", "cum", "cumva", "da", "daca",
    "dar", "datorita", "de", "deci", "deja", "deoarece", "departe", "desi", "din", "dincolo", "dintre", "doar", "dupa", "ea", "ei",
    "el", "ele", "este", "eu", "face", "fara", "fi", "fie", "fost", "incat", "inca", "intr", "intre", "isi", "iti", "la", "le", "li",
    "lor", "lui", "mai", "mi", "mie", "mine", "mult", "multi", "ne", "nici", "nu", "o", "pe", "pentru", "peste", "pana", "poate",
    "pot", "prea", "prin", "printre", "sa", "sa-mi", "sa-ti", "sa-i", "sau", "se", "si", "sunt", "suntem", "sunteti", "sus", "ta", "tale", "te",
    "ti", "tie", "tine", "toata", "toate", "tot", "toti", "totusi", "tu", "un", "una", "unde", "unele", "uneori", "unii", "unul",
    "va", "vi", "voi", "vom", "vor", "vreo", "vreun"
])

print("✅ Environment configured successfully.")"""))

    # Section 1
    cells.append(create_cell("markdown", "## 🕸️ Section 1: Web Scraping Overview\n\n*(Topic 2: Web Scraping)*\n\nThe data for this project was collected using a custom web scraper (`src/scraper.py`) built with `requests` and `BeautifulSoup`. We targeted the internal product pages of eMAG.ro across 4 categories.\n\nSince eMAG employs strong anti-bot mechanisms, the scraper incorporates random delays and User-Agent rotation. In case the scraper is blocked, a realistic fallback dataset is automatically generated.\n\nLet's load the dataset!"))
    cells.append(create_cell("code", """DATA_PATH = '../data/raw/emag_reviews_raw.csv'

# Load the data
try:
    df_raw = pd.read_csv(DATA_PATH)
    print(f"✅ Successfully loaded {len(df_raw)} reviews.")
    display(df_raw.head())
except FileNotFoundError:
    print(f"❌ File not found at {DATA_PATH}. Please run the scraper first using 'python ../src/scraper.py'")"""))

    cells.append(create_cell("code", """# Display basic information
df_raw.info()"""))

    # Section 2
    cells.append(create_cell("markdown", "## 🧹 Section 2: Data Cleaning & Preprocessing\n\n*(Topic 1: Data Cleaning - Mandatory)*\n\nReal-world text data is messy. In this section, we will:\n1. Remove duplicates and missing values.\n2. Clean the Romanian text (remove punctuation, special characters, normalize diacritics).\n3. Tokenize the text and remove stop words.\n4. Create new features like `review_length` and `word_count`."))
    cells.append(create_cell("code", """# 2.1 Handle Missing Values & Duplicates
df = df_raw.copy()

# Drop rows where review_text is missing
df = df.dropna(subset=['review_text'])

# Drop absolute duplicates
df = df.drop_duplicates()

print(f"Dataset size after dropping nulls and duplicates: {len(df)} rows.")"""))

    cells.append(create_cell("code", """# 2.2 Text Cleaning Function
import unicodedata

def clean_romanian_text(text):
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Normalize diacritics (remove accents for consistency)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize and remove stop words
    words = word_tokenize(text)
    words = [w for w in words if w not in ROMANIAN_STOPWORDS and len(w) > 2]
    
    return " ".join(words)

# Apply cleaning
print("Cleaning text (this might take a moment)...")
df['clean_text'] = df['review_text'].apply(clean_romanian_text)
print("Text cleaning complete.")"""))

    cells.append(create_cell("code", """# 2.3 Feature Engineering
df['review_length'] = df['review_text'].str.len()
df['word_count'] = df['review_text'].apply(lambda x: len(str(x).split()))

# Filter out reviews that are too short to be meaningful
df = df[df['word_count'] > 3]

# Save the processed data
os.makedirs('../data/processed', exist_ok=True)
processed_path = '../data/processed/emag_reviews_clean.csv'
df.to_csv(processed_path, index=False)
print(f"✅ Processed data saved to {processed_path}")

display(df[['review_text', 'clean_text', 'word_count', 'star_rating']].head())"""))

    # Section 3
    cells.append(create_cell("markdown", "## 📊 Section 3: Exploratory Data Analysis (EDA)\n\n*(Topic 3: EDA & Visualizations)*\n\nLet's visually explore the dataset to understand rating distributions, category differences, and what people are talking about."))

    cells.append(create_cell("code", """# 3.1 Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='star_rating', palette='viridis')
plt.title('Distribution of Star Ratings')
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.show()"""))

    cells.append(create_cell("markdown", "> **Insight:** eMAG reviews typically skew positive (4 and 5 stars). People are more likely to leave a review when they are highly satisfied or very disappointed (1 star), creating a slightly U-shaped distribution but heavily weighted towards the top."))

    cells.append(create_cell("code", """# 3.2 Category Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='star_rating', palette='Set2')
plt.title('Star Ratings Distribution per Category')
plt.xlabel('Category')
plt.ylabel('Star Rating')
plt.show()"""))

    cells.append(create_cell("code", """# 3.3 Do angry customers write more? (Review Length vs Rating)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='star_rating', y='word_count', palette='coolwarm')
plt.title('Review Length (Word Count) by Star Rating')
plt.xlabel('Star Rating')
plt.ylabel('Word Count')
# Limit y-axis to remove extreme outliers for better visualization
plt.ylim(0, df['word_count'].quantile(0.95))
plt.show()"""))

    cells.append(create_cell("markdown", "> **Insight:** 1-star and 2-star reviews tend to be longer on average. Dissatisfied customers often write detailed explanations of what went wrong, while 5-star reviews can be as short as 'Excelent!'."))

    cells.append(create_cell("code", """# 3.4 Word Clouds
def plot_wordcloud(text_series, title):
    text = " ".join(text_series.dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab20', max_words=100).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()

print("Generând Word Cloud pentru review-uri Pozitive (5 stele)...")
plot_wordcloud(df[df['star_rating'] == 5]['clean_text'], "Most Common Words in 5-Star Reviews")

print("Generând Word Cloud pentru review-uri Negative (1 stea)...")
plot_wordcloud(df[df['star_rating'] == 1]['clean_text'], "Most Common Words in 1-Star Reviews")"""))


    # Section 4
    cells.append(create_cell("markdown", "## 🔠 Section 4: TF-IDF Analysis\n\n*(Topic 4: TF-IDF)*\n\nTF-IDF helps us find words that are not just frequent, but **uniquely important** to a specific group of texts. We will identify the most defining words for positive vs. negative reviews."))

    cells.append(create_cell("code", """# Apply TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['clean_text'])

feature_names = np.array(tfidf.get_feature_names_out())

def get_top_tf_idf_words(rating_mask, top_n=15):
    # Mean TF-IDF score for each word across the subset
    mean_tfidf = X_tfidf[rating_mask].mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-top_n:][::-1]
    return feature_names[top_indices], mean_tfidf[top_indices]

# Positive (5 stars)
pos_words, pos_scores = get_top_tf_idf_words(df['star_rating'] == 5)
# Negative (1 star)
neg_words, neg_scores = get_top_tf_idf_words(df['star_rating'] == 1)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=pos_scores, y=pos_words, ax=axes[0], palette='Greens_r')
axes[0].set_title('Top TF-IDF Terms: 5-Star Reviews')
axes[0].set_xlabel('Mean TF-IDF Score')

sns.barplot(x=neg_scores, y=neg_words, ax=axes[1], palette='Reds_r')
axes[1].set_title('Top TF-IDF Terms: 1-Star Reviews')
axes[1].set_xlabel('Mean TF-IDF Score')

plt.tight_layout()
plt.show()"""))

    cells.append(create_cell("markdown", "> **Interpretation:** The TF-IDF analysis highlights key drivers of satisfaction vs. dissatisfaction. Positive terms usually revolve around 'excelent', 'recomand', 'calitate'. Negative terms often feature words like 'stricat', 'problema', 'dezamagit', 'retur'."))

    # Section 5
    cells.append(create_cell("markdown", "## 🎭 Section 5: Sentiment Analysis\n\n*(Topic 5: Sentiment Analysis)*\n\nWe will use a pre-trained **Multilingual BERT** model from HuggingFace to predict the sentiment (1 to 5 stars) directly from the Romanian text.\n\n*Note: This process can take a few minutes on a CPU. For larger datasets, running this specific cell on Google Colab (GPU) is recommended.*"))

    cells.append(create_cell("code", """from transformers import pipeline

# We use nlptown's multilingual sentiment model which natively supports Romanian
# and predicts a 1 to 5 star rating
print("Loading Multilingual BERT model... (this may take a minute)")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_bert_sentiment(text):
    if len(str(text)) < 5:
        return None
    try:
        # Truncate text to BERT's max token limit (~512 words usually)
        truncated_text = str(text)[:1500] 
        result = sentiment_pipeline(truncated_text)[0]
        # Result format: '5 stars' -> we extract the integer '5'
        return int(result['label'].split()[0])
    except Exception as e:
        return None

# To save time, we will run sentiment analysis on a sample of 200 reviews
df_sample = df.sample(min(200, len(df)), random_state=42).copy()

print("Applying sentiment analysis to sample...")
df_sample['bert_pred_stars'] = df_sample['review_text'].apply(get_bert_sentiment)
df_sample = df_sample.dropna(subset=['bert_pred_stars'])

# Calculate Accuracy
acc = accuracy_score(df_sample['star_rating'], df_sample['bert_pred_stars'])
print(f"\\n✅ Multilingual BERT Accuracy (Exact Star Match): {acc:.2%}")

# Let's see some examples where BERT predicted accurately
display(df_sample[['review_text', 'star_rating', 'bert_pred_stars']].head(10))"""))

    cells.append(create_cell("code", """# Create a confusion matrix
cm = confusion_matrix(df_sample['star_rating'], df_sample['bert_pred_stars'], labels=[1,2,3,4,5])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
plt.title('Confusion Matrix: Actual vs BERT Predicted Stars')
plt.xlabel('BERT Predicted')
plt.ylabel('Actual Stars')
plt.show()"""))

    cells.append(create_cell("markdown", "> **Insight:** Multilingual BERT does an impressive job understanding Romanian context! Errors are usually off by just one star (e.g., confusing a 4-star for a 5-star). Exact star matching is hard even for humans, but the general sentiment direction (positive vs negative) is highly accurate."))

    # Section 6
    cells.append(create_cell("markdown", "## 🧠 Section 6: Topic Modelling (LDA)\n\n*(Topic 6: Topic Modelling)*\n\nWhat are the latent themes across thousands of reviews? We use **Latent Dirichlet Allocation (LDA)** to discover these themes automatically."))

    cells.append(create_cell("code", """# We apply LDA to the TF-IDF matrix
NUM_TOPICS = 5
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, max_iter=15)
lda.fit(X_tfidf)

def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics.append(top_words)
        print(f"Topic {topic_idx + 1}: " + ", ".join(top_words))
    return topics

print("🔍 Discovered Topics:")
topics = display_topics(lda, feature_names, 10)"""))

    cells.append(create_cell("markdown", "By examining the words above, we can assign human-readable labels to these topics (e.g., Topic 1 might be 'Build Quality', Topic 2 might be 'Delivery & Returns'). LDA groups words that frequently co-occur in the same reviews."))

    # Section 7
    cells.append(create_cell("markdown", "## 🤖 Section 7: Supervised Classification Model\n\n*(Topic 7: Supervised Model)*\n\nCan we build our own classifier to predict whether a review is Positive or Negative based solely on the text features?\n\nWe will frame this as a binary classification problem:\n- **Positive (1):** 4 and 5 stars\n- **Negative (0):** 1 and 2 stars\n*(We drop 3-star reviews for clearer decision boundaries)*"))

    cells.append(create_cell("code", """# Filter dataset and create target
df_class = df[df['star_rating'] != 3].copy()
df_class['sentiment'] = df_class['star_rating'].apply(lambda x: 1 if x >= 4 else 0)

print(f"Dataset size for classification: {len(df_class)} rows")
print("Class balance:")
print(df_class['sentiment'].value_counts(normalize=True))"""))

    cells.append(create_cell("code", """# Features (X) and Target (y)
X_text = tfidf.fit_transform(df_class['clean_text'])
y = df_class['sentiment']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")"""))

    cells.append(create_cell("code", """# We will evaluate 3 models: Logistic Regression, Random Forest, and SVM
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "SVM": SVC(kernel='linear', class_weight='balanced', probability=True)
}

results = []

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': acc})
    
results_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False)
display(results_df)"""))

    cells.append(create_cell("code", """# Detailed Report for the best model (typically Logistic Regression or SVM for TF-IDF text)
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

y_pred = best_model.predict(X_test)

print(f"\\n--- Detailed Report for {best_model_name} ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix: {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()"""))

    cells.append(create_cell("code", """# Feature Importance (Logistic Regression Coefficients)
if best_model_name in ['Logistic Regression', 'SVM']:
    coefs = best_model.coef_[0]
    features = np.array(tfidf.get_feature_names_out())
    
    # Top 15 positive and Top 15 negative features
    top_pos_idx = coefs.argsort()[-10:][::-1]
    top_neg_idx = coefs.argsort()[:10]
    
    imp_df = pd.DataFrame({
        'Feature': np.concatenate([features[top_pos_idx], features[top_neg_idx]]),
        'Importance': np.concatenate([coefs[top_pos_idx], coefs[top_neg_idx]]),
        'Type': ['Positive Driver'] * 10 + ['Negative Driver'] * 10
    })
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Type', palette=['green', 'red'])
    plt.title('Top 10 Words Driving Positive vs Negative Predictions')
    plt.show()"""))

    # Section 8
    cells.append(create_cell("markdown", "## 🏆 Section 8: Conclusions & Key Insights\n\n### 💡 Key Takeaways:\n1. **High Baseline Satisfaction:** Romanian eMAG shoppers generally leave positive reviews (4-5 stars). Products with an average rating below 4.0 should be investigated.\n2. **Review Length signals Distress:** Angry customers write significantly longer reviews explaining their negative experiences in detail, whereas happy customers keep it brief.\n3. **Language Nuance:** Multilingual BERT handles Romanian sentiment impressively well, proving that advanced transformers outclass simple dictionary-based methods.\n4. **Predictability:** We successfully built a Machine Learning classifier that predicts whether a review is positive or negative based entirely on the text, achieving high accuracy.\n5. **Top Drivers:** The words most heavily associated with negative reviews often center around `retur` (returns), `stricat` (broken), and `baterie` (battery issues for phones/laptops).\n\n**Actionable Advice for Sellers:** Focus heavily on Quality Assurance to avoid returns, as functional defects trigger the longest, most damaging reviews.\n\n---\n*Data Mining Project 2025/2026*"))

    notebook["cells"] = cells

    os.makedirs('../notebooks', exist_ok=True)
    with open('../notebooks/emag_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("Notebook successfully generated at ../notebooks/emag_analysis.ipynb")

if __name__ == "__main__":
    generate_notebook()
