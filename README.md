# Kindle Review Sentiment Analysis

A notebook comparing bag-of-words, TF-IDF, and averaged Word2Vec representations for predicting review ratings from Kindle review text.

## Workflow

1. Load `all_kindle_review.csv` and select `reviewText` and `rating`.
2. Lowercase text, remove special characters, stopwords, URLs, HTML, and extra whitespace, then lemmatize words.
3. Split reviews into training and test sets (80/20).
4. Train Gaussian Naive Bayes models on bag-of-words and TF-IDF features.
5. Train a 100-dimensional Word2Vec model on training reviews, average word vectors, and fit another Gaussian Naive Bayes model.
6. Compare accuracy and display a Word2Vec classification report.

The notebook predicts the original rating labels. It does not convert ratings into binary positive/negative sentiment labels.

## Files

- `Kindle_Sentiment_Analysis.ipynb`: preprocessing, training, and evaluation.
- `all_kindle_review.csv`: input dataset included in the repository.
- `add_word2vec_cells.py`: utility that edits a notebook to add Word2Vec cells; it contains a machine-specific path and is unnecessary for running the existing notebook.

## Run locally

Use Python 3 in a virtual environment:

```bash
git clone https://github.com/Sanskar1314/Kindle_Review_Sentiment_Analysis.git
cd Kindle_Review_Sentiment_Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab numpy pandas nltk scikit-learn gensim beautifulsoup4 lxml
jupyter lab
```

On Windows, activate with `.venv\Scripts\activate`. Launch Jupyter from the repository root and run the notebook cells in order.

Download NLTK resources before preprocessing:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

## Reproduction notes

The notebook converts document-term matrices to dense arrays, which can use considerable memory. Train/test splitting has no fixed random seed, so measured scores may vary. Results are displayed inside the notebook; no deployed service or saved prediction pipeline is included.
