"""
Script to add Word2Vec cells to the Kindle Sentiment Analysis notebook.
Run this script from the same directory as the notebook.
"""
import json

notebook_path = "Kindle_Sentiment_Analysis.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": "w2v_header",
        "metadata": {},
        "source": [
            "## Word2Vec & Average Word2Vec\n",
            "Train a Word2Vec model on the review corpus, then represent each review as the **average** of its word vectors."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_imports",
        "metadata": {},
        "outputs": [],
        "source": [
            "from gensim.models import Word2Vec\n",
            "import numpy as np"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_tokenize",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Tokenize each review into a list of words\n",
            "X_train_tokens = [text.split() for text in X_train]\n",
            "X_test_tokens = [text.split() for text in X_test]\n",
            "\n",
            "print(f'Training samples: {len(X_train_tokens)}')\n",
            "print(f'Test samples: {len(X_test_tokens)}')\n",
            "print(f'Example tokens: {X_train_tokens[0][:10]}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_train",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Word2Vec model on the training data\n",
            "w2v_model = Word2Vec(\n",
            "    sentences=X_train_tokens,\n",
            "    vector_size=100,   # Dimensionality of word vectors\n",
            "    window=5,          # Context window size\n",
            "    min_count=2,       # Ignore words appearing less than 2 times\n",
            "    workers=4,         # Parallel threads\n",
            "    epochs=20          # Training iterations\n",
            ")\n",
            "\n",
            "print(f'Vocabulary size: {len(w2v_model.wv.key_to_index)}')\n",
            "print(f'Vector size: {w2v_model.wv.vector_size}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_similar",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Quick sanity check - most similar words\n",
            "w2v_model.wv.most_similar('book')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "avg_w2v_func",
        "metadata": {},
        "outputs": [],
        "source": [
            "def avg_word2vec(tokenized_texts, model, vector_size):\n",
            "    \"\"\"\n",
            "    For each document (list of tokens), compute the average of the\n",
            "    word vectors for all tokens that exist in the Word2Vec vocabulary.\n",
            "    If none of the tokens are in the vocabulary, return a zero vector.\n",
            "    \"\"\"\n",
            "    vocab = model.wv.key_to_index\n",
            "    features = []\n",
            "    for tokens in tokenized_texts:\n",
            "        word_vectors = [model.wv[word] for word in tokens if word in vocab]\n",
            "        if len(word_vectors) > 0:\n",
            "            features.append(np.mean(word_vectors, axis=0))\n",
            "        else:\n",
            "            features.append(np.zeros(vector_size))\n",
            "    return np.array(features)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "avg_w2v_transform",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Build Average Word2Vec feature matrices\n",
            "X_train_w2v = avg_word2vec(X_train_tokens, w2v_model, 100)\n",
            "X_test_w2v = avg_word2vec(X_test_tokens, w2v_model, 100)\n",
            "\n",
            "print(f'X_train_w2v shape: {X_train_w2v.shape}')\n",
            "print(f'X_test_w2v shape: {X_test_w2v.shape}')"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_model_train",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train Gaussian Naive Bayes on Average Word2Vec features\n",
            "nb_model_w2v = GaussianNB().fit(X_train_w2v, y_train)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_predict",
        "metadata": {},
        "outputs": [],
        "source": [
            "y_pred_w2v = nb_model_w2v.predict(X_test_w2v)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "w2v_evaluate",
        "metadata": {},
        "outputs": [],
        "source": [
            "print('Word2Vec Accuracy:', accuracy_score(y_test, y_pred_w2v))\n",
            "print('\\nClassification Report (Word2Vec):')\n",
            "print(classification_report(y_test, y_pred_w2v))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": "accuracy_comparison",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compare all approaches\n",
            "print('='*50)\n",
            "print('Accuracy Comparison')\n",
            "print('='*50)\n",
            "print(f'BOW Accuracy:      {accuracy_score(y_test, y_pred_bow):.4f}')\n",
            "print(f'TF-IDF Accuracy:   {accuracy_score(y_test, y_pred_tfidf):.4f}')\n",
            "print(f'Word2Vec Accuracy: {accuracy_score(y_test, y_pred_w2v):.4f}')"
        ]
    }
]

# Remove the last empty cell and append new cells
if nb["cells"] and nb["cells"][-1]["source"] == []:
    nb["cells"] = nb["cells"][:-1]

nb["cells"].extend(new_cells)

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("✅ Successfully added 11 Word2Vec cells to the notebook!")
