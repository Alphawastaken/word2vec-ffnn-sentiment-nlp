# 💡 Tweet Sentiment Classifier with Word2Vec & FFNN


## 📋 Table of Contents

* [Overview](#📝-overview)
* [Prerequisites](#✅-prerequisites)
* [Project Structure](#📂-project-structure)
* [Pipeline Walkthrough](#🔄-pipeline-walkthrough)
* [Results & Visuals](#📊-results--visuals)
* [Usage](#🚀-usage)
* [License](#📜-license)
* [Contact](#📬-contact)

---

## 📝 Overview

A tweet sentiment classification pipeline using:

* **Word2Vec embeddings** trained on your dataset
* A **3-layer Feedforward Neural Network (FFNN)** in PyTorch
* Advanced preprocessing: emoji handling, contractions, negations, repeated characters
* Comprehensive visual reporting: word clouds, t-SNE projections, learning/ROC curves

---

## ✅ Prerequisites

Install required dependencies:

```bash
pip install pandas numpy gensim torch scikit-learn matplotlib seaborn textblob emoji contractions
```

The notebook also downloads NLTK’s `punkt` tokenizer automatically.

---

## 📂 Project Structure

```
tweet-sentiment-ffnn/
├── sentiment_ffnn.ipynb       # Main Jupyter notebook
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT license
```

---

## 🔄 Pipeline Walkthrough

**Setup & Imports**

* Loads essential libraries and installs text cleaning tools
* Sets a fixed seed for reproducibility

**Data Loading**

* Reads `train_dataset.csv`, `val_dataset.csv`, and `test_dataset.csv` into DataFrames

**Text Preprocessing**

* Reduces character repetition ("soooo" → "soo")
* Handles negations by prefixing subsequent words with `NOT_` ("not happy" → "NOT\_happy")
* Converts emojis to text, expands contractions, removes noise
* Tokenizes cleaned text

**Word2Vec Embedding**

* Trains a skip-gram Word2Vec model (vector\_size = 400)
* Computes average embedding per tweet

**PyTorch Dataset & DataLoader**

* Wraps embeddings and labels for batch processing during training

**Model Definition**

* Defines a 3-layer FFNN with BatchNorm, Dropout, class-weighted CrossEntropy, AdamW optimizer, and LR scheduler

**Training Loop**

* Runs for 25 epochs, tracking training/validation losses and metrics
* Saves the best model based on validation accuracy

**Evaluation & Submission**

* Loads the best model and predicts on the test set
* Saves `submission.csv` containing `ID` & `Label` for submission

**Visualization Module**

* Automatically generates and saves:

  * Word clouds (raw vs. cleaned)
  * Token-length histograms
  * Sentiment polarity distribution (TextBlob)
  * Class balance bar plots
  * t-SNE embeddings visualization
  * Learning rate comparison & learning/ROC curves
  * Confusion matrix and optimizer comparison plots

---

## 📊 Results & Visuals

**Validation Performance**

* Accuracy ≈ 0.795
* Precision ≈ 0.813
* Recall ≈ 0.774
* F1‏-Score ≈ 0.793

**Sample Visual Outputs:**

* Raw word cloud
* Cleaned word cloud
* t‏-SNE embeddings
* Loss & accuracy curves
* ROC curve (AUC displayed)

---

## 🚀 Usage

**Clone the repo**

```bash
git clone https://github.com/Alphawastaken/word2vec-ffnn-sentiment-nlp.git
cd word2vec-ffnn-sentiment-nlp
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the Notebook**

```bash
jupyter notebook word2vec-ffnn-sentiment-nlp
```

Execute all cells to preprocess data, train the model, evaluate performance, and generate visuals.

**Results & Outputs**

* Evaluation metrics are printed within the notebook
* Visuals saved to `notebook_images/`
* Predictions exported in `submission.csv`

**Optionally Customize**

* Modify preprocessing steps or embedding parameters
* Experiment with model hyperparameters
* Analyze visuals for insights and improvements

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for full terms.

---

## 📬 Contact

GitHub: \[Alphawastaken]
Open to feedback, improvements, or research collaborations 😊
