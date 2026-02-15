# Introduction to Natural Language Processing (NLP)

A comprehensive hands-on tutorial series covering fundamental NLP concepts and techniques using Python. This collection of Jupyter notebooks provides practical, executable examples for learning text processing, analysis, and machine learning applications.

## 📚 Overview

This project offers a structured learning path through 13 interactive notebooks, progressing from basic text preprocessing to advanced machine learning classification. Each notebook includes detailed explanations, code examples, and real-world applications.

## 🎯 Learning Objectives

By completing this tutorial series, you will learn to:

- **Preprocess text data** using industry-standard techniques
- **Apply regular expressions** for pattern matching and text cleaning
- **Tokenize** documents into sentences and words
- **Normalize text** using stemming and lemmatization
- **Extract n-grams** for context-aware analysis
- **Tag parts of speech** to understand grammatical structure
- **Identify named entities** (people, organizations, locations)
- **Analyze sentiment** using multiple approaches
- **Leverage transformer models** (BERT, DistilBERT) via HuggingFace
- **Vectorize text** using Bag-of-Words and TF-IDF
- **Discover topics** in document collections using LDA and LSI
- **Build text classifiers** using machine learning algorithms

## 📋 Prerequisites

### Required Knowledge
- **Python basics:** Variables, loops, functions, data structures
- **Pandas fundamentals:** DataFrames, basic data manipulation
- **Basic machine learning concepts** (helpful but not required)

### Software Requirements
- **Python:** 3.7 or higher
- **Jupyter Notebook or JupyterLab**
- **Required Libraries:**
  ```
  pandas
  numpy
  matplotlib
  nltk
  spacy
  textblob
  vaderSentiment
  transformers (HuggingFace)
  scikit-learn
  gensim
  ```

## 🚀 Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone <repository-url>
cd Introduction-to-NLP
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv nlp_env
nlp_env\Scripts\activate

# Mac/Linux
python3 -m venv nlp_env
source nlp_env/bin/activate
```

### 3. Install Required Packages
```bash
pip install pandas numpy matplotlib nltk spacy textblob vaderSentiment transformers scikit-learn gensim
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

### 5. Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

### 6. Launch Jupyter Notebook
```bash
jupyter notebook
```

## 📖 Notebook Descriptions

### **01 - Text Preprocessing**
**Concepts:** Lowercasing, stop word removal  
**Libraries:** NLTK  
**Skills:** Basic text cleaning, custom stop word lists  
**Use Cases:** Preparing text for analysis, noise reduction

**What you'll learn:**
- Why preprocessing is critical for NLP
- Lowercasing to normalize text
- Removing stop words (common, uninformative words)
- Creating custom stop word lists for domain-specific text

---

### **02 - Regular Expressions**
**Concepts:** Pattern matching, text cleaning  
**Libraries:** Python `re` module  
**Skills:** Regex syntax, search, replace, anchors  
**Use Cases:** Email extraction, punctuation removal, pattern detection

**What you'll learn:**
- Core regex syntax (`.`, `*`, `+`, `?`, `[]`, `^`, `$`, `|`)
- `re.search()` vs `re.match()` vs `re.findall()`
- Practical patterns for cleaning real-world text
- Extracting structured information (emails, phone numbers, URLs)

---

### **03 - Tokenization**
**Concepts:** Sentence segmentation, word tokenization  
**Libraries:** NLTK  
**Skills:** `sent_tokenize()`, `word_tokenize()`  
**Use Cases:** Document splitting, feature extraction

**What you'll learn:**
- Difference between sentence and word tokenization
- Why `.split()` is insufficient for real text
- Handling contractions and punctuation
- Preparing tokenized text for downstream tasks

---

### **04 - Stemming and Lemmatization**
**Concepts:** Word normalization  
**Libraries:** NLTK (PorterStemmer, WordNetLemmatizer)  
**Skills:** Root form extraction, morphological analysis  
**Use Cases:** Search engines, text classification, vocabulary reduction

**What you'll learn:**
- **Stemming:** Fast, rule-based suffix removal (PorterStemmer)
- **Lemmatization:** Dictionary-based reduction to base form
- When to use stemming vs lemmatization
- Impact on downstream NLP tasks
- Comparison table and practical examples

---

### **05 - N-grams**
**Concepts:** Word sequences, context modeling  
**Libraries:** NLTK  
**Skills:** Unigrams, bigrams, trigrams, frequency analysis  
**Use Cases:** Phrase detection, language modeling, feature engineering

**What you'll learn:**
- What n-grams are and why they capture context
- Generating unigrams (1 word), bigrams (2 words), trigrams (3 words)
- Frequency analysis of n-grams
- Trade-offs: vocabulary explosion vs context capture
- Applications in text classification and search

---

### **06 - Parts of Speech (POS) Tagging**
**Concepts:** Grammatical tagging  
**Libraries:** spaCy  
**Skills:** POS identification, frequency analysis  
**Use Cases:** Information extraction, grammar checking, feature engineering

**What you'll learn:**
- Understanding POS tags (NOUN, VERB, ADJ, etc.)
- Using spaCy for accurate POS tagging
- Extracting and analyzing POS frequencies
- Filtering text by POS (e.g., extracting only nouns)
- Applications in NLP pipelines

---

### **07 - Named Entity Recognition (NER)**
**Concepts:** Entity extraction and classification  
**Libraries:** spaCy, displaCy  
**Skills:** Identifying PERSON, ORG, GPE, DATE entities  
**Use Cases:** Information extraction, knowledge graphs, question answering

**What you'll learn:**
- What named entities are (people, places, organizations, dates)
- Using spaCy's pre-trained NER model
- Visualizing entities with displaCy
- Importance of proper capitalization and preprocessing
- Real-world applications (news analysis, chatbots, search)

---

### **08 - Sentiment Analysis**
**Concepts:** Opinion mining, polarity detection  
**Libraries:** TextBlob, VADER  
**Skills:** Polarity scoring, compound scores, negation handling  
**Use Cases:** Review analysis, social media monitoring, customer feedback

**What you'll learn:**
- **TextBlob:** Simple polarity and subjectivity scores
- **VADER:** Social media-optimized sentiment analysis
- Comparing TextBlob vs VADER (strengths and weaknesses)
- Handling negations, intensifiers, and emojis
- When to use rule-based vs ML-based sentiment analysis
- Interpreting scores and choosing the right tool

---

### **09 - Transformer Models**
**Concepts:** Attention mechanism, transfer learning, pre-trained models  
**Libraries:** HuggingFace Transformers  
**Skills:** Using pipelines, fine-tuning, model comparison  
**Use Cases:** State-of-the-art sentiment analysis, classification, NER

**What you'll learn:**
- What transformers are and why they revolutionized NLP
- Understanding the attention mechanism
- Using HuggingFace pipelines for quick inference
- Comparing models: BERT, DistilBERT, BERTweet
- When to use transformers vs traditional models
- Trade-offs: accuracy vs computational cost

---

### **10 - Text Vectorization (Bag-of-Words)**
**Concepts:** Converting text to numerical features  
**Libraries:** scikit-learn (CountVectorizer)  
**Skills:** BoW representation, sparse matrices  
**Use Cases:** Text classification, clustering, similarity

**What you'll learn:**
- Why ML models need numerical input
- Bag-of-Words concept and limitations
- Using `CountVectorizer` to create BoW matrices
- Understanding sparse matrix representation
- Parameters: `max_features`, `ngram_range`, `min_df`, `max_df`
- When BoW is sufficient and when to use TF-IDF

---

### **11 - TF-IDF (Term Frequency-Inverse Document Frequency)**
**Concepts:** Weighted text representation  
**Libraries:** scikit-learn (TfidfVectorizer)  
**Skills:** TF-IDF calculation, feature importance  
**Use Cases:** Document ranking, search engines, classification

**What you'll learn:**
- **TF (Term Frequency):** How often a word appears in a document
- **IDF (Inverse Document Frequency):** How unique a word is across documents
- Mathematical formula: TF-IDF = TF × IDF
- Why TF-IDF is better than BoW for many tasks
- Interpreting TF-IDF scores
- Comparison to CountVectorizer
- Parameters and tuning tips

---

### **12 - Topic Modeling**
**Concepts:** Unsupervised topic discovery  
**Libraries:** Gensim (LDA, LSI)  
**Skills:** LDA, LSA, coherence optimization, topic interpretation  
**Use Cases:** Document organization, trend analysis, recommendation

**What you'll learn:**
- What topic modeling is and how it works
- **LDA (Latent Dirichlet Allocation):** Probabilistic topic modeling
- **LSA/LSI (Latent Semantic Analysis):** Matrix factorization approach
- Preprocessing pipeline for topic modeling
- Creating dictionary and document-term matrix
- Determining optimal number of topics using coherence scores
- Interpreting and labeling topics
- Real-world applications and best practices

---

### **13 - Building a Text Classifier**
**Concepts:** Supervised machine learning for text  
**Libraries:** scikit-learn (Logistic Regression, Naive Bayes, SVM)  
**Skills:** Train-test split, model evaluation, comparison  
**Use Cases:** Spam detection, sentiment classification, content categorization

**What you'll learn:**
- End-to-end ML pipeline for text classification
- Preprocessing and vectorization
- Training three classifiers:
  - **Logistic Regression:** Fast, interpretable baseline
  - **Naive Bayes:** Probabilistic, optimized for text
  - **SVM:** Maximum accuracy with margin optimization
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrix interpretation
- Model comparison and selection criteria
- Hyperparameter tuning and cross-validation
- Best practices and common pitfalls

---

## 🗺️ Recommended Learning Path

### **Beginner Track (Fundamentals)**
1. 01 - Text Preprocessing
2. 02 - Regular Expressions
3. 03 - Tokenization
4. 04 - Stemming and Lemmatization
5. 05 - N-grams

**Outcome:** Solid foundation in text preprocessing and basic NLP concepts.

---

### **Intermediate Track (Analysis & Features)**
6. 06 - Parts of Speech (POS) Tagging
7. 07 - Named Entity Recognition (NER)
8. 08 - Sentiment Analysis
9. 10 - Text Vectorization (Bag-of-Words)
10. 11 - TF-IDF

**Outcome:** Ability to extract insights and convert text to ML features.

---

### **Advanced Track (Modeling)**
11. 09 - Transformer Models
12. 12 - Topic Modeling
13. 13 - Building a Text Classifier

**Outcome:** Build production-ready NLP models using both traditional ML and deep learning.

---

## 💡 Tips for Success

### **Learning Strategies:**
1. **Execute Every Cell:** Don't just read—run the code and observe outputs
2. **Experiment:** Modify parameters and see what changes
3. **Use Your Own Data:** Apply techniques to text you care about
4. **Take Notes:** Document insights and findings in markdown cells
5. **Build Projects:** Combine techniques to solve real problems

### **Common Challenges:**
- **Library Installation:** Use virtual environments to avoid conflicts
- **Data Downloads:** Ensure NLTK and spaCy models are downloaded
- **Memory Issues:** Large datasets may require batching or sampling
- **Model Training:** Start small, then scale up as you understand the process

### **Resources:**
- **NLTK Documentation:** https://www.nltk.org/
- **spaCy Documentation:** https://spacy.io/
- **HuggingFace Documentation:** https://huggingface.co/docs
- **scikit-learn Documentation:** https://scikit-learn.org/
- **Gensim Documentation:** https://radimrehurek.com/gensim/

---

## 📊 Datasets Included

- **news_articles.csv:** Sample news articles for topic modeling and classification

### **Where to Find More Data:**
- **Kaggle Datasets:** https://www.kaggle.com/datasets (text classification, sentiment, NER)
- **UCI ML Repository:** https://archive.ics.uci.edu/ml/index.php
- **HuggingFace Datasets:** https://huggingface.co/datasets
- **Twitter API:** For social media sentiment analysis
- **Reddit API (PRAW):** For community text analysis

---

## 🛠️ Troubleshooting

### **Issue: NLTK data not found**
```python
import nltk
nltk.download('all')  # Download all NLTK data (may take time)
```

### **Issue: spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

### **Issue: HuggingFace model download fails**
- Check internet connection
- Use smaller models first (e.g., `distilbert-base-uncased`)
- Set cache directory: `transformers.set_cache_dir('path/to/cache')`

### **Issue: Jupyter kernel crashes with large datasets**
- Reduce dataset size (sample rows)
- Use batch processing
- Increase RAM or use cloud computing (Google Colab, AWS)

---

## 🚀 Next Steps & Project Ideas

### **Beginner Projects:**
1. **Email Spam Classifier:** Use Naive Bayes with TF-IDF
2. **Movie Review Sentiment Analysis:** TextBlob or VADER
3. **Twitter Hashtag Analyzer:** Extract and visualize trending topics

### **Intermediate Projects:**
4. **News Article Categorizer:** Multi-class classification with SVM
5. **Resume Parser:** NER to extract names, skills, education
6. **Customer Feedback Dashboard:** Sentiment + topic modeling

### **Advanced Projects:**
7. **Chatbot with Intent Recognition:** Transformers + classification
8. **Fake News Detector:** Ensemble models with linguistic features
9. **Document Summarization:** Extractive or abstractive using transformers
10. **Question Answering System:** BERT-based QA on custom documents

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome! If you find issues or have ideas for additional notebooks, please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear descriptions

---

## 📜 License

This project is open-source and available for educational purposes. Please cite this repository if you use it in your work or research.

---

## 🙏 Acknowledgments

This tutorial series leverages several excellent open-source libraries:

- **NLTK** by the Natural Language Toolkit Project
- **spaCy** by Explosion AI
- **HuggingFace Transformers** by HuggingFace
- **scikit-learn** by the scikit-learn developers
- **Gensim** by Radim Řehůřek
- **TextBlob** and **VADER** sentiment analysis libraries

---

## 📧 Contact & Support

For questions, suggestions, or collaboration:

- **Issues:** Open an issue on GitHub
- **Discussions:** Use the repository discussions page
- **Email:** [Your contact email]

---

## 🎓 Additional Learning Resources

### **Books:**
- *Speech and Language Processing* by Jurafsky & Martin
- *Natural Language Processing with Python* (NLTK book)
- *Applied Text Analysis with Python* by Bengfort, Bilbro & Ojeda

### **Online Courses:**
- **Coursera:** Natural Language Processing Specialization (deeplearning.ai)
- **fast.ai:** Practical Deep Learning for Coders (NLP sections)
- **Stanford CS224N:** NLP with Deep Learning

### **Communities:**
- **Reddit:** r/LanguageTechnology, r/MachineLearning
- **Stack Overflow:** [nlp tag](https://stackoverflow.com/questions/tagged/nlp)
- **HuggingFace Forums:** https://discuss.huggingface.co/

---

**Happy Learning! 🎉**

Master the fundamentals, experiment with real data, and build amazing NLP applications!
