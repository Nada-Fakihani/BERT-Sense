# SentBERT-NLP: BERT-powered Sentiment Analysis

## 📌 Project Overview
This project applies **Natural Language Processing (NLP)** and **Deep Learning** to analyze sentiment in tweets using the **SMILE Twitter Emotion dataset**. The goal is to classify emotions expressed in tweets related to museum experiences using **BERT-based models** and traditional Machine Learning approaches.

## 📊 Dataset: SMILE Twitter Emotion Dataset
- **Source:** SMILE Project ([culturesmile.org](http://www.culturesmile.org))
- **Size:** 3,085 tweets
- **Emotions:** Anger, Disgust, Happiness, Surprise, Sadness
- **Timeframe:** Collected between May 2013 and June 2015
- **License:** CC-BY for annotations, Twitter retains ownership of tweets

## 🚀 Features & Methodology
- **Exploratory Data Analysis (EDA)**: Understanding data distribution, cleaning, and preprocessing.
- **Text Tokenization & Encoding**: Using **BERT tokenizer** for embedding representations.
- **Model Training**: Comparing deep learning (**BERT**) with classical ML models (Random Forest, SVM, XGBoost, Logistic Regression, LightGBM, Stacking).
- **Evaluation Metrics**: Accuracy, F1-score, and Confusion Matrix for performance comparison.
- **Hyperparameter Tuning**: Optimization for best results.

## 🛠️ Tech Stack
- **Programming Language**: Python 🐍
- **Deep Learning Framework**: TensorFlow / PyTorch
- **NLP Libraries**: Hugging Face Transformers, NLTK, SpaCy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: BERT, SVM, Random Forest, XGBoost, LightGBM, Stacking Classifier

## 🔧 Installation
To run the project, install the dependencies:
```bash
pip install -r requirements.txt
```

## 📂 Project Structure
```
SentBERT-NLP/
│-- data/                  # Dataset files (not included in repo)
│-- notebooks/             # Jupyter notebooks for analysis
│-- src/                   # Python scripts for preprocessing & training
│-- models/                # Saved trained models
│-- README.md              # Project documentation
│-- requirements.txt       # Dependencies
```

## 🎯 How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SentBERT-NLP.git
   ```
2. Navigate into the project directory:
   ```bash
   cd SentBERT-NLP
   ```
3. Run the notebook or scripts for training and evaluation.

## 📈 Results & Findings
- **Model Performance:**
  - **BERT Accuracy:** 77%  
  - **Random Forest Accuracy:** 81%  
  - **SVM Accuracy:** 80%  
  - **LightGBM Accuracy:** 85%  
  - **Stacking Accuracy:** 83%  
- **Challenges:**
  - Some emotions (e.g., **surprise, disgust**) were harder to classify due to data imbalance.
  - Feature engineering and hyperparameter tuning improved ML model performance but still lagged behind deep learning.
  - UndefinedMetricWarnings were encountered due to missing predicted samples for certain classes.

## 📌 Future Improvements
- Fine-tuning BERT with more training epochs and larger datasets.
- Implementing an **interactive web app** using Streamlit or Flask.
- Expanding the dataset with real-time Twitter scraping.
- Addressing class imbalance by data augmentation or better weighting strategies.

## 🏆 Credits & Citation
Dataset by Bo Wang, Adam Tsakalidis, Maria Liakata, Arkaitz Zubiaga, Rob Procter, and Eric Jensen.

📢 **Feel free to contribute!** Fork, star, and submit pull requests. 🚀
