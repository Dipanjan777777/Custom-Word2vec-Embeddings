# SMS Spam Detection with Custom Word2Vec Embeddings

This project demonstrates SMS spam classification using custom-trained Word2Vec embeddings. The workflow leverages the Gensim library to train word vectors from scratch on the SMS dataset, then uses these embeddings as features for various machine learning models.

## Highlights: Custom Word2Vec Embeddings

- **Why Custom Word2Vec?**  
  Training Word2Vec embeddings on your own SMS dataset ensures that the word vectors capture the specific semantics, slang, and context unique to SMS messages, which may not be present in generic pretrained models.

- **Text Preprocessing:**  
  SMS messages are cleaned, tokenized, and lemmatized to prepare for embedding.  
  - Lowercasing, punctuation removal, stopword filtering, and lemmatization are applied for optimal results.

- **Word2Vec Training:**  
  The cleaned corpus is used to train a Word2Vec model using Gensim.  
  - The model learns vector representations for each word based on its context within the SMS dataset.
  - You can inspect vocabulary, vector shapes, and similarity queries to understand the learned relationships.

- **Feature Extraction:**  
  Each message is represented by the average of its word vectors, resulting in a fixed-size feature vector for each SMS.  
  - This approach transforms variable-length text into consistent numerical features for machine learning.
  - Out-of-vocabulary words are ignored, ensuring robust feature generation.

- **Advantages of Custom Embeddings:**  
  - Captures domain-specific language and abbreviations common in SMS.
  - Improves downstream model performance compared to generic embeddings.
  - Enables interpretability through similarity queries and vector arithmetic.

- **Model Comparison:**  
  Multiple classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes, KNN, AdaBoost, XGBoost) are trained and evaluated using the custom Word2Vec features.

- **Performance Metrics:**  
  Accuracy, precision, and recall are reported for each model to compare effectiveness.

## Environment Setup

1. **Python Version:**  
   Use Python 3.8 or newer for best compatibility.

2. **Create a Virtual Environment (Recommended):**  
   ```
   python -m venv venv
   ```
   Activate the environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

## Setup Process

1. **Clone the Repository:**  
   Download or clone this project to your local machine.

2. **Install Dependencies:**  
   Install required Python packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Download Pretrained Word2Vec (Optional):**  
   If you want to use Google's pretrained vectors, download `GoogleNews-vectors-negative300.bin` and place it in the `model` directory.

4. **Run the Notebook:**  
   Open `sms_spam_using_word2vec.ipynb` in Jupyter or VS Code and execute the cells step by step.

5. **Dataset:**  
   Ensure `spam.csv` is present in the project directory.

---

For any issues or questions, please refer to the notebook comments or open an issue.
