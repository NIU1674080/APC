# Sentiment Analysis Project

## Dataset Overview

The dataset used in this project is a collection of tweets, each labeled with a sentiment: **positive**, **neutral**, or **negative**. The dataset is structured with the following key columns:

| Column Name    | Description                                                |
|----------------|------------------------------------------------------------|
| `sentiment`    | Sentiment label (negative, neutral, positive)              |
| `text`         | The tweet's raw text                                       |
| `selected_text`| Annotated part of the tweet relevant to the sentiment       |
| `textID`       | Unique identifier for each tweet                           |

If we delete the ID column, this is how it looks:

| sentiment  | text                                                       | selected_text                                           |
|------------|-------------------------------------------------------------|---------------------------------------------------------|
| positive   | "I love the weather today, it's amazing!"                   | "love the weather today"                               |
| neutral    | "This is a neutral tweet with no clear sentiment."          | "This is a neutral tweet"                              |
| negative   | "I am so upset with the service I received."                | "upset with the service"                               |
| positive   | "I am so happy about my recent achievements!"               | "happy about my recent achievements"                   |
| neutral    | "The event was okay, nothing extraordinary."                | "The event was okay"                                   |


## Project Outline

This project follows a clear series of steps to preprocess the data, extract features, and build sentiment classification models:

1. **Data Cleaning and Preprocessing**:
   - Removal of unnecessary columns (`textID`, `selected_text`).
   - Handling missing values in the `text` column.
   - Text cleaning using regular expressions to remove URLs, user mentions, and special characters.
   - Tokenization and removal of stopwords from the tweet text.

2. **Feature Extraction**:
   - **Bag of Words (BoW)** and **TF-IDF** vectorization techniques are applied to transform the tweet texts into numerical features for sentiment classification.

3. **Model Training**:
   - Several machine learning models (Logistic Regression, Naive Bayes, etc.) are trained on the cleaned and vectorized data.
   - Hyperparameter tuning is performed to optimize model performance.

4. **Model Evaluation**:
   - The models are evaluated using classification metrics like **F1-Score**, **Jaccard Score**, and confusion matrices.
   - Cross-validation is performed to validate the models' robustness.

5. **Results Visualization**:
   - Visualization of the most frequent words for each sentiment using **WordClouds**.
   - Donut plots to show the most common words for each sentiment in both selected and predicted text.

## Expected Results

The expected results from this project are as follows:
- The models should be able to predict the sentiment (positive, neutral, or negative) of tweets with a reasonable accuracy.
- Visualizations should provide insights into the most common words associated with each sentiment.
- The Jaccard score distribution across different sentiments should highlight the overlap between predicted and selected text.
- The comparison between the number of words in selected versus predicted text should give insight into how the models are interpreting tweet content.

## Dependencies

This project requires the following Python packages:

- **kagglehub**: For downloading the dataset.
- **emoji**: For converting emojis to text representations.
- **matplotlib**: For plotting graphs.
- **numpy**: For numerical operations.
- **os**: For interacting with the file system.
- **pandas**: For data manipulation.
- **seaborn**: For statistical data visualization.
- **nltk**: For text preprocessing and stopword removal.
- **plotly**: For creating interactive visualizations.
- **wordcloud**: For generating word clouds.
- **scikit-learn**: For machine learning models and feature extraction.
- **tensorflow**: For deep learning tasks.
- **PIL (Pillow)**: For image handling in word cloud generation.

To install all the dependencies, you should create a txt file with:
kagglehub
emoji
matplotlib
numpy
pandas
seaborn
nltk
plotly
wordcloud
scikit-learn
tensorflow
pillow

And then use the following command:

```bash
pip install -r requirements.txt

