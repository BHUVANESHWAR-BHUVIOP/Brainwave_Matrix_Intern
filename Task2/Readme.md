# <p align="center"> Social Media Sentiment Analysis </p> 
## Project Overview:
This project involves processing and analyzing a sentiment dataset to understand public opinion on social media. The analysis includes data cleaning, stemming, text processing, and the training of a sentiment classification model using natural language processing (NLP) techniques

## Requirements:
<li>Python 3.x
<li>pandas
<li>numpy
<li>matplotlib
<li>seaborn
<li>re 


## Dataset:
The dataset used in this project was downloaded from Kaggle and contains a large number of processed tweets labeled as either positive or negative sentiments.

## Columns in the Dataset:
Sentiment: Sentiment label (0 for negative, 1 for positive).
Text: The tweet content.

## Steps:
#### 1. Data Processing
Sentiment Mapping: The dataset contains two labels, 0 for negative tweets and 1 for positive tweets.
Text Cleaning: Performed text preprocessing including converting text to lowercase, removing special characters, and cleaning the text using regular expressions.

#### 2. Stemming
Stemming Process: Reduced words to their root form using the PorterStemmer to standardize the text data for better model performance.

#### 3. Data Splitting
Train-Test Split: The data was split into training and testing datasets to evaluate the model's performance.

#### 4.Model Training and Evaluation
Model Used: Logistic Regression was employed to classify the sentiment of the tweets.
Evaluation: The model's accuracy was measured to determine its effectiveness

#### 5. Data Visualization
Sentiment Distribution: Visualized the distribution of positive and negative sentiments in the dataset. This provides insights into the overall balance between positive and negative tweets.
The dataset contains two sentiment classes: 0 (negative) and 1 (positive).
A bar chart can be used to represent the distribution of these sentiment labels across the dataset.
Sentiment Trends Over Time: Although the dataset focuses primarily on tweet content and sentiment, visualizing sentiment trends over time is possible by mapping tweets to specific periods (e.g., day or month, if timestamp data is available).
Text-Based Analysis: Word cloud or bar charts can be employed to visualize the most frequent words in positive and negative tweets. This helps identify common themes or topics associated with different sentiment labels.
Sentiment Classification Results: A confusion matrix can be plotted after the model evaluation to visualize the model’s performance in classifying sentiments. This gives a clear indication of false positives and false negatives, along with accurate predictions.



# Program:

```python
# Import necessary libraries
import pandas as pd
# Load the dataset
# Replace 'your_dataset.csv' with the actual file name or path of your dataset
df = pd.read_csv('sentimentdataset.csv')
```
```python
# Display the first few rows of the dataset to understand its structure
print("Dataset Preview:")
display(df.head())
```
![1](https://github.com/user-attachments/assets/efabf105-7e06-4bc3-925e-1c3a4a1c3666)

```python
# Get basic information about the dataset (column names, data types, missing values)
print("\nDataset Information:")
df.info()
```
![2](https://github.com/user-attachments/assets/105707ba-8443-4814-9b75-a084d19511df)
```python
# Check for missing values in each column
print("\nMissing Values in Dataset:")
print(df.isnull().sum())
```
![3](https://github.com/user-attachments/assets/c7349610-24b7-4149-8669-5c4daaaee066)

```python
# Get a summary of numerical columns
print("\nStatistical Summary of Numerical Columns:")
display(df.describe())
```
![4](https://github.com/user-attachments/assets/940ed593-7b7f-4a63-87d6-636309e0b3a1)


```python
# Drop the unnecessary columns
df_cleaned = df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

# Remove leading/trailing spaces from 'Platform' column
df_cleaned['Platform'] = df_cleaned['Platform'].str.strip()

# Verify the changes
print("Cleaned Unique Platforms in the Dataset:")
print(df_cleaned['Platform'].unique())

# Display the first few rows of the cleaned dataset
display(df_cleaned.head())
```
![5](https://github.com/user-attachments/assets/7ceb4155-63ca-4918-a002-23bce33ba048)

```python
import spacy

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text using spaCy
def preprocess_text_spacy(text):
    # Process text with spaCy
    doc = nlp(text.lower())
    # Remove stopwords, punctuation, and lemmatize tokens
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(words)

# Apply the preprocessing to the 'Text' column
df_cleaned['Cleaned_Text'] = df_cleaned['Text'].apply(preprocess_text_spacy)

# Display the cleaned text data
print("Sample of Cleaned Text Data (spaCy):")
display(df_cleaned[['Text', 'Cleaned_Text']].head())
```
![6](https://github.com/user-attachments/assets/6bca9260-d6fe-4ffd-856d-3668906a8f02)

```python
# Plot the distribution of the new mapped sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='Mapped_Sentiment', data=df_cleaned, palette='coolwarm')
plt.title('Distribution of Mapped Sentiments')
plt.xlabel('Mapped Sentiment')
plt.ylabel('Count')
plt.show()

```
![7](https://github.com/user-attachments/assets/235435c7-354a-438c-93dc-7bfc170eedfe)

```python
# Plot sentiment distribution across platforms
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', hue='Mapped_Sentiment', data=df_cleaned, palette='coolwarm')
plt.title('Sentiment Distribution Across Platforms')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

```
![8](https://github.com/user-attachments/assets/be8cd232-d0fa-449d-bbba-fe01d2f2d61b)

```python
# Sentiment trend over time (by Year and Month)
df_cleaned['Date'] = pd.to_datetime(df_cleaned[['Year', 'Month', 'Day']])

# Group by Date and Sentiment
sentiment_trends = df_cleaned.groupby(['Date', 'Mapped_Sentiment']).size().unstack(fill_value=0)

# Plot the trend
plt.figure(figsize=(12, 6))
sentiment_trends.plot(kind='line', figsize=(12, 6), marker='o')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.legend(title='Sentiment')
plt.show()

```
![9](https://github.com/user-attachments/assets/d72dc647-4129-4d55-b4eb-aca9eb61bd21)

```python
# Plot sentiment distribution across countries
plt.figure(figsize=(12, 8))
sns.countplot(x='Country', hue='Mapped_Sentiment', data=df_cleaned, palette='coolwarm', order=df_cleaned['Country'].value_counts().index)
plt.title('Sentiment Distribution Across Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()

```
![10](https://github.com/user-attachments/assets/51b0a9d5-49cf-43d4-ba60-682766d6528e)

## Applications

1.Public Sentiment Analysis: Analyzing tweets to gauge public sentiment on various topics.
2.Text Preprocessing: Using stemming and text cleaning to prepare data for sentiment analysis models.
3.Model Training: Training machine learning models on large sentiment datasets for accurate predictions.


## Result:
The project successfully implemented sentiment analysis using NLP techniques, achieving a reliable accuracy with logistic regression.

## For a detailed analysis and visualizations, please refer to the attached Jupyter Notebook file.
