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
Sentiment: Sentiment label (0 for negative, 1 for positive).</br>
Text: The tweet content.

## Steps:
#### 1. Data Processing
Sentiment Mapping: The dataset contains two labels, 0 for negative tweets and 1 for positive tweets.</br>
Text Cleaning: Performed text preprocessing including converting text to lowercase, removing special characters, and cleaning the text using regular expressions.

#### 2. Stemming
Stemming Process: Reduced words to their root form using the PorterStemmer to standardize the text data for better model performance.

#### 3. Data Splitting
Train-Test Split: The data was split into training and testing datasets to evaluate the model's performance.

#### 4.Model Training and Evaluation
Model Used: Logistic Regression was employed to classify the sentiment of the tweets.</br>
Evaluation: The model's accuracy was measured to determine its effectiveness

#### 5. Data Visualization
Sentiment Distribution: Visualized the distribution of positive and negative sentiments in the dataset. This provides insights into the overall balance between positive and negative tweets.</br>
The dataset contains two sentiment classes: 0 (negative) and 1 (positive).</br>
A bar chart can be used to represent the distribution of these sentiment labels across the dataset.</br>
Sentiment Trends Over Time: Although the dataset focuses primarily on tweet content and sentiment, visualizing sentiment trends over time is possible by mapping tweets to specific periods (e.g., day or month, if timestamp data is available).</br>
Text-Based Analysis: Word cloud or bar charts can be employed to visualize the most frequent words in positive and negative tweets. This helps identify common themes or topics associated with different sentiment labels.</br>
Sentiment Classification Results: A confusion matrix can be plotted after the model evaluation to visualize the model’s performance in classifying sentiments. This gives a clear indication of false positives and false negatives, along with accurate predictions.</br>



# Program:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the dataset
# Replace 'your_dataset.csv' with the actual file name or path of your dataset
twitter_data=pd.read_csv('/content/training.1600000.processed.noemoticon.csv',encoding='ISO-8859-1')```
```python
# Display the first few rows of the dataset to understand its structure
twitter_data.head()
```
![Screenshot 2024-09-25 195321](https://github.com/user-attachments/assets/3ca50557-03bb-4c98-a823-bf8060c54a2e)

```python
#printing the stopwords in english
print(stopwords.words('english'))
```
![Screenshot 2024-09-25 195754](https://github.com/user-attachments/assets/c7908caf-c5b7-4a05-861f-e924ef888236)


```python
# Check for missing values in each column
twitter_data.isnull().sum()
```
![Screenshot 2024-09-25 195618](https://github.com/user-attachments/assets/4d1d99ae-67fa-40a3-88e1-d5f69869d835)

```python
# stemming is the process of reducing a word to its root word
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]

  return stemmed_content
print(twitter_data['stemmed_content'])

```
![Screenshot 2024-09-25 200019](https://github.com/user-attachments/assets/7be28054-ec22-4c49-b266-a32cf185a5c9)


```python
# splitting the data to training data and splitting data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,X_test.shape)
print(X_train)
print(X_test)
```
![Screenshot 2024-09-25 200249](https://github.com/user-attachments/assets/c71bf2ca-ab22-4d2a-a429-534409783ffc)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Example: Assuming `y_test` are the true labels and `y_pred` are the predictions
Y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

```
![Screenshot 2024-09-25 200355](https://github.com/user-attachments/assets/bfb8d1a3-5713-4e19-b5e4-50ed93282361)

```python
plt.figure(figsize=(10, 6))
plt.stem(top_features['feature'], top_features['importance'], linefmt='-', markerfmt='o', basefmt=" ", use_line_collection=True)
plt.title('Top 20 Important TF-IDF Features')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.show()

```
![Screenshot 2024-09-25 200446](https://github.com/user-attachments/assets/a000af52-f12b-4395-8db9-1f1a0f6f72d0)

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Example: Assuming 'text' is the column with text data
# Combine all text for visualization
all_words = ' '.join(twitter_data['text'])

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Text Data')
plt.show()
```
![Screenshot 2024-09-25 200535](https://github.com/user-attachments/assets/126ed0f8-1520-4676-8a8f-dec7c68fbc8a)

```python

# Visualize the distribution of tweet lengths
twitter_data['tweet_length'] = twitter_data['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(twitter_data['tweet_length'], bins=50, kde=True)
plt.title('Distribution of Tweet Lengths')
plt.xlabel('Tweet Length')
plt.ylabel('Frequency')
plt.show()

```
![Screenshot 2024-09-25 200630](https://github.com/user-attachments/assets/ee93b72f-e657-423b-a85d-aa243a5242a9)

```python
# Create a bar chart
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Sentiments')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()

```
![Screenshot 2024-09-25 200717](https://github.com/user-attachments/assets/bf381509-16e3-44e4-949c-b25c1a995f68)


## Applications

1.Public Sentiment Analysis: Analyzing tweets to gauge public sentiment on various topics.</br>
2.Text Preprocessing: Using stemming and text cleaning to prepare data for sentiment analysis models.</br>
3.Model Training: Training machine learning models on large sentiment datasets for accurate predictions.</br>


## Result:
The project successfully implemented sentiment analysis using NLP techniques, achieving a reliable accuracy with logistic regression.

## For a detailed analysis and visualizations, please refer to the attached Jupyter Notebook file.
