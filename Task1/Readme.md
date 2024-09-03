# <p align="center"> Sales Data Analysis </p> 
## Overview
This analysis explores sales data to derive insights using various visualizations. The dataset, train.csv, contains details about transactions, including order dates, customer segments, sales amounts, and more.

## Dataset Source and Description:
The dataset used in this analysis was sourced from Kaggle, a popular platform for data science competitions and datasets. The dataset, titled Sales Forecasting, was downloaded directly using Kaggle’s API integration. It includes various details about sales transactions such as order dates, customer segments, sales amounts, and more.

## Columns in the dataset:

Order Date: The date when the order was placed.<br>
Ship Date: The date when the order was shipped.<br>
Customer Segment: The segment of customers (e.g., Consumer, Corporate).<br>
Postal Code: The postal code for delivery, with missing values handled by forward and backward filling.<br>
Sales: The total sales amount for each order.<br>
Additional Columns: (Details based on the dataset specifics, e.g., Quantity, Discount, etc.)

## Acquisition from Kaggle

To acquire the dataset, the Kaggle API was used to programmatically download the required file. The following steps were taken:

Install Kaggle API: Installed the Kaggle API client to enable downloading datasets directly from Kaggle.<br>
Authenticate Kaggle Account: Used a Kaggle API token to authenticate and access the dataset.<br>
Download Dataset: Downloaded the train.csv file from the Kaggle dataset.

```python
!pip install kaggle
import kaggle

# Download the dataset from Kaggle
!kaggle datasets download rohitsahoo/sales-forecasting -f train.csv

# Unzip the dataset
import zipfile
zip = zipfile.ZipFile('train.csv.zip')
zip.extractall()
zip.close()
```

## Data Preprocessing

Date Conversion: Converted 'Order Date' and 'Ship Date' to datetime format for better time-based analysis.<br>
Handling Missing Values: Missing values in 'Postal Code' were handled by using forward-fill (method='ffill') to ensure continuity in data.

## Exploratory Data Analysis (EDA)

The Exploratory Data Analysis (EDA) focuses on uncovering patterns, trends, and relationships within the dataset through visualizations and statistical summaries:

### 1. Sales Over Time
Type: Line Chart<br>
Description: Displays the total sales amount over time, showing how sales vary by date.<br>
### Code:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('retail_sales_dataset.csv')

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Aggregate sales by date
sales_over_time = df.groupby('Date')['Total Amount'].sum().reset_index()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(sales_over_time['Date'], sales_over_time['Total Amount'], marker='o')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```
### 2. Sales by Product Category
Type: Bar Chart<br>
Description: Shows total sales for each product category.<br>
### Code:
```python
import seaborn as sns

# Aggregate sales by product category
sales_by_category = df.groupby('Product Category')['Total Amount'].sum().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Total Amount', y='Product Category', data=sales_by_category, palette='viridis')
plt.title('Sales by Product Category')
plt.xlabel('Total Amount')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()
```

### 3. Sales by Gender
Type: Pie Chart<br>
Description: Visualizes the proportion of total sales by gender.<br>
### Code:
```python
# Aggregate sales by gender
sales_by_gender = df.groupby('Gender')['Total Amount'].sum().reset_index()

# Plot
plt.figure(figsize=(8, 8))
plt.pie(sales_by_gender['Total Amount'], labels=sales_by_gender['Gender'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Sales by Gender')
plt.show()
```
### 4. Sales by Age
Type: Bar Chart<br>
Description: Displays total sales for each age group.<br>
### Code:
```python
# Aggregate sales by age
sales_by_age = df.groupby('Age')['Total Amount'].sum().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(x='Age', y='Total Amount', data=sales_by_age, palette='magma')
plt.title('Sales by Age')
plt.xlabel('Age')
plt.ylabel('Total Amount')
plt.grid(True)
plt.tight_layout()
plt.show()
```

### 5. Box Plot (Total Amount by Product Category)
Type: Box Plot<br>
Description: Shows the distribution and outliers of total amounts for each product category.<br>
### Code:
```python
# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Product Category', y='Total Amount', data=df, palette='Set2')
plt.title('Box Plot of Total Amount by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 6. Scatter Plot (Total Amount vs. Quantity)
Type: Scatter Plot<br>
Description: Visualizes the relationship between total amount and quantity, color-coded by product category.<br>
### Code:
``` python
# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='Total Amount', data=df, hue='Product Category', palette='viridis', alpha=0.7)
plt.title('Scatter Plot of Total Amount vs. Quantity')
plt.xlabel('Quantity')
plt.ylabel('Total Amount')
plt.legend(title='Product Category')
plt.grid(True)
plt.tight_layout()
plt.show()
```
### 7. Violin Plot (Total Amount by Age)
Type: Violin Plot<br>
Description: Illustrates the distribution of total amounts across different age groups.<br>
### Code:
```python
# Violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Age', y='Total Amount', data=df, palette='magma')
plt.title('Violin Plot of Total Amount by Age')
plt.xlabel('Age')
plt.ylabel('Total Amount')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 8. Heatmap of Sales by Product Category and Gender
Type: Heatmap<br>
Description: Shows the intensity of total sales across different product categories and genders.<br>
### Code:
```python
import numpy as np

# Pivot table for heatmap
heatmap_data = df.pivot_table(index='Product Category', columns='Gender', values='Total Amount', aggfunc=np.sum)

# Heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.0f')
plt.title('Heatmap of Total Amount by Product Category and Gender')
plt.xlabel('Gender')
plt.ylabel('Product Category')
plt.tight_layout()
plt.show()
```

## Result:
Thus, to perform Sales data analysis of a commercial store is successful.


## For a detailed analysis and visualizations, please refer to the attached Jupyter Notebook file.
