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

```python

# Sales by Customer Segment
sales_by_segment = df.groupby('Segment')['Sales'].sum()
# Sales by Product Category and Sub-Category
sales_by_category = df.groupby('Category')['Sales'].sum()
sales_by_sub_category = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)
# Sales by Shipping Mode
sales_by_ship_mode = df.groupby('Ship Mode')['Sales'].sum()
# Shipping Duration Calculation
df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
avg_shipping_duration = df['Shipping Duration'].mean()

```
### 1. Sales by Customer Segment
Type: Pie Chart<br>
Description:  Visualizes total sales for each customer segment to identify key customer groups.<br>
### Code:
```python
import pandas as pd
import matplotlib.pyplot as ply

df = pd.read_csv("train.csv")
df

plt.figure(figsize=(8, 8))
colors = sns.color_palette('coolwarm', len(sales_by_segment))
sales_by_segment.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=140)
plt.title('Sales Distribution by Customer Segment', fontsize=15, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.show()
```
### 2.  Sales by Product Category
Type: Bar Chart<br>
Description: Shows total sales for each product category.<br>
### Code:
```python
import seaborn as sns

plt.figure(figsize=(10, 6))
colors = sns.color_palette("Set2", len(sales_by_category))
sales_by_category.plot(kind='bar', color=colors)
plt.title('Sales by Product Category', fontsize=15, fontweight='bold')
plt.xlabel('Product Category')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()
```

### 3. Top 10 Sub-Categories by Sales
Type: Bar Chart<br>
Description: Visualizes the Top 10 Sub-Categories by Sales.<br>
### Code:
```python
plt.figure(figsize=(10, 6))
colors = sns.color_palette("Spectral", len(sales_by_sub_category))
sales_by_sub_category.plot(kind='bar', color=colors)
plt.title('Top 10 Sub-Categories by Sales', fontsize=15, fontweight='bold')
plt.xlabel('Sub-Category')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()
plt.show()
```
### 4. Sales by Shipping Mode
Type: Donut Chart<br>
Description: Displays total Sales by Shipping Mode.<br>
### Code:
```python
plt.figure(figsize=(8, 8))
colors = sns.color_palette("pastel", len(sales_by_ship_mode))
sales_by_ship_mode.plot(kind='pie', autopct='%1.1f%%', colors=colors, startangle=140, wedgeprops={'width': 0.4})
plt.title('Sales Distribution by Shipping Mode', fontsize=15, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.show()
```

### 5. Shipping Duration by Shipping Mode
Type: Box Plot<br>
Description: Shows the Shipping Duration by Shipping Mode.<br>
### Code:
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Ship Mode', y='Shipping Duration', palette="coolwarm")
plt.title('Shipping Duration by Ship Mode', fontsize=15, fontweight='bold')
plt.xlabel('Ship Mode')
plt.ylabel('Shipping Duration (days)')
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()
```

### 6. Sales Distribution by Region
Type: Violin Plot<br>
Description: Visualizes the Sales Distribution by Region.<br>
### Code:
``` python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Region', y='Sales', palette='muted', inner='quartile')
plt.title('Sales Distribution by Region', fontsize=15, fontweight='bold')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()
```
### 7. Monthly Sales Trend
Type: Line Plot<br>
Description:  Displays total Sales by Monthly Trends.<br>
### Code:
```python
plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o', color='orange')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 8. Seasonal Decompose
Type: Line Chart<br>
Description: Shows the Seasonal Trends <br>
### Code:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Set 'Order Date' as the index for time series analysis
df.set_index('Order Date', inplace=True)

# Resample sales data to a daily frequency and fill missing dates
daily_sales = df['Sales'].resample('D').sum().fillna(0)

# Perform time series decomposition
decomposition = seasonal_decompose(daily_sales, model='additive')

# Plot the decomposed components
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Time Series Decomposition of Sales Data')
plt.show()
```

## Result:
Thus, to perform Sales data analysis of a commercial store is successful.


## For a detailed analysis and visualizations, please refer to the attached Jupyter Notebook file.
