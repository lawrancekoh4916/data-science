# Data Science Portfolio

## Overview
Welcome to my data science portfolio. This repository showcases my expertise in data analysis, visualization, and machine learning through various real-world projects. Each project demonstrates my proficiency in Python and essential data science libraries while solving complex analytical challenges.

## Projects

### 1. COVID-19 Global Impact Analysis
A comprehensive analysis of the COVID-19 pandemic using data from "Our World in Data". This project demonstrates:

- **Advanced Data Visualization** using:
  - Multi-axis plots for comparing cases and deaths
  - Time series trend analysis
  - Box plots for yearly distribution analysis
  - Bar plots for continental comparisons
  - Interactive user-input driven visualizations
- **Statistical Analysis** including:
  - Fatality rate calculations and trends
  - Regional distribution analysis
  - Year-wise monthly trend analysis
- **Data Processing Techniques**:
  - Missing value imputation
  - Feature engineering (e.g., daily growth rates)
  - Data cleaning and preprocessing
  - Time-based aggregations

[View Project](./01_covid19_analysis/01_covid19_global_analysis.ipynb)

### 2-1. Titanic Survival Analysis
An exploratory data analysis of the famous Titanic dataset, investigating factors that influenced passenger survival. Key analyses include:

- **Data Cleaning & Preprocessing**:
  - Missing value imputation for Age and Embarked columns
  - Feature engineering (Family Size calculation)
  - Data type handling and column management
- **Statistical Analysis**:
  - Survival rates by passenger class
  - Age distribution analysis
  - Fare analysis and correlation with survival
  - Family size impact on survival
- **Passenger Demographics**:
  - Class distribution
  - Embarkation port analysis
  - Age and gender distribution
- **Key Findings**:
  - Higher class passengers had better survival rates
  - Age correlation with survival outcomes
  - Family size influence on survival chances
  - Port of embarkation survival patterns

[View Project](./02_titanic_analysis/02-1_titanic_survival_analysis.ipynb)

### 2-2. Titanic Data Visualizations
A comprehensive visualization study of the Titanic dataset, focusing on different aspects of passenger demographics and their relationships. Features include:

- **Survival Analysis Visualizations**:
  - Distribution of survivors vs non-survivors
  - Class-wise passenger distribution
  - Age distribution analysis
- **Passenger Demographics Visualization**:
  - Age distribution through histograms
  - Class distribution analysis
  - Fare distribution through box plots
- **Correlation Analysis**:
  - Heatmap of numerical variables
  - Relationship between age and passenger class
- **Technical Implementation**:
  - Seaborn for statistical visualizations
  - Matplotlib for custom plots
  - NumPy and Pandas for data handling

[View Project](./02_titanic_analysis/02-2_titanic_visualisations.ipynb)

### 3. Text Summarization
A simple project demonstrating text summarization techniques. This project showcases:

- **Natural Language Processing (NLP)**:
  - Text preprocessing and tokenization
  - Frequency-based summarization
  - Implementation of basic NLP techniques
- **Data Handling**:
  - Text data cleaning and preparation
  - Handling of large text datasets

[View Project](./03_text_summarization/03_text_summarization.ipynb)

### 4. Hotel Sentiment Analysis
An analysis of hotel reviews to understand sentiment trends and correlations between different aspects of hotel services and overall guest satisfaction. This project includes:

- **Data Import and Setup**: Utilizes libraries such as `pandas`, `numpy`, `scipy`, and `matplotlib` for data manipulation and visualization.
- **Data Loading**: Reads hotel review data from an Excel file, including sentiment scores and star ratings.
- **Data Analysis**:
  - Sentiment Analysis: Calculates sentiment scores for hotel aspects like room, service, location, food, and value.
  - Correlation Analysis: Examines the relationship between overall guest satisfaction (GRI) and specific sentiments for different star categories.
  - Visualization: Includes code for visualizing data trends and correlations.
- **Statistical Testing**: Uses Pearson correlation tests to determine dependencies between overall satisfaction and specific sentiment scores.

[View Project](./04_hotel_sentiment_analysis/04_hotel_sentiment_analysis.ipynb)

### 5. Customer Segmentation using RFM Analysis
This project involves analyzing transaction data to segment customers based on Recency, Frequency, and Monetary Value (RFM) metrics. Key features include:

- **Data Import and Setup**: Utilizes libraries such as `pandas`, `numpy`, `scipy`, and `matplotlib` for data manipulation and visualization.
- **Data Loading**: Reads transaction data from a CSV file, including transaction dates and amounts.
- **RFM Analysis**:
  - Calculation of Recency, Frequency, and Monetary Value for each customer.
  - Segmentation of customers into quantiles for scoring.
  - Grouping customers by RFM scores to identify most valuable, loyal, and at-risk customers.
- **Visualization**: Plots distributions of RFM metrics and customer segments.

[View Project](./05_customer_segmentation/05_customer_segmentation.ipynb)

### 6. Fake News Detection
This project focuses on detecting fake news using Natural Language Processing (NLP) techniques. Key features include:

- **Data Description**:
  - The dataset includes text data from social media and news platforms, along with content tags and labels indicating the truthfulness of the content.
  - Labels are categorized as: Half-True, False, Mostly-True, True, Barely-True, and Not-Known.
- **Data Preprocessing**:
  - Handling missing values in the dataset.
  - Reclassification of labels into binary categories: Fake News (1) and Real News (0).
- **Modeling**:
  - Utilizes various machine learning models such as Logistic Regression, Naive Bayes, Random Forest, and Gradient Boosting.
  - Implements a Stacking Classifier to combine the predictions of multiple models for improved accuracy.
- **Evaluation**:
  - The models are evaluated based on metrics like Accuracy, Precision, Recall, and ROC_AUC.
  - The Stacking Classifier is used to make final predictions on the test set.
- **Visualization**:
  - The distribution of predictions is visualized to understand the model's performance.

[View Project](./06_fake_news_detection/06_fake_news_detection.ipynb)

### 7. AI-Augmented Project Management
This capstone project leverages AI and machine learning to enhance project management processes. Key features include:

- **Data Import and Setup**: Utilizes libraries such as `pandas`, `numpy`, `scipy`, and `matplotlib` for data manipulation and visualization.
- **Data Loading**: Reads project task data from an Excel file, including task descriptions and associated labels.
- **Text Preprocessing**:
  - Cleaning and tokenization of text data using regular expressions and NLP libraries like `spacy`.
  - Lemmatization and removal of stop words to prepare text for analysis.
- **Modeling**:
  - Implementation of machine learning models such as Naive Bayes, Random Forest, and Gradient Boosting for task classification.
  - Use of a Stacking Classifier to improve model performance by combining multiple models.
- **Evaluation**:
  - Models are evaluated using metrics like Accuracy and F1 Score.
  - Visualization of model performance through confusion matrices.
- **Visualization**:
  - Distribution of task labels is visualized to understand the dataset.

[View Project](./07_ai_augmented_project_management/07_ai_augmented_project_management.ipynb)

## Technical Skills Demonstrated

### Programming & Tools
- Python
- Jupyter Notebooks
- Data Manipulation: Pandas
- Visualization: Matplotlib, Seaborn
- Data Processing: NumPy
- Natural Language Processing: NLTK, Spacy
- Custom formatting and styling for better data presentation

### Data Science Techniques
- **Exploratory Data Analysis (EDA)**
- **Data Cleaning & Preprocessing**
  - Missing value handling
  - Duplicate removal
  - Data type conversions
- **Time Series Analysis**:
  - Trend analysis
  - Monthly/Yearly aggregations
  - Rolling averages
- **Statistical Analysis**:
  - Distribution analysis
  - Rate calculations
  - Demographic analysis
  - Survival rate analysis
- **Interactive Analysis**:
  - User-input based visualizations
  - Dynamic metric selection
  - Filtered data exploration
- **Machine Learning Techniques**:
  - Implementation of classification algorithms such as Logistic Regression, Naive Bayes, Random Forest, and Gradient Boosting.
  - Use of a Stacking Classifier to improve model performance by combining multiple models.
- **Model Evaluation**:
  - Evaluation of models using metrics like Accuracy, Precision, Recall, and ROC_AUC.
  - Visualization of model performance through confusion matrices and distribution plots.
- **Data Handling**:
  - Handling of imbalanced datasets by reclassifying labels into binary categories.
  - Use of text vectorization techniques such as CountVectorizer and TfidfVectorizer for feature extraction.
- **Natural Language Processing (NLP)**:
  - Text summarization
  - Tokenization and text cleaning
  - Text preprocessing including handling missing values and reclassification of text labels.
  - Application of NLP techniques to detect fake news content.

## Repository Structure
├── datasets/
│   ├── 01_covid19/
│   │   └── raw_data.csv
│   ├── 02_titanic/
│   │   └── passenger_data.csv
│   ├── 03_text_summarization/
│   │   └── articles.csv
│   ├── 04_hotel_sentiment/
│   │   └── reviews.csv
│   ├── 05_customer_segmentation/
│   │   └── customer_data.csv
│   ├── 06_fake_news_detection/
│   │   ├── Train.csv
│   │   └── Test.csv
│   └── 07_project_management/
│       └── project_data.csv
├── 01_covid19_analysis/
│   └── 01_covid19_global_analysis.ipynb
├── 02_titanic_analysis/
│   ├── 02_titanic_survival_analysis.ipynb
│   └── 03_titanic_visualisations.ipynb
├── 03_text_summarization/
│   └── 03_text_summarization.ipynb
├── 04_hotel_sentiment_analysis/
│   └── 04_hotel_sentiment_analysis.ipynb
├── 05_customer_segmentation/
│   └── 05_customer_segmentation.ipynb
├── 06_fake_news_detection/
│   └── 06_fake_news_detection.ipynb
├── 07_ai_augmented_project_management/
│   └── 07_ai_augmented_project_management.ipynb
├── LICENSE
└── README.md

## Getting Started
1. Clone the repository:
bash
git clone https://github.com/yourusername/data-science-portfolio.git

2. Install required dependencies:
bash
pip install pandas numpy matplotlib seaborn jupyter

3. Navigate to specific project directories and open the Jupyter notebooks to view the analysis.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter

## Contact & Connect
Feel free to reach out if you have any questions about my projects or would like to collaborate:
- LinkedIn: https://www.linkedin.com/in/lawrancekoh/
- Email: lawrancekoh@outlook.com

## License
This project is licensed under the MIT License - see the LICENSE file for details.
