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

[View Project](./COVID-19_analysis/01_covid19_global_analysis.ipynb)

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

[View Project](./Titanic%20EDA/02-1_titanic_survival_analysis.ipynb)

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

## Technical Skills Demonstrated

### Programming & Tools
- Python
- Jupyter Notebooks
- Data Manipulation: Pandas
- Visualization: Matplotlib, Seaborn
- Data Processing: NumPy
- Natural Language Processing: NLTK or similar libraries
- Custom formatting and styling for better data presentation

### Data Science Techniques
- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
  - Missing value handling
  - Duplicate removal
  - Data type conversions
- Time Series Analysis
  - Trend analysis
  - Monthly/Yearly aggregations
  - Rolling averages
- Statistical Analysis
  - Distribution analysis
  - Rate calculations
  - Demographic analysis
  - Survival rate analysis
- Interactive Analysis
  - User-input based visualizations
  - Dynamic metric selection
  - Filtered data exploration
- Natural Language Processing
  - Text summarization
  - Tokenization and text cleaning

## Repository Structure
├── datasets/
│   ├── covid19/
│   │   └── raw_data.csv
│   └── titanic/
│       └── passenger_data.csv
├── 01_covid19_analysis/
│   └── 01_covid19_global_analysis.ipynb
├── 02_titanic_analysis/
│   ├── 02_titanic_survival_analysis.ipynb
│   └── 03_titanic_visualisations.ipynb
├── 03_text_summarization/
│   └── 03_text_summarization.ipynb
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
