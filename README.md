# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## 1. Project Overview

This project uses Machine Learning techniques to segment real estate buyers based on their demographic information, geographic location, acquisition purpose, loan behavior, and property transaction details.

The main objective is to identify different buyer groups and understand their purchasing patterns to support real estate marketing strategies and investment profiling.

A Streamlit dashboard is developed to visualize buyer segments, geographic patterns, property analysis, and important business insights.

## 2. Project Objectives

- Analyze real estate buyer data.
- Understand buyer demographics and purchasing behavior.
- Identify buyer segments using clustering algorithms.
- Compare K-Means and Hierarchical Clustering.
- Analyze buyer distribution across countries and regions.
- Study property prices and floor areas.
- Develop an interactive Streamlit dashboard.
- Generate useful insights for real estate decision-making.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Plotly | Interactive charts |
| Streamlit | Web dashboard |
| Jupyter Notebook | Data analysis and experimentation |

## 4. Dataset Information

The project uses two datasets.

### Clients Dataset

- Total Records: 2,000
- Total Columns: 12

Important Columns:

- client_id
- client_type
- first_name
- last_name
- date_of_birth
- gender
- country
- region
- acquisition_purpose
- satisfaction_score
- loan_applied
- referral_channel

### Properties Dataset

- Total Records: 10,000
- Total Columns: 9

Important Columns:

- listing_id
- tower_number
- transaction_date
- unit_category
- unit_number
- floor_area_sqft
- sale_price
- listing_status
- client_ref

### Dataset Links

- Dataset Link 1: [Google Drive Dataset 1](https://drive.google.com/file/d/11_otjj-nmpjrllKuxmxGruuEZ-p6k0fY/view?usp=drive_link)
- Dataset Link 2: [Google Drive Dataset 2](https://drive.google.com/file/d/1BAMxf07o3BewWkNg-0aPJMgOBvbkK8Uv/view?usp=drive_link)

Local Dataset Files:

- `data/clients.csv`
- `data/properties.csv`

## 5. Project Methodology

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Missing Value Analysis
5. Categorical Data Encoding
6. Feature Selection
7. Feature Scaling
8. K-Means Clustering
9. Hierarchical Clustering
10. Model Evaluation
11. PCA Analysis
12. Geographic Buyer Analysis
13. Property and Buyer Analysis
14. Cluster Interpretation
15. Streamlit Dashboard Development

## 6. Machine Learning Algorithms

### K-Means Clustering

K-Means is used to group buyers based on similarities in their selected features.

- Number of Clusters: 4
- Silhouette Score: 0.4055

### Hierarchical Clustering

Hierarchical Clustering is used to identify buyer groups through a hierarchical clustering approach.

- Number of Clusters: 2
- Silhouette Score: 0.3779

### Model Comparison

K-Means with 4 clusters achieved a higher silhouette score than the evaluated Hierarchical Clustering model.

However, the K-Means clusters are imbalanced, so the smaller clusters require careful interpretation.

## 7. Key Project Results

| Metric | Result |
|---|---:|
| Total Buyers | 2,000 |
| Unique Client IDs | 2,000 |
| Total K-Means Clusters | 4 |
| Missing Cluster Values | 0 |
| Linked Property Records | 7,305 |
| Countries | 10 |
| Regions | 57 |

### Cluster Distribution

| Cluster | Buyers |
|---|---:|
| Cluster 0 | 356 |
| Cluster 1 | 1,623 |
| Cluster 2 | 10 |
| Cluster 3 | 11 |

## 8. Streamlit Dashboard

The interactive dashboard provides:

- Buyer Segmentation Overview
- Cluster Distribution
- Buyer Type Analysis
- Acquisition Purpose Analysis
- Loan Application Analysis
- Geographic Buyer Analysis
- Country-wise Cluster Analysis
- Property Analysis
- Customer Satisfaction Analysis
- Cluster Insights
- Final Buyer Segmentation Data
- CSV Download Option

### Dashboard Filters

Users can filter the dashboard by:

- Country
- Region
- Acquisition Purpose
- Client Type

## 9. Project Folder Structure

```text
Real_Estate_Buyer_Segmentation/
│
├── data/
│   ├── clients.csv
│   ├── properties.csv
│   ├── processed/
│   └── final/
│
├── models/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Data_Preprocessing.ipynb
│
├── reports/
│   ├── final_cluster_interpretation.csv
│   ├── insights.md
│   └── research_paper.md
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── clustering.py
│   ├── cluster_analysis.py
│   └── model_evaluation.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## 10. Installation and Setup

### Step 1: Download the Project

Download the project and open its folder in Visual Studio Code.

### Step 2: Install Required Packages

```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Main Project

```bash
python main.py
```

### Step 4: Run the Streamlit Dashboard

```bash
python -m streamlit run app.py
```

For Windows PowerShell using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The dashboard will open in your web browser.

## 11. Important Findings and Limitations

- The dataset contains 2,000 buyers and 10,000 property records.
- A total of 7,305 property records are linked to buyers.
- The K-Means model produced four clusters.
- Cluster 1 contains the majority of buyers.
- Clusters 2 and 3 contain very small buyer groups.
- Geographic features have a strong influence on the observed cluster distribution.
- Two PCA components explain approximately 6.59% of the total variance.

**Limitation:** The available dataset does not contain a buyer income column. Therefore, actual buyer income and high-net-worth status cannot be determined directly from this dataset.

## 12. Conclusion

This project demonstrates how Machine Learning and data visualization can be applied to real estate buyer segmentation and investment profiling.

K-Means and Hierarchical Clustering were evaluated to identify buyer groups. The Streamlit dashboard provides an interactive way to explore buyer behavior, geographic distribution, property transactions, and cluster insights.

The results can support further research and help real estate businesses understand customer patterns and improve data-driven decision-making.

---

**Project Domain:** Machine Learning | Data Science | Real Estate Analytics

**Programming Language:** Python