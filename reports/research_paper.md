# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Abstract
Real estate markets contain buyers with different demographic characteristics, geographic locations, purchasing purposes, and financing behaviors. Understanding these differences can help businesses improve marketing strategies and investment decisions.

This project applies data analysis and machine learning clustering techniques to segment real estate buyers. K-Means and Hierarchical Clustering are evaluated to identify groups with similar characteristics. A Streamlit dashboard is developed to visualize buyer segments and important business insights.

## 1. Introduction
Buyer segmentation is an important process in real estate market analysis. Different buyers have different property requirements and acquisition purposes.

Traditional analysis may not easily identify hidden patterns among large datasets. Machine learning clustering techniques can help discover groups of buyers with similar characteristics.

The main aim of this project is to use machine learning to understand real estate buyer behavior and support better business decisions.

## 2. Objectives
- Analyze real estate buyer demographics.
- Identify buyer purchasing purposes.
- Study geographic buyer distribution.
- Analyze loan application behavior.
- Apply K-Means and Hierarchical Clustering.
- Compare clustering model performance.
- Develop an interactive Streamlit dashboard.

## 3. Dataset Description
The project uses two datasets:

### Clients Dataset
Contains 2,000 buyer records and 12 columns.

Important attributes include:
- Client ID
- Client Type
- Date of Birth
- Gender
- Country
- Region
- Acquisition Purpose
- Satisfaction Score
- Loan Applied
- Referral Channel

### Properties Dataset
Contains 10,000 property records and 9 columns.

Important attributes include:
- Listing ID
- Transaction Date
- Unit Category
- Floor Area
- Sale Price
- Listing Status
- Client Reference

## 4. Methodology

### 4.1 Data Cleaning
- Loaded the client and property datasets.
- Checked missing values and duplicate records.
- Cleaned sale price values.
- Converted transaction dates into date format.
- Examined missing client references.

### 4.2 Exploratory Data Analysis
Performed analysis on:
- Buyer types
- Acquisition purposes
- Loan applications
- Geographic distribution
- Property prices
- Buyer satisfaction
- Referral channels

### 4.3 Feature Engineering
- Selected relevant buyer features.
- Encoded categorical variables.
- Applied feature scaling.
- Prepared the feature matrix for clustering.

### 4.4 Machine Learning Models
Two clustering algorithms were evaluated:

**K-Means Clustering**
- Used to group buyers into four clusters.
- Silhouette Score: 0.4055.

**Hierarchical Clustering**
- Evaluated using two clusters.
- Silhouette Score: 0.3779.

### 4.5 Model Evaluation
The Silhouette Score was used to compare clustering quality.

K-Means achieved a higher score than Hierarchical Clustering in the evaluated configurations.

However, the K-Means clusters were imbalanced. Therefore, the small clusters should be interpreted carefully.

## 5. Results and Discussion

The analysis identified 2,000 buyers across 10 countries and 57 regions.

Key findings:
- Individual buyers represent the majority of the dataset.
- Home acquisition is more common than investment acquisition.
- Approximately 36.8% of buyers applied for loans.
- The USA has the highest buyer count.
- The average linked property sale price is approximately $345,072.

K-Means produced four clusters:
- Cluster 0: 356 buyers
- Cluster 1: 1,623 buyers
- Cluster 2: 10 buyers
- Cluster 3: 11 buyers

Geographic patterns strongly influence some clusters. For example, Cluster 1 mainly contains buyers from the USA and Canada, while Cluster 3 contains buyers from Germany.

Clusters 2 and 3 are very small, so their patterns should not be generalized without further investigation.

## 6. Streamlit Dashboard
An interactive dashboard was developed using Streamlit and Plotly.

Dashboard features:
- Buyer segmentation overview
- Cluster distribution
- Buyer type analysis
- Acquisition purpose analysis
- Loan application analysis
- Geographic buyer analysis
- Property analysis
- Cluster insights
- Data filtering and download options

Users can filter buyers by country, region, acquisition purpose, and client type.

## 7. Limitations
- Buyer income information is not available.
- Property sale price cannot be treated as buyer income.
- Some property records have missing client references.
- The clustering results contain highly imbalanced groups.
- Further feature selection and model tuning may improve segmentation quality.

## 8. Future Scope
- Test additional clustering algorithms such as DBSCAN.
- Explore different feature selection techniques.
- Improve cluster balance and interpretability.
- Add more buyer financial information if available.
- Develop advanced recommendation features.
- Improve the dashboard with additional interactive visualizations.

## 9. Conclusion
This project demonstrates the use of data analytics and machine learning for real estate buyer segmentation.

K-Means and Hierarchical Clustering were evaluated, and K-Means achieved the higher Silhouette Score in the tested configurations.

The Streamlit dashboard provides an interactive way to explore buyer characteristics, geographic patterns, financing behavior, and property transactions.

The results can support real estate businesses in understanding buyer behavior and developing data-driven marketing strategies.

## 10. Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Jupyter Notebook