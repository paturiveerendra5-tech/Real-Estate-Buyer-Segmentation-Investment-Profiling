# 🏠 Real Estate Buyer Segmentation & Investment Profiling

### Machine Learning-Based Real Estate Buyer Intelligence

An end-to-end Machine Learning and Data Analytics project designed to analyze real estate buyers, identify customer segments, understand purchasing patterns, and generate meaningful business insights through an interactive Streamlit dashboard.

---

## 📌 1. Project Overview

The **Real Estate Buyer Segmentation and Investment Profiling** project applies machine learning and data analytics to understand buyer behavior in the real estate market.

The project uses K-Means and Hierarchical Clustering to identify groups of buyers with similar characteristics.

An interactive Streamlit dashboard presents buyer demographics, purchasing purposes, geographic patterns, property transactions, customer satisfaction, and machine learning evaluation results.

## 🎯 2. Project Objectives

- Analyze real estate buyer demographics and behavior.
- Identify buyer segments using machine learning.
- Understand home-buying and investment purposes.
- Explore geographic and property-related trends.
- Analyze loan application patterns.
- Study customer satisfaction.
- Compare clustering model performance.
- Build an interactive real estate analytics dashboard.
- Generate meaningful insights to support data-driven decisions.

## 📊 3. Dataset Information

The project uses two primary datasets.

| Dataset | Records | Description |
|---|---:|---|
| Buyers Dataset | 2,000 | Buyer demographics and characteristics |
| Properties Dataset | 10,000 | Property and transaction information |
| Countries | 10 | Geographic coverage |
| Regions | 57 | Regional coverage |
| Linked Property Records | 7,305 | Properties linked to buyer references |

**Dataset attributes include:**

**Buyers Dataset**
- Client ID
- Client Type
- Gender
- Date of Birth
- Country
- Region
- Acquisition Purpose
- Satisfaction Score
- Loan Applied
- Referral Channel

**Properties Dataset**
- Listing ID
- Transaction Date
- Unit Category
- Floor Area
- Sale Price
- Listing Status
- Client Reference

The property dataset contains 2,695 records with missing client references.

## 🛠️ 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| PCA | Dimensionality reduction |
| Jupyter Notebook | Data analysis and experimentation |

## 🤖 5. Machine Learning Methodology

### 5.1 Data Preprocessing
- Loaded and inspected buyer and property datasets.
- Checked missing values and duplicate records.
- Cleaned and prepared data.
- Converted relevant data types.
- Encoded categorical variables.
- Applied feature scaling.

### 5.2 K-Means Clustering

K-Means was applied to group buyers based on similarities in their prepared features.

- Number of clusters: 4
- Silhouette Score: 0.4055

### 5.3 Hierarchical Clustering

Hierarchical Clustering was used as an alternative approach to identify buyer groups.

- Number of clusters: 2
- Silhouette Score: 0.3779

### 5.4 PCA Visualization

Principal Component Analysis (PCA) was used to project high-dimensional buyer features into two dimensions for visualization.

The first two principal components explain approximately 6.59% of the total variance.

### 5.5 Model Evaluation

The clustering models were evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## 📈 6. Model Evaluation Results

| Evaluation Metric | K-Means | Hierarchical |
|---|---:|---:|
| Number of Clusters | 4 | 2 |
| Silhouette Score | 0.4055 | 0.3779 |
| Davies-Bouldin Index | 2.9275 | 4.7778 |
| Calinski-Harabasz Index | 48.1669 | 77.9608 |

**Model Comparison:**

K-Means achieved a higher Silhouette Score and a lower Davies-Bouldin Index than Hierarchical Clustering in the evaluated configurations.

Therefore, K-Means performed better on these two evaluation measures. The Calinski-Harabasz Index was higher for Hierarchical Clustering, so model selection should consider all evaluation measures and cluster interpretability.

## 📊 7. Interactive Dashboard Features

The project includes a Streamlit dashboard with 13 sections.

| No. | Dashboard Section |
|---:|---|
| 1 | Overview |
| 2 | Cluster Analysis |
| 3 | Buyer Analysis |
| 4 | Geographic Analysis |
| 5 | Property Analysis |
| 6 | Customer Satisfaction |
| 7 | Cluster Insights |
| 8 | ML Model Evaluation |
| 9 | K-Means Optimization |
| 10 | PCA Visualization |
| 11 | Buyer Data Explorer |
| 12 | About Project |
| 13 | Technical Validation |

### Dashboard Capabilities

- Interactive buyer filters.
- Country and region analysis.
- Buyer segmentation visualization.
- Property transaction analysis.
- Customer satisfaction analysis.
- Cluster-level insights.
- Machine learning model comparison.
- K-Means optimization.
- PCA visualization.
- Buyer data exploration.

## 📁 8. Project Structure

```text
Real_Estate_Buyer_Segmentation/
│
├── data/
│   ├── clients.csv
│   ├── properties.csv
│   ├── final/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── README.md
│   ├── project_insights.md
│   ├── final_project_report.md
│   ├── model_evaluation_comparison.csv
│   └── kmeans_cluster_optimization.csv
│
├── src/
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

*Note: The structure is illustrative. Update filenames to match the actual files in your repository.*

## ⚙️ 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/paturiveerendra5-tech/Real-Estate-Buyer-Segmentation-Investment-Profiling.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Real-Estate-Buyer-Segmentation-Investment-Profiling
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The dashboard will open in your web browser.

## 🌐 10. Live Dashboard

Explore the deployed application:

[Open Real Estate Buyer Intelligence Dashboard](https://real-estate-buyer-segmentation-investment-profilin-7fkgkpsvghr.streamlit.app/)

## 🔗 11. GitHub Repository

Explore the complete project source code:

[View GitHub Repository](https://github.com/paturiveerendra5-tech/Real-Estate-Buyer-Segmentation-Investment-Profiling)

## 📑 12. Project Reports

The `reports/` directory contains supporting analytical outputs and documentation.

- Project Insights Report
- Final Project Report
- Model Evaluation Comparison
- K-Means Cluster Optimization
- Additional analysis and validation outputs

## 🔍 13. Key Findings

- The project analyzes 2,000 buyers across 10 countries and 57 regions.
- Individual buyers represent the majority of the dataset.
- Home acquisition is more common than investment acquisition.
- Approximately 36.8% of buyers applied for loans.
- K-Means identified four buyer clusters.
- The largest cluster contains 1,623 buyers.
- The second-largest cluster contains 356 buyers.
- Two smaller clusters contain 10 and 11 buyers.
- The K-Means Silhouette Score is 0.4055.
- The first two PCA components explain approximately 6.59% of the variance.

**Important Observation:** The clustering results are imbalanced. Smaller clusters may represent niche groups or outliers and should be investigated further.

## ⚠️ 14. Project Limitations

- Buyer income information is not available.
- Property sale price should not be interpreted as buyer income or personal wealth.
- Some property records have missing client references.
- The clustering results contain highly imbalanced groups.
- The two-dimensional PCA visualization captures a limited proportion of total variance.
- Model performance depends on the selected features and preprocessing methods.

## 🚀 15. Future Enhancements

- Explore additional clustering algorithms such as DBSCAN.
- Improve feature selection and clustering optimization.
- Develop advanced buyer behavior prediction.
- Add interactive geographic visualizations.
- Integrate real-time property market data.
- Develop personalized property recommendation features.
- Improve cluster interpretation and business insights.

## 👨‍💻 16. Project Information

| Category | Details |
|---|---|
| Project Title | Real Estate Buyer Segmentation and Investment Profiling |
| Domain | Data Science & Machine Learning |
| Project Type | Machine Learning and Data Analytics |
| Application | Interactive Analytics Dashboard |
| Framework | Python and Streamlit |
| Algorithms | K-Means and Hierarchical Clustering |

---

**© 2026 Real Estate Buyer Segmentation & Investment Profiling Project**