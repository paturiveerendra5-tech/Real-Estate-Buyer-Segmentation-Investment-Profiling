# Project Insights

## 1. Project Overview
This project focuses on Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence.

The objective is to identify buyer groups and understand their purchasing patterns using demographic, geographic, financing, and property transaction information.

## 2. Dataset Summary
- Total Buyers: 2,000
- Total Property Records: 10,000
- Countries: 10
- Regions: 57
- Linked Property Records: 7,305
- Missing Client References in Properties: 2,695

## 3. Key Findings

### Buyer Type
- Individual buyers: 1,897
- Company buyers: 103
- Individual buyers represent the majority of the dataset.

### Acquisition Purpose
- Home buyers: 1,385
- Investment buyers: 615
- Most buyers purchased properties for residential purposes.

### Loan Analysis
- Buyers who applied for loans: 736
- Buyers who did not apply for loans: 1,264
- Approximately 36.8% of buyers applied for loans.

### Geographic Analysis
- USA has the highest number of buyers with 1,538.
- Canada has 85 buyers.
- UK has 95 buyers.
- California has the highest regional buyer count with 633.

### Property Analysis
- Average Sale Price: $345,072
- Average Floor Area: 1,141.15 sq. ft.
- Apartment is the most common property category.

## 4. Machine Learning Results

K-Means Clustering:
- Number of Clusters: 4
- Silhouette Score: 0.4055

Hierarchical Clustering:
- Number of Clusters: 2
- Silhouette Score: 0.3779

K-Means achieved the higher silhouette score among the evaluated models.

## 5. Cluster Distribution

- Cluster 0: 356 buyers
- Cluster 1: 1,623 buyers
- Cluster 2: 10 buyers
- Cluster 3: 11 buyers

Cluster 1 represents the largest buyer segment. Clusters 2 and 3 contain very small groups and require cautious interpretation.

## 6. Recommendations
- Focus marketing strategies on the large individual buyer segment.
- Develop separate marketing campaigns for home buyers and investment buyers.
- Use geographic insights to understand regional demand.
- Study loan application patterns to improve buyer support.
- Investigate the small clusters further before making business decisions.

## 7. Limitations
- The dataset does not contain buyer income information.
- Sale price represents property transaction value, not buyer income.
- Some property records do not have client references.
- The clustering results are imbalanced, with two very small clusters.

## 8. Conclusion
The project demonstrates how machine learning and data analytics can help identify real estate buyer segments, understand purchasing behavior, and support data-driven marketing and investment decisions.