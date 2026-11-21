# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# STEP 2: STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_TEMPLATE = "plotly_white"


# ============================================================
# STEP 3: CUSTOM CSS DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f5f7fb;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #142b4a;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.section-note {
    color: #64748b;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 4: DEFINE PROJECT PATHS
# ============================================================

project_path = Path(__file__).resolve().parent

final_path = project_path / "data" / "final"

processed_path = project_path / "data" / "processed"

property_path = project_path / "data" / "properties.csv"


# ============================================================
# STEP 5: LOAD DATASETS
# ============================================================

@st.cache_data
def load_data():

    buyers = pd.read_csv(
        final_path / "final_buyer_segmentation.csv"
    )

    cluster_summary = pd.read_csv(
        final_path / "final_cluster_summary.csv"
    )

    cluster_distribution = pd.read_csv(
        final_path / "final_cluster_distribution.csv"
    )

    return buyers, cluster_summary, cluster_distribution


try:

    buyers, cluster_summary, cluster_distribution = load_data()

    buyers.columns = buyers.columns.str.strip()

except Exception as e:

    st.error(f"Dataset Loading Error: {e}")
    st.stop()


# ============================================================
# STEP 6: DATA VALIDATION
# ============================================================

if "cluster" in buyers.columns:

    cluster_column = "cluster"

elif "kmeans_cluster" in buyers.columns:

    cluster_column = "kmeans_cluster"

else:

    st.error("Cluster column not found in buyer dataset.")
    st.stop()


buyers[cluster_column] = pd.to_numeric(
    buyers[cluster_column],
    errors="coerce"
)

buyers = buyers.dropna(
    subset=[cluster_column]
).copy()

buyers[cluster_column] = buyers[cluster_column].astype(int)


# ============================================================
# STEP 7: DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Real Estate Buyer Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Buyer Segmentation and Investment Profiling'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

st.info(
    "Explore real estate buyer profiles, geographic patterns, "
    "property transactions and machine learning clusters."
)


# ============================================================
# STEP 8: SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🎛️ Dashboard Filters")

filtered_buyers = buyers.copy()


# Country Filter

if "country" in filtered_buyers.columns:

    countries = sorted(
        filtered_buyers["country"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_countries = st.sidebar.multiselect(
        "Select Country",
        countries,
        default=countries
    )

    filtered_buyers = filtered_buyers[
        filtered_buyers["country"].isin(selected_countries)
    ]


# Region Filter

if "region" in filtered_buyers.columns:

    regions = sorted(
        filtered_buyers["region"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_regions = st.sidebar.multiselect(
        "Select Region",
        regions,
        default=regions
    )

    filtered_buyers = filtered_buyers[
        filtered_buyers["region"].isin(selected_regions)
    ]


# Acquisition Purpose Filter

if "acquisition_purpose" in filtered_buyers.columns:

    purposes = sorted(
        filtered_buyers["acquisition_purpose"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_purposes = st.sidebar.multiselect(
        "Acquisition Purpose",
        purposes,
        default=purposes
    )

    filtered_buyers = filtered_buyers[
        filtered_buyers["acquisition_purpose"].isin(selected_purposes)
    ]


# Client Type Filter

if "client_type" in filtered_buyers.columns:

    client_types = sorted(
        filtered_buyers["client_type"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_client_types = st.sidebar.multiselect(
        "Client Type",
        client_types,
        default=client_types
    )

    filtered_buyers = filtered_buyers[
        filtered_buyers["client_type"].isin(selected_client_types)
    ]


st.sidebar.divider()

st.sidebar.metric(
    "Filtered Buyers",
    f"{len(filtered_buyers):,}"
)


if filtered_buyers.empty:

    st.warning("No buyers found. Please change the filters.")

    st.stop()


# ============================================================
# STEP 9: BUYER SEGMENTATION OVERVIEW
# ============================================================

st.divider()

st.header("1. Buyer Segmentation Overview")

total_buyers = len(filtered_buyers)

total_clusters = filtered_buyers[cluster_column].nunique()

total_countries = (
    filtered_buyers["country"].nunique()
    if "country" in filtered_buyers.columns else 0
)

total_regions = (
    filtered_buyers["region"].nunique()
    if "region" in filtered_buyers.columns else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Buyers", f"{total_buyers:,}")

col2.metric("Total Clusters", total_clusters)

col3.metric("Countries", total_countries)

col4.metric("Regions", total_regions)


# ============================================================
# STEP 10: CLUSTER DISTRIBUTION ANALYSIS
# ============================================================

st.divider()

st.header("2. Cluster Distribution Analysis")

cluster_counts = (
    filtered_buyers[cluster_column]
    .value_counts()
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Buyers")
)

cluster_counts["Cluster"] = (
    cluster_counts["Cluster"].astype(str)
)

col1, col2 = st.columns(2)

with col1:

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Buyers",
        text="Buyers",
        title="Buyer Distribution by Cluster",
        color="Cluster",
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")


with col2:

    fig = px.pie(
        cluster_counts,
        names="Cluster",
        values="Buyers",
        title="Cluster Percentage Distribution",
        hole=0.45,
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")


st.caption(
    "Note: Cluster numbers are machine-learning labels. "
    "They do not represent wealth categories."
)


# ============================================================
# STEP 11: BUYER TYPE ANALYSIS
# ============================================================

st.divider()

st.header("3. Buyer Type Analysis")

if "client_type" in filtered_buyers.columns:

    client_type_data = (
        filtered_buyers["client_type"]
        .value_counts()
        .rename_axis("Client Type")
        .reset_index(name="Buyers")
    )

    fig = px.bar(
        client_type_data,
        x="Client Type",
        y="Buyers",
        text="Buyers",
        color="Client Type",
        title="Individual vs Company Buyers",
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")

else:

    st.info("Client type column is not available.")


# ============================================================
# STEP 12: ACQUISITION PURPOSE ANALYSIS
# ============================================================

st.divider()

st.header("4. Acquisition Purpose Analysis")

if "acquisition_purpose" in filtered_buyers.columns:

    purpose_data = (
        filtered_buyers["acquisition_purpose"]
        .value_counts()
        .rename_axis("Purpose")
        .reset_index(name="Buyers")
    )

    fig = px.pie(
        purpose_data,
        names="Purpose",
        values="Buyers",
        title="Home vs Investment Purpose",
        hole=0.45,
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")

else:

    st.info("Acquisition purpose column is not available.")


# ============================================================
# STEP 13: LOAN APPLICATION ANALYSIS
# ============================================================

st.divider()

st.header("5. Loan Application Analysis")

if "loan_applied" in filtered_buyers.columns:

    loan_data = (
        filtered_buyers["loan_applied"]
        .value_counts()
        .rename_axis("Loan Status")
        .reset_index(name="Buyers")
    )

    fig = px.bar(
        loan_data,
        x="Loan Status",
        y="Buyers",
        text="Buyers",
        color="Loan Status",
        title="Loan Application Distribution",
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")

else:

    st.info("Loan application column is not available.")


# ============================================================
# STEP 14: GEOGRAPHIC BUYER ANALYSIS
# ============================================================

st.divider()

st.header("6. Geographic Buyer Analysis")

col1, col2 = st.columns(2)


# Country Analysis

with col1:

    if "country" in filtered_buyers.columns:

        country_data = (
            filtered_buyers["country"]
            .value_counts()
            .head(10)
            .rename_axis("Country")
            .reset_index(name="Buyers")
        )

        fig = px.bar(
            country_data,
            x="Buyers",
            y="Country",
            orientation="h",
            title="Top 10 Countries",
            template=CHART_TEMPLATE
        )

        st.plotly_chart(fig, width="stretch")


# Region Analysis

with col2:

    if "region" in filtered_buyers.columns:

        region_data = (
            filtered_buyers["region"]
            .value_counts()
            .head(10)
            .rename_axis("Region")
            .reset_index(name="Buyers")
        )

        fig = px.bar(
            region_data,
            x="Buyers",
            y="Region",
            orientation="h",
            title="Top 10 Regions",
            template=CHART_TEMPLATE
        )

        st.plotly_chart(fig, width="stretch")


# ============================================================
# STEP 15: GLOBAL BUYER DISTRIBUTION MAP
# ============================================================

st.subheader("🌍 Global Buyer Distribution")

if "country" in filtered_buyers.columns:

    map_data = (
        filtered_buyers["country"]
        .dropna()
        .value_counts()
        .rename_axis("Country")
        .reset_index(name="Buyers")
    )

    map_data["Country"] = map_data["Country"].replace({
        "USA": "United States",
        "UK": "United Kingdom"
    })

    fig_map = px.choropleth(
        map_data,
        locations="Country",
        locationmode="country names",
        color="Buyers",
        hover_name="Country",
        color_continuous_scale="Blues",
        title="Global Buyer Distribution by Country",
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig_map, width="stretch")

else:

    st.warning("Country column is not available.")


# ============================================================
# STEP 16: COUNTRY-WISE CLUSTER ANALYSIS
# ============================================================

st.divider()

st.header("7. Country-wise Cluster Analysis")

if "country" in filtered_buyers.columns:

    country_cluster_analysis = pd.crosstab(
        filtered_buyers["country"],
        filtered_buyers[cluster_column]
    )

    st.dataframe(
        country_cluster_analysis,
        width="stretch"
    )

else:

    st.info("Country column is not available.")


# ============================================================
# STEP 17: PROPERTY ANALYSIS
# ============================================================

st.divider()

st.header("8. Property Analysis")

if property_path.exists():

    properties = pd.read_csv(property_path)

    properties.columns = properties.columns.str.strip()

    if (
        "client_ref" in properties.columns
        and "client_id" in filtered_buyers.columns
    ):

        buyer_ids = (
            filtered_buyers["client_id"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        properties["client_ref_clean"] = (
            properties["client_ref"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        filtered_properties = properties[
            properties["client_ref_clean"].isin(buyer_ids)
        ].copy()

        st.metric(
            "Filtered Property Records",
            f"{len(filtered_properties):,}"
        )

        if not filtered_properties.empty:

            # Sale Price Analysis

            if "sale_price" in filtered_properties.columns:

                filtered_properties["sale_price"] = pd.to_numeric(
                    filtered_properties["sale_price"]
                    .astype(str)
                    .str.replace(r"[$,]", "", regex=True),
                    errors="coerce"
                )

                valid_prices = (
                    filtered_properties["sale_price"].dropna()
                )

                if not valid_prices.empty:

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Average Sale Price",
                        f"${valid_prices.mean():,.2f}"
                    )

                    col2.metric(
                        "Minimum Sale Price",
                        f"${valid_prices.min():,.2f}"
                    )

                    col3.metric(
                        "Maximum Sale Price",
                        f"${valid_prices.max():,.2f}"
                    )

                    fig = px.histogram(
                        x=valid_prices,
                        nbins=30,
                        title="Property Sale Price Distribution",
                        labels={"x": "Sale Price (USD)"},
                        template=CHART_TEMPLATE
                    )

                    st.plotly_chart(fig, width="stretch")


            # Unit Category Analysis

            if "unit_category" in filtered_properties.columns:

                unit_data = (
                    filtered_properties["unit_category"]
                    .value_counts()
                    .rename_axis("Unit Category")
                    .reset_index(name="Count")
                )

                fig = px.pie(
                    unit_data,
                    names="Unit Category",
                    values="Count",
                    title="Property Unit Category",
                    hole=0.4,
                    template=CHART_TEMPLATE
                )

                st.plotly_chart(fig, width="stretch")


            # Floor Area Analysis

            if "floor_area_sqft" in filtered_properties.columns:

                filtered_properties["floor_area_sqft"] = pd.to_numeric(
                    filtered_properties["floor_area_sqft"],
                    errors="coerce"
                )

                valid_area = (
                    filtered_properties["floor_area_sqft"].dropna()
                )

                if not valid_area.empty:

                    fig = px.histogram(
                        x=valid_area,
                        nbins=30,
                        title="Property Floor Area Distribution",
                        labels={"x": "Floor Area (sqft)"},
                        template=CHART_TEMPLATE
                    )

                    st.plotly_chart(fig, width="stretch")

        else:

            st.info(
                "No linked property records found for selected buyers."
            )

    else:

        st.warning(
            "Client ID or client_ref column is not available."
        )

else:

    st.warning("Properties dataset file was not found.")


# ============================================================
# STEP 18: CUSTOMER SATISFACTION ANALYSIS
# ============================================================

st.divider()

st.header("9. Customer Satisfaction Analysis")

if "satisfaction_score" in filtered_buyers.columns:

    satisfaction_data = (
        filtered_buyers["satisfaction_score"]
        .value_counts()
        .sort_index()
        .rename_axis("Satisfaction Score")
        .reset_index(name="Buyers")
    )

    fig = px.bar(
        satisfaction_data,
        x="Satisfaction Score",
        y="Buyers",
        text="Buyers",
        title="Customer Satisfaction Distribution",
        template=CHART_TEMPLATE
    )

    st.plotly_chart(fig, width="stretch")

else:

    st.info("Satisfaction score column is not available.")


# ============================================================
# STEP 19: CLUSTER INSIGHTS
# ============================================================

st.divider()

st.header("10. Cluster Insights")


def show_distribution(data, column, display_name):

    if column in data.columns:

        result = (
            data[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .rename_axis(display_name)
            .reset_index(name="Buyers")
        )

        result["Percentage"] = (
            result["Buyers"] / len(data) * 100
        ).round(2)

        st.dataframe(
            result,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(f"{column} column is not available.")


for cluster_id in sorted(
    filtered_buyers[cluster_column].unique()
):

    cluster_data = filtered_buyers[
        filtered_buyers[cluster_column] == cluster_id
    ].copy()

    with st.expander(
        f"Cluster {cluster_id} Insights",
        expanded=False
    ):

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Buyers",
            f"{len(cluster_data):,}"
        )

        col2.metric(
            "Countries",
            cluster_data["country"].nunique()
            if "country" in cluster_data.columns else 0
        )

        col3.metric(
            "Regions",
            cluster_data["region"].nunique()
            if "region" in cluster_data.columns else 0
        )

        st.write("### Buyer Type Distribution")

        show_distribution(
            cluster_data,
            "client_type",
            "Client Type"
        )

        st.write("### Acquisition Purpose")

        show_distribution(
            cluster_data,
            "acquisition_purpose",
            "Purpose"
        )

        st.write("### Loan Application")

        show_distribution(
            cluster_data,
            "loan_applied",
            "Loan Status"
        )

        st.write("### Country Distribution")

        show_distribution(
            cluster_data,
            "country",
            "Country"
        )

        st.write("### Region Distribution")

        show_distribution(
            cluster_data,
            "region",
            "Region"
        )

        if "satisfaction_score" in cluster_data.columns:

            satisfaction = pd.to_numeric(
                cluster_data["satisfaction_score"],
                errors="coerce"
            )

            average_satisfaction = satisfaction.mean()

            if pd.notna(average_satisfaction):

                st.metric(
                    "Average Satisfaction Score",
                    round(average_satisfaction, 2)
                )


# ============================================================
# STEP 20: FILTERED BUYER DATA AND CSV DOWNLOAD
# ============================================================

st.divider()

st.header("11. Filtered Buyer Data")

st.markdown(
    '<p class="section-note">'
    'Review the buyer records included in the current dashboard view '
    'and download them as a CSV file.'
    '</p>',
    unsafe_allow_html=True
)

st.dataframe(
    filtered_buyers,
    width="stretch",
    hide_index=True
)

csv_data = filtered_buyers.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Buyer Data (CSV)",
    data=csv_data,
    file_name="filtered_real_estate_buyers.csv",
    mime="text/csv"
)


# ============================================================
# STEP 21: PROJECT INFORMATION AND LIMITATIONS
# ============================================================

st.divider()

st.header("12. About This Project")

info_left, info_right = st.columns(2)

with info_left:

    st.markdown("""
    **Project Objective**

    Use machine-learning clustering to explore real estate
    buyer profiles and purchasing patterns.

    **Algorithms and Methods**

    - K-Means Clustering
    - Hierarchical Clustering
    - Principal Component Analysis (PCA)
    - Silhouette Score
    """)

with info_right:

    st.markdown("""
    **Technology Stack**

    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Plotly
    - Streamlit

    **Buyer Information**

    - Buyer type and acquisition purpose
    - Country and region
    - Loan application status
    - Satisfaction score
    - Linked property transaction details
    """)


st.warning(
    "Important data limitation: the dataset does not contain a buyer "
    "income field. Sale price represents property transaction price, "
    "not buyer income or personal wealth. Cluster labels should not "
    "be interpreted as wealth categories. Some clusters contain "
    "very few buyers."
)


# ============================================================
# STEP 22: TECHNICAL DATA VALIDATION
# ============================================================

st.divider()

with st.expander("Technical Data Validation"):

    validation1, validation2, validation3 = st.columns(3)

    validation1.metric(
        "Filtered Buyer Rows",
        f"{len(filtered_buyers):,}"
    )

    validation2.metric(
        "Unique Client IDs",
        f"{filtered_buyers['client_id'].nunique():,}"
        if "client_id" in filtered_buyers.columns
        else "Not available"
    )

    validation3.metric(
        "Missing Cluster Values",
        f"{filtered_buyers[cluster_column].isna().sum():,}"
    )

    st.write("Buyer Dataset Shape:", buyers.shape)

    st.write(
        "Available Buyer Columns:",
        ", ".join(buyers.columns)
    )

    st.write(
        "Cluster Summary Dataset Shape:",
        cluster_summary.shape
    )

    st.write(
        "Cluster Distribution Dataset Shape:",
        cluster_distribution.shape
    )


# ============================================================
# STEP 23: DASHBOARD FOOTER
# ============================================================

st.divider()

st.success(
    "REAL ESTATE BUYER SEGMENTATION DASHBOARD LOADED SUCCESSFULLY!"
)

st.caption(
    "Real Estate Buyer Intelligence Dashboard | "
    "Machine Learning Project"
)