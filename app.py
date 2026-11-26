# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path
from sklearn.decomposition import PCA


# ============================================================
# STEP 2: PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHART_TEMPLATE = "plotly_white"


# ============================================================
# STEP 3: DASHBOARD THEME
# ============================================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background: linear-gradient(
        135deg,
        #E8F4FF,
        #F8FBFF,
        #EAF8F4
    );
}

[data-testid="stMain"] {
    background-color: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}


/* Sidebar Background */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #102A43,
        #174E70,
        #087E8B
    );
}

[data-testid="stSidebar"] * {
    color: white;
}


/* Sidebar Menu Default Colour */

[data-testid="stSidebar"] [data-testid="stRadio"] label {

    background: rgba(255,255,255,0.08);

    border: 1px solid rgba(255,255,255,0.15);

    border-radius: 10px;

    padding: 9px;

    margin: 5px 0;

    transition: all 0.2s ease;
}


/* Sidebar Hover Colour */

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {

    background: #247BA0;

    border-color: #8FE3DF;

}


/* Selected Menu Colour */

[data-testid="stSidebar"] [data-testid="stRadio"]
label:has(input:checked) {

    background: linear-gradient(
        90deg,
        #00A896,
        #02C39A
    ) !important;

    border: 1px solid #B8FFF0 !important;

    color: #062F35 !important;

    font-weight: bold !important;

    box-shadow: 0 3px 12px rgba(0,0,0,0.22);
}

[data-testid="stSidebar"] [data-testid="stRadio"]
label:has(input:checked) * {

    color: #062F35 !important;

    font-weight: bold !important;

}


/* Main Headings */

h1, h2, h3 {

    color: #153B5B;

    font-weight: bold;

}


/* KPI Cards */

[data-testid="stMetric"] {

    background: linear-gradient(
        145deg,
        #FFFFFF,
        #EEF8FF
    );

    padding: 18px;

    border-radius: 14px;

    border: 1px solid #D6E4F0;

    box-shadow: 0 3px 8px rgba(0,0,0,0.08);

}


/* Buttons */

.stButton > button,
.stDownloadButton > button {

    background: linear-gradient(
        90deg,
        #176B87,
        #00A896
    );

    color: white;

    border-radius: 9px;

    border: none;

    font-weight: bold;

}

.stButton > button:hover,
.stDownloadButton > button:hover {

    background: #087E8B;

    color: white;

}


/* Divider */

hr {

    border-color: #B8D4E8;

}


/* Dashboard Header */

.hero {

    padding: 22px;

    border-radius: 18px;

    background: linear-gradient(
        110deg,
        #123B5D,
        #176B87,
        #00A896
    );

    color: white;

    margin-bottom: 15px;

    box-shadow: 0 5px 18px rgba(18,59,93,0.15);

}

.hero h1 {

    color: white !important;

}

.hero p {

    color: #E8FFFF !important;

}

</style>
""", unsafe_allow_html=True)


# ============================================================
# STEP 4: PROJECT PATHS
# ============================================================

project_path = Path(__file__).resolve().parent

final_path = project_path / "data" / "final"

processed_path = project_path / "data" / "processed"

reports_path = project_path / "reports"

buyers_path = final_path / "final_buyer_segmentation.csv"

summary_path = final_path / "final_cluster_summary.csv"

distribution_path = final_path / "final_cluster_distribution.csv"

property_path = project_path / "data" / "properties.csv"

evaluation_path = reports_path / "model_evaluation_comparison.csv"

optimization_path = reports_path / "kmeans_cluster_optimization.csv"

scaled_path = processed_path / "scaled_features.csv"

clustering_path = processed_path / "clustering_results.csv"


# ============================================================
# STEP 5: LOAD DATA
# ============================================================

@st.cache_data
def load_csv(path):

    if path.exists():

        return pd.read_csv(path)

    return None


buyers = load_csv(buyers_path)

cluster_summary = load_csv(summary_path)

cluster_distribution = load_csv(distribution_path)

properties = load_csv(property_path)

evaluation = load_csv(evaluation_path)

optimization = load_csv(optimization_path)

scaled_data = load_csv(scaled_path)

clustering_results = load_csv(clustering_path)


if buyers is None:

    st.error("Buyer dataset not found!")

    st.info(
        "Check data/final/final_buyer_segmentation.csv"
    )

    st.stop()


# ============================================================
# STEP 6: DATA PREPARATION
# ============================================================

buyers.columns = buyers.columns.str.strip()


if "cluster" not in buyers.columns:

    if "kmeans_cluster" in buyers.columns:

        buyers.rename(
            columns={"kmeans_cluster": "cluster"},
            inplace=True
        )

    else:

        st.error("Cluster column not found!")

        st.stop()


buyers["cluster"] = pd.to_numeric(
    buyers["cluster"],
    errors="coerce"
)

buyers = buyers.dropna(
    subset=["cluster"]
).copy()

buyers["cluster"] = buyers["cluster"].astype(int)


numeric_columns = [
    "satisfaction_score",
    "property_records",
    "unique_listings",
    "average_sale_price",
    "median_sale_price",
    "average_floor_area"
]

for col in numeric_columns:

    if col in buyers.columns:

        buyers[col] = pd.to_numeric(
            buyers[col],
            errors="coerce"
        )


# ============================================================
# STEP 7: SIDEBAR NAVIGATION
# ============================================================

pages = [

    "Overview",
    "Cluster Analysis",
    "Buyer Analysis",
    "Geographic Analysis",
    "Property Analysis",
    "Customer Satisfaction",
    "Cluster Insights",
    "ML Model Evaluation",
    "K-Means Optimization",
    "PCA Visualization",
    "Buyer Data Explorer",
    "About Project",
    "Technical Validation"

]


# ============================================================
# FILTER OPTIONS
# ============================================================

country_options = (
    sorted(buyers["country"].dropna().unique().tolist())
    if "country" in buyers.columns else []
)

all_region_options = (
    sorted(buyers["region"].dropna().unique().tolist())
    if "region" in buyers.columns else []
)

purpose_options = (
    sorted(buyers["acquisition_purpose"].dropna().unique().tolist())
    if "acquisition_purpose" in buyers.columns else []
)

client_options = (
    sorted(buyers["client_type"].dropna().unique().tolist())
    if "client_type" in buyers.columns else []
)

cluster_options = (
    sorted(buyers["cluster"].dropna().unique().tolist())
    if "cluster" in buyers.columns else []
)


# ============================================================
# RESET FILTER FUNCTION
# ============================================================

def reset_filters():

    st.session_state["country_filter"] = country_options

    st.session_state["region_filter"] = all_region_options

    st.session_state["purpose_filter"] = purpose_options

    st.session_state["client_filter"] = client_options

    st.session_state["cluster_filter"] = cluster_options


# ============================================================
# INITIALIZE FILTER VALUES
# ============================================================

if "country_filter" not in st.session_state:
    st.session_state["country_filter"] = country_options

if "region_filter" not in st.session_state:
    st.session_state["region_filter"] = all_region_options

if "purpose_filter" not in st.session_state:
    st.session_state["purpose_filter"] = purpose_options

if "client_filter" not in st.session_state:
    st.session_state["client_filter"] = client_options

if "cluster_filter" not in st.session_state:
    st.session_state["cluster_filter"] = cluster_options


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏠 Real Estate")

    st.caption(
        "Buyer Segmentation & Investment Profiling"
    )

    st.markdown("---")

    st.markdown("### 📚 Dashboard Menu")

    page = st.radio(
        "Select Section",
        pages,
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### 🎛️ Dashboard Filters")


    # --------------------------------------------------------
    # COUNTRY FILTER
    # --------------------------------------------------------

    selected_countries = st.multiselect(
        "Select Country",
        country_options,
        key="country_filter"
    )


    # --------------------------------------------------------
    # REGION FILTER
    # --------------------------------------------------------

    country_subset = buyers.copy()

    if "country" in buyers.columns:

        country_subset = buyers[
            buyers["country"].isin(selected_countries)
        ]

    region_options = (
        sorted(country_subset["region"].dropna().unique().tolist())
        if "region" in country_subset.columns else []
    )

    # Remove regions that are not available for selected countries

    st.session_state["region_filter"] = [
        region for region in st.session_state["region_filter"]
        if region in region_options
    ]

    selected_regions = st.multiselect(
        "Select Region",
        region_options,
        key="region_filter"
    )


    # --------------------------------------------------------
    # ACQUISITION PURPOSE
    # --------------------------------------------------------

    selected_purposes = st.multiselect(
        "Acquisition Purpose",
        purpose_options,
        key="purpose_filter"
    )


    # --------------------------------------------------------
    # BUYER TYPE
    # --------------------------------------------------------

    selected_clients = st.multiselect(
        "Buyer Type",
        client_options,
        key="client_filter"
    )


    # --------------------------------------------------------
    # CLUSTER FILTER
    # --------------------------------------------------------

    selected_clusters = st.multiselect(
        "Select Cluster",
        cluster_options,
        key="cluster_filter"
    )


    # --------------------------------------------------------
    # RESET BUTTON
    # --------------------------------------------------------

    st.markdown("---")

    st.button(
        "↩️ Reset Filters",
        use_container_width=True,
        on_click=reset_filters
    )


    # --------------------------------------------------------
    # SIDEBAR FOOTER
    # --------------------------------------------------------

    st.info("Real Estate Buyer Intelligence")

    st.caption("Machine Learning Project")
# ============================================================
# STEP 8: APPLY FILTERS
# ============================================================

filtered_buyers = buyers.copy()


if "country" in filtered_buyers.columns:

    filtered_buyers = filtered_buyers[
        filtered_buyers["country"].isin(selected_countries)
    ]


if "region" in filtered_buyers.columns:

    filtered_buyers = filtered_buyers[
        filtered_buyers["region"].isin(selected_regions)
    ]


if "acquisition_purpose" in filtered_buyers.columns:

    filtered_buyers = filtered_buyers[
        filtered_buyers["acquisition_purpose"].isin(selected_purposes)
    ]


if "client_type" in filtered_buyers.columns:

    filtered_buyers = filtered_buyers[
        filtered_buyers["client_type"].isin(selected_clients)
    ]


filtered_buyers = filtered_buyers[
    filtered_buyers["cluster"].isin(selected_clusters)
]


total_buyers = len(filtered_buyers)

total_clusters = filtered_buyers["cluster"].nunique()


if filtered_buyers.empty:

    st.warning(
        "No buyers found. Please change your dashboard filters."
    )

    st.stop()


# ============================================================
# STEP 9: PROFESSIONAL HELPER FUNCTIONS
# ============================================================

# 1. PROFESSIONAL CHART FUNCTION

def show_chart(fig, height=420):

    fig.update_layout(

        template=CHART_TEMPLATE,

        height=height,

        paper_bgcolor="rgba(255,255,255,0)",

        plot_bgcolor="rgba(255,255,255,0.75)",

        font=dict(
            family="Arial",
            size=13,
            color="#153B5B"
        ),

        title=dict(
            font=dict(
                size=19,
                color="#153B5B"
            ),
            x=0.02,
            xanchor="left"
        ),

        margin=dict(
            l=25,
            r=25,
            t=70,
            b=35
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=0
        ),

        hoverlabel=dict(
            bgcolor="#102A43",
            font_color="white",
            font_size=13
        )

    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#D6E4F0"
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E6EEF5",
        showline=False,
        zeroline=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------------------------------------------------
# 2. COUNT TABLE FUNCTION
# ------------------------------------------------------------

def count_table(data, column, label):

    return (

        data[column]

        .fillna("Unknown")

        .astype(str)

        .value_counts()

        .rename_axis(label)

        .reset_index(name="Buyers")

    )


# ------------------------------------------------------------
# 3. MONEY FORMAT FUNCTION
# ------------------------------------------------------------

def money(value):

    if pd.isna(value):

        return "N/A"

    return f"${value:,.0f}"


# ------------------------------------------------------------
# 4. SAFE NUMBER FORMAT FUNCTION
# ------------------------------------------------------------

def safe_number(value, decimals=2):

    if pd.isna(value):

        return "N/A"

    return f"{value:,.{decimals}f}"



# ============================================================
# STEP 10: DASHBOARD HEADER
# ============================================================

st.title("🏠 Real Estate Buyer Intelligence")

st.subheader(
    "Machine Learning Based Buyer Segmentation and Investment Profiling"
)

st.caption(
    "Smart Analytics | Buyer Insights | Machine Learning"
)

st.caption(
    f"📊 Viewing: {page} | Showing {total_buyers:,} Buyers"
)

st.divider()
# ============================================================
# STEP 11: OVERVIEW
# ============================================================

if page == "Overview":

    st.header("1. Buyer Segmentation Overview")

    col1, col2, col3, col4 = st.columns(4)

    countries = (
        filtered_buyers["country"].nunique()
        if "country" in filtered_buyers.columns
        else 0
    )

    regions = (
        filtered_buyers["region"].nunique()
        if "region" in filtered_buyers.columns
        else 0
    )

    with col1:

        st.metric(
            "Total Buyers",
            f"{total_buyers:,}"
        )

    with col2:

        st.metric(
            "Total Clusters",
            total_clusters
        )

    with col3:

        st.metric(
            "Countries",
            countries
        )

    with col4:

        st.metric(
            "Regions",
            regions
        )


    st.subheader("Buyer Dataset Preview")

    st.dataframe(
        filtered_buyers.head(10),
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Buyer Distribution by Cluster")

    cluster_counts = (
        filtered_buyers["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Buyers")
    )

    fig = px.bar(
        cluster_counts,
        x="Cluster",
        y="Buyers",
        color="Cluster",
        text="Buyers",
        title="Buyer Distribution by Cluster",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    show_chart(fig)


    if "acquisition_purpose" in filtered_buyers.columns:

        purpose = count_table(
            filtered_buyers,
            "acquisition_purpose",
            "Purpose"
        )

        fig = px.pie(
            purpose,
            names="Purpose",
            values="Buyers",
            hole=0.45,
            title="Acquisition Purpose Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        show_chart(fig)


# ============================================================
# STEP 12: CLUSTER ANALYSIS
# ============================================================

elif page == "Cluster Analysis":

    st.header("2. Cluster Distribution Analysis")

    cluster_counts = (
        filtered_buyers["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Buyers")
    )

    cluster_counts["Percentage"] = (
        cluster_counts["Buyers"] / len(filtered_buyers) * 100
    ).round(2)

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Buyers",
            color="Cluster",
            text="Buyers",
            title="Buyer Distribution by Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        show_chart(fig)

    with col2:

        fig = px.pie(
            cluster_counts,
            names="Cluster",
            values="Buyers",
            hole=0.45,
            title="Cluster Percentage Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        show_chart(fig)

    st.divider()

    st.subheader("3. Cluster Summary")

    summary_columns = [
        c for c in [
            "satisfaction_score",
            "average_sale_price",
            "median_sale_price",
            "average_floor_area",
            "property_records"
        ]
        if c in filtered_buyers.columns
    ]

    if summary_columns:

        current_summary = (
            filtered_buyers
            .groupby("cluster")[summary_columns]
            .mean(numeric_only=True)
            .reset_index()
        )

        buyer_counts = (
            filtered_buyers.groupby("cluster")
            .size()
            .reset_index(name="total_buyers")
        )

        current_summary = current_summary.merge(
            buyer_counts,
            on="cluster"
        )

        st.dataframe(
            current_summary,
            use_container_width=True,
            hide_index=True
        )

    elif cluster_summary is not None:

        st.dataframe(
            cluster_summary,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("Cluster summary not available.")


# ============================================================
# STEP 13: BUYER ANALYSIS
# ============================================================

elif page == "Buyer Analysis":

    st.header("4. Buyer Analysis")

    # BUYER TYPE

    st.subheader("4. Buyer Type Analysis")

    if "client_type" in filtered_buyers.columns:

        buyer_type = count_table(
            filtered_buyers,
            "client_type",
            "Buyer Type"
        )

        fig = px.bar(
            buyer_type,
            x="Buyer Type",
            y="Buyers",
            color="Buyer Type",
            text="Buyers",
            title="Buyer Type Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )

        show_chart(fig)

    # ACQUISITION PURPOSE

    st.divider()

    st.subheader("5. Acquisition Purpose Analysis")

    if "acquisition_purpose" in filtered_buyers.columns:

        purpose = count_table(
            filtered_buyers,
            "acquisition_purpose",
            "Purpose"
        )

        fig = px.pie(
            purpose,
            names="Purpose",
            values="Buyers",
            hole=0.4,
            title="Buyer Acquisition Purpose",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        show_chart(fig)

    # LOAN ANALYSIS

    st.divider()

    st.subheader("6. Loan Application Analysis")

    if "loan_applied" in filtered_buyers.columns:

        loan_data = count_table(
            filtered_buyers,
            "loan_applied",
            "Loan Status"
        )

        fig = px.bar(
            loan_data,
            x="Loan Status",
            y="Buyers",
            color="Loan Status",
            text="Buyers",
            title="Loan Application Distribution",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        show_chart(fig)

    # REFERRAL CHANNEL

    st.divider()

    st.subheader("7. Referral Channel Analysis")

    if "referral_channel" in filtered_buyers.columns:

        referral = count_table(
            filtered_buyers,
            "referral_channel",
            "Referral Channel"
        )

        fig = px.bar(
            referral,
            x="Referral Channel",
            y="Buyers",
            color="Referral Channel",
            text="Buyers",
            title="Buyer Referral Channel",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        show_chart(fig)


# ============================================================
# STEP 14: GEOGRAPHIC ANALYSIS
# ============================================================

elif page == "Geographic Analysis":

    st.header("8. Geographic Analysis")

    col1, col2 = st.columns(2)

    # COUNTRY ANALYSIS

    if "country" in filtered_buyers.columns:

        country_data = count_table(
            filtered_buyers,
            "country",
            "Country"
        )

        with col1:

            fig = px.bar(
                country_data,
                x="Country",
                y="Buyers",
                color="Country",
                title="Buyers by Country",
                color_discrete_sequence=px.colors.qualitative.Bold
            )

            fig.update_layout(
                xaxis_tickangle=-45
            )

            show_chart(fig)

    # REGION ANALYSIS

    if "region" in filtered_buyers.columns:

        region_data = count_table(
            filtered_buyers,
            "region",
            "Region"
        ).head(15)

        with col2:

            fig = px.bar(
                region_data,
                x="Region",
                y="Buyers",
                color="Region",
                title="Top 15 Buyer Regions",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig.update_layout(
                xaxis_tickangle=-45
            )

            show_chart(fig)

    # GLOBAL MAP

    st.divider()

    st.header("9. Global Buyer Distribution Map")

    if "country" in filtered_buyers.columns:

        map_data = (
            filtered_buyers
            .groupby("country")
            .size()
            .reset_index(name="Buyers")
        )

        country_mapping = {
            "USA": "United States",
            "UK": "United Kingdom"
        }

        map_data["country"] = map_data["country"].replace(
            country_mapping
        )

        fig = px.choropleth(
            map_data,
            locations="country",
            locationmode="country names",
            color="Buyers",
            hover_name="country",
            color_continuous_scale="Blues",
            title="Global Real Estate Buyer Distribution"
        )

        show_chart(fig)



# ============================================================
# STEP 15: PROPERTY ANALYSIS
# ============================================================

elif page == "Property Analysis":

    st.header("10. Property Analysis")

    if properties is None:

        st.warning("Properties dataset not found.")

    else:

        property_data = properties.copy()

        property_data.columns = (
            property_data.columns.str.strip()
        )

        # ----------------------------------------------------
        # FILTER PROPERTY DATA
        # ----------------------------------------------------

        if (
            "client_ref" in property_data.columns
            and "client_id" in filtered_buyers.columns
        ):

            buyer_ids = (
                filtered_buyers["client_id"]
                .astype(str)
                .str.strip()
                .unique()
            )

            property_data["client_ref_clean"] = (
                property_data["client_ref"]
                .astype(str)
                .str.strip()
            )

            property_data = property_data[
                property_data["client_ref_clean"].isin(buyer_ids)
            ].copy()

        # ----------------------------------------------------
        # DETECT COLUMNS
        # ----------------------------------------------------

        price_column = next(
            (
                col for col in [
                    "sale_price",
                    "price",
                    "property_price"
                ]
                if col in property_data.columns
            ),
            None
        )

        type_column = next(
            (
                col for col in [
                    "unit_category",
                    "property_type",
                    "unit_type",
                    "property_category"
                ]
                if col in property_data.columns
            ),
            None
        )

        area_column = next(
            (
                col for col in [
                    "floor_area_sqft",
                    "floor_area",
                    "area_sqft",
                    "property_area"
                ]
                if col in property_data.columns
            ),
            None
        )

        # ----------------------------------------------------
        # DATA CONVERSION
        # ----------------------------------------------------

        if price_column:

            property_data[price_column] = pd.to_numeric(
                property_data[price_column]
                .astype(str)
                .str.replace(r"[$,]", "", regex=True),
                errors="coerce"
            )

        if area_column:

            property_data[area_column] = pd.to_numeric(
                property_data[area_column],
                errors="coerce"
            )

        # ----------------------------------------------------
        # PROPERTY OVERVIEW
        # ----------------------------------------------------

        st.subheader("Property Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Property Records",
                f"{len(property_data):,}"
            )

        with col2:

            if "client_ref" in property_data.columns:

                st.metric(
                    "Unique Buyers",
                    f"{property_data['client_ref'].nunique():,}"
                )

        with col3:

            if price_column:

                st.metric(
                    "Average Sale Price",
                    money(property_data[price_column].mean())
                )

        with col4:

            if area_column:

                st.metric(
                    "Average Floor Area",
                    f"{property_data[area_column].mean():,.2f}"
                )

        # ----------------------------------------------------
        # PROPERTY SALE PRICE ANALYSIS
        # ----------------------------------------------------

        if price_column:

            st.divider()

            st.subheader("Property Sale Price Analysis")

            price_data = property_data.dropna(
                subset=[price_column]
            )

            if not price_data.empty:

                fig = px.histogram(
                    price_data,
                    x=price_column,
                    nbins=30,
                    title="Property Sale Price Distribution",
                    color_discrete_sequence=["#246B8E"],
                    labels={
                        price_column: "Sale Price"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Sale Price",
                    yaxis_title="Number of Properties"
                )

                show_chart(fig)

        # ----------------------------------------------------
        # PROPERTY TYPE DISTRIBUTION
        # ----------------------------------------------------

        if type_column:

            st.divider()

            st.subheader("Property Type Distribution")

            type_data = count_table(
                property_data,
                type_column,
                "Property Type"
            )

            if not type_data.empty:

                fig = px.pie(
                    type_data,
                    names="Property Type",
                    values="Buyers",
                    hole=0.45,
                    title="Property Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                show_chart(fig)

        # ----------------------------------------------------
        # PROPERTY FLOOR AREA ANALYSIS
        # ----------------------------------------------------

        if area_column:

            st.divider()

            st.subheader("Property Floor Area Analysis")

            area_data = property_data.dropna(
                subset=[area_column]
            )

            if not area_data.empty:

                fig = px.histogram(
                    area_data,
                    x=area_column,
                    nbins=30,
                    title="Property Floor Area Distribution",
                    color_discrete_sequence=["#17A589"],
                    labels={
                        area_column: "Floor Area"
                    }
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Floor Area",
                    yaxis_title="Number of Properties"
                )

                show_chart(fig)

        # ----------------------------------------------------
        # PROPERTY PRICE VS FLOOR AREA
        # ----------------------------------------------------

        if price_column and area_column:

            st.divider()

            st.subheader("Property Price vs Floor Area")

            scatter_data = property_data.dropna(
                subset=[price_column, area_column]
            )

            if not scatter_data.empty:

                fig = px.scatter(
                    scatter_data,
                    x=area_column,
                    y=price_column,
                    color=type_column if type_column else None,
                    title="Property Price vs Floor Area",
                    labels={
                        area_column: "Floor Area",
                        price_column: "Sale Price"
                    },
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                fig.update_layout(
                    template="plotly_white"
                )

                show_chart(fig)

        # ----------------------------------------------------
        # PROPERTY DATA EXPLORER
        # ----------------------------------------------------

        st.divider()

        st.subheader("Property Dataset Preview")

        st.caption(
            f"Showing {min(len(property_data), 500):,} "
            f"of {len(property_data):,} property records"
        )

        st.dataframe(
            property_data.head(500),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# STEP 16: CUSTOMER SATISFACTION
# ============================================================

elif page == "Customer Satisfaction":

    st.header("11. Customer Satisfaction Analysis")

    if "satisfaction_score" in filtered_buyers.columns:

        satisfaction = filtered_buyers.copy()

        satisfaction["satisfaction_score"] = pd.to_numeric(
            satisfaction["satisfaction_score"],
            errors="coerce"
        )

        satisfaction = satisfaction.dropna(
            subset=["satisfaction_score"]
        )

        if satisfaction.empty:

            st.warning("No satisfaction scores available.")

        else:

            # ------------------------------------------------
            # SATISFACTION OVERVIEW
            # ------------------------------------------------

            st.subheader("Customer Satisfaction Overview")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Average Satisfaction",
                    f"{satisfaction['satisfaction_score'].mean():.2f} / 5"
                )

            with col2:

                st.metric(
                    "Highest Score",
                    satisfaction["satisfaction_score"].max()
                )

            with col3:

                st.metric(
                    "Lowest Score",
                    satisfaction["satisfaction_score"].min()
                )

            # ------------------------------------------------
            # SATISFACTION SCORE DISTRIBUTION
            # ------------------------------------------------

            st.divider()

            st.subheader("Customer Satisfaction Score Distribution")

            score_data = (
                satisfaction["satisfaction_score"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            score_data.columns = [
                "Satisfaction Score",
                "Buyers"
            ]

            score_data["Percentage"] = (
                score_data["Buyers"]
                / score_data["Buyers"].sum()
                * 100
            )

            col1, col2 = st.columns(2)

            with col1:

                fig = px.bar(
                    score_data,
                    x="Satisfaction Score",
                    y="Buyers",
                    color="Satisfaction Score",
                    title="Satisfaction Score Distribution",
                    text="Buyers",
                    color_continuous_scale="Viridis"
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Satisfaction Score",
                    yaxis_title="Number of Buyers"
                )

                show_chart(fig)

            with col2:

                fig = px.pie(
                    score_data,
                    names="Satisfaction Score",
                    values="Buyers",
                    hole=0.45,
                    title="Customer Satisfaction Percentage",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                fig.update_traces(
                    textposition="inside",
                    textinfo="percent+label"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                show_chart(fig)

            # ------------------------------------------------
            # AVERAGE SATISFACTION BY CLUSTER
            # ------------------------------------------------

            if "cluster" in satisfaction.columns:

                st.divider()

                st.subheader("Average Satisfaction by Cluster")

                cluster_satisfaction = (
                    satisfaction
                    .groupby("cluster")["satisfaction_score"]
                    .mean()
                    .reset_index()
                )

                cluster_satisfaction.columns = [
                    "Cluster",
                    "Average Satisfaction"
                ]

                fig = px.bar(
                    cluster_satisfaction,
                    x="Cluster",
                    y="Average Satisfaction",
                    color="Cluster",
                    title="Average Satisfaction by Cluster",
                    text_auto=".2f",
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Buyer Cluster",
                    yaxis_title="Average Satisfaction",
                    yaxis_range=[0, 5]
                )

                show_chart(fig)

            # ------------------------------------------------
            # SATISFACTION DATA TABLE
            # ------------------------------------------------

            st.divider()

            st.subheader("Satisfaction Data Summary")

            st.dataframe(
                score_data,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.warning("Satisfaction score not available.")


# ============================================================
# STEP 17: CLUSTER INSIGHTS
# ============================================================

elif page == "Cluster Insights":

    st.header("12. Cluster Insights")

    for column, display_name in [

        ("client_type", "Buyer Type"),

        ("acquisition_purpose", "Acquisition Purpose"),

        ("loan_applied", "Loan Applied"),

        ("country", "Country"),

        ("region", "Region")

    ]:

        if column in filtered_buyers.columns:

            result = (
                filtered_buyers[column]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
                .rename_axis(display_name)
                .reset_index(name="Buyers")
            )

            result["Percentage"] = (
                result["Buyers"] /
                len(filtered_buyers) * 100
            ).round(2)

            st.subheader(
                display_name + " Distribution"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True
                )

            with col2:

                fig = px.bar(
                    result,
                    x=display_name,
                    y="Buyers",
                    color=display_name,
                    title=display_name + " Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )

                show_chart(fig)


# ============================================================
# STEP 18: ML MODEL EVALUATION
# ============================================================

elif page == "ML Model Evaluation":

    st.header("13. Machine Learning Model Evaluation")

    if evaluation is not None:

        evaluation.columns = evaluation.columns.str.strip()

        st.subheader("Model Performance Metrics")

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True
        )

        model_column = next(
            (
                col for col in [
                    "Model",
                    "model",
                    "Algorithm",
                    "algorithm"
                ]
                if col in evaluation.columns
            ),
            None
        )

        metric_columns = [
            col for col in evaluation.columns
            if any(
                term in col.lower()
                for term in [
                    "silhouette",
                    "davies",
                    "calinski"
                ]
            )
        ]

        if model_column:

            for metric in metric_columns:

                st.subheader(metric + " Comparison")

                fig = px.bar(
                    evaluation,
                    x=model_column,
                    y=metric,
                    color=model_column,
                    title=metric + " Comparison",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    text_auto=True
                )

                show_chart(fig)

        st.info(
            "Model evaluation reports show original model results. "
            "They are not recalculated when dashboard filters change."
        )

    else:

        st.warning("Model evaluation report not found.")


# ============================================================
# STEP 19: K-MEANS OPTIMIZATION
# ============================================================

elif page == "K-Means Optimization":

    st.header("14. K-Means Cluster Optimization")

    if optimization is not None:

        optimization.columns = optimization.columns.str.strip()

        st.subheader("K-Means Optimization Results")

        st.dataframe(
            optimization,
            use_container_width=True,
            hide_index=True
        )

        k_column = next(
            (
                col for col in [
                    "k",
                    "n_clusters",
                    "clusters",
                    "K"
                ]
                if col in optimization.columns
            ),
            None
        )

        silhouette_column = next(
            (
                col for col in optimization.columns
                if "silhouette" in col.lower()
            ),
            None
        )

        if k_column and silhouette_column:

            optimization[silhouette_column] = pd.to_numeric(
                optimization[silhouette_column],
                errors="coerce"
            )

            valid_rows = optimization.dropna(
                subset=[silhouette_column]
            )

            if not valid_rows.empty:

                fig = px.line(
                    valid_rows,
                    x=k_column,
                    y=silhouette_column,
                    markers=True,
                    title="Silhouette Score for Different K Values",
                    color_discrete_sequence=["#246B8E"]
                )

                show_chart(fig)

                best_row = valid_rows.loc[
                    valid_rows[silhouette_column].idxmax()
                ]

                st.success(
                    f"Best K Value: {best_row[k_column]} | "
                    f"Silhouette Score: "
                    f"{best_row[silhouette_column]:.4f}"
                )

    else:

        st.warning("K-Means optimization report not found.")


# ============================================================
# STEP 20: PCA VISUALIZATION
# ============================================================

elif page == "PCA Visualization":

    st.header("15. Principal Component Analysis (PCA)")

    if scaled_data is not None and clustering_results is not None:

        try:

            numeric_data = scaled_data.select_dtypes(
                include=["number"]
            ).copy()

            numeric_data = numeric_data.fillna(0)

            if numeric_data.empty:

                st.warning("No numeric scaled features found.")

            elif len(numeric_data) < 2:

                st.warning("Not enough rows for PCA.")

            else:

                pca = PCA(n_components=2)

                components = pca.fit_transform(
                    numeric_data
                )

                pca_df = pd.DataFrame(
                    components,
                    columns=["PC1", "PC2"]
                )

                if len(pca_df) == len(clustering_results):

                    found_cluster = False

                    for cluster_col in [
                        "kmeans_cluster",
                        "hierarchical_cluster"
                    ]:

                        if cluster_col in clustering_results.columns:

                            found_cluster = True

                            plot_data = pca_df.copy()

                            plot_data["Cluster"] = (
                                clustering_results[cluster_col]
                                .astype(str)
                                .values
                            )

                            fig = px.scatter(
                                plot_data,
                                x="PC1",
                                y="PC2",
                                color="Cluster",
                                title=cluster_col.replace(
                                    "_", " "
                                ).title() + " PCA Visualization",
                                color_discrete_sequence=px.colors.qualitative.Bold
                            )

                            show_chart(fig)

                    if not found_cluster:

                        st.warning("Cluster columns not found.")

                    variance = (
                        pca.explained_variance_ratio_.sum() * 100
                    )

                    st.info(
                        f"Total Variance Explained by 2 PCA Components: "
                        f"{variance:.2f}%"
                    )

                    st.caption(
                        "PCA is calculated on the original processed dataset, "
                        "not on the selected sidebar filters."
                    )

                else:

                    st.warning(
                        "Scaled features and clustering results "
                        "have different row counts."
                    )

        except Exception as error:

            st.error(
                f"PCA Visualization Error: {error}"
            )

    else:

        st.warning(
            "Scaled features or clustering results not found."
        )


# ============================================================
# STEP 21: BUYER DATA EXPLORER
# ============================================================

elif page == "Buyer Data Explorer":

    st.header("16. Buyer Data Explorer")

    st.subheader("Complete Buyer Dataset")

    st.metric(
        "Total Buyer Records",
        f"{len(filtered_buyers):,}"
    )

    st.dataframe(
        filtered_buyers,
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_buyers.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download Buyer Dataset CSV",
        data=csv_data,
        file_name="real_estate_buyer_segmentation.csv",
        mime="text/csv"
    )


# ============================================================
# STEP 22: ABOUT PROJECT
# ============================================================

elif page == "About Project":

    st.header("17. About This Project")

    st.markdown("""
    ### Project Overview

    This project uses Machine Learning techniques to
    segment real estate buyers and analyze their investment profiles.

    ### Main Objectives

    - Understand buyer behavior.
    - Identify buyer segments.
    - Analyze customer satisfaction.
    - Study property transaction patterns.
    - Understand geographic buyer distribution.
    - Compare clustering model performance.

    ### Machine Learning Algorithms

    - K-Means Clustering
    - Hierarchical Clustering
    - Principal Component Analysis (PCA)

    ### Technologies Used

    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - Plotly
    - NumPy

    ### Data Limitations

    - Property records may not be available for every buyer.
    - Small clusters may represent niche groups or outliers.
    - Property sale price is not the same as buyer income.
    - PCA two-dimensional plots show only part of the original
      feature information.
    - Clustering results represent patterns in the available data,
      not guaranteed buyer wealth categories.
    """)


# ============================================================
# STEP 23: TECHNICAL VALIDATION
# ============================================================

elif page == "Technical Validation":

    st.header("18. Technical Validation")

    validation_data = {

        "Validation Check": [

            "Buyer Dataset Loaded",

            "Cluster Column Available",

            "Total Buyers Before Filters",

            "Total Buyers After Filters",

            "Total Clusters",

            "Missing Cluster Values",

            "Property Dataset Available",

            "Model Evaluation Report Available",

            "Optimization Report Available",

            "Scaled Features Available",

            "Clustering Results Available"

        ],

        "Status": [

            "PASS",

            "PASS",

            len(buyers),

            len(filtered_buyers),

            filtered_buyers["cluster"].nunique(),

            filtered_buyers["cluster"].isna().sum(),

            "YES" if properties is not None else "NO",

            "YES" if evaluation is not None else "NO",

            "YES" if optimization is not None else "NO",

            "YES" if scaled_data is not None else "NO",

            "YES" if clustering_results is not None else "NO"

        ]

    }

    validation_df = pd.DataFrame(validation_data)

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# STEP 24: FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:20px;
        border-radius:14px;
        background:linear-gradient(
            90deg,
            #E5F4FB,
            #E4F8F3
        );
    ">

    <h3>🏠 Real Estate Buyer Intelligence Dashboard</h3>

    <p>
    Machine Learning Based Buyer Segmentation and
    Investment Profiling
    </p>

    <p>
    Developed using Python, Streamlit, Pandas,
    Scikit-learn and Plotly
    </p>

    <p>
    © 2026 Real Estate Buyer Segmentation Project
    </p>

    </div>
    """,
    unsafe_allow_html=True
)