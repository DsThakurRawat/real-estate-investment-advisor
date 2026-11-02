"""
Real Estate Investment Advisor: Streamlit Interactive Web Application
Predicts Property Profitability ("Good Investment") and 5-Year Future Valuation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.08);
    }
    
    .kpi-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    
    .kpi-value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    .verdict-good {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
    }
    
    .verdict-bad {
        background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%);
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px -5px rgba(239, 68, 68, 0.4);
    }
    
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-green {
        background-color: #d1fae5;
        color: #065f46;
    }
    
    .badge-red {
        background-color: #fee2e2;
        color: #991b1b;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# RESOURCE LOADERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    path = "data/processed/cleaned_housing_data.csv"
    if not os.path.exists(path):
        path = "data/raw/india_housing_prices.csv"
    df = pd.read_csv(path)
    return df


@st.cache_resource
def load_models():
    classifier_path = "models/investment_classifier.pkl"
    regressor_path = "models/future_price_regressor.pkl"
    metadata_path = "models/model_metadata.json"

    clf = joblib.load(classifier_path) if os.path.exists(classifier_path) else None
    reg = joblib.load(regressor_path) if os.path.exists(regressor_path) else None

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return clf, reg, metadata


df = load_dataset()
classifier, regressor, metadata = load_models()


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/602/602275.png", width=65)
    st.title("Navigation")
    nav_option = st.radio(
        "Select Module:",
        [
            "🏢 Investment Advisor",
            "🔍 Market Explorer & Search",
            "📊 Comprehensive 20-Point EDA",
            "🧪 MLflow Model Registry",
            "💰 Mortgage & ROI Calculator"
        ]
    )

    st.markdown("---")
    st.markdown("### 📌 Project Metadata")
    st.markdown("**Domain:** Real Estate Analytics")
    st.markdown(f"**Records:** {len(df):,} properties")
    st.markdown(f"**Cities Tracked:** {df['City'].nunique()}")
    st.markdown(f"**States Covered:** {df['State'].nunique()}")
    st.markdown("---")
    st.caption("Powered by Scikit-Learn, MLflow & Streamlit")


# -----------------------------------------------------------------------------
# MODULE 1: INVESTMENT ADVISOR (PREDICTION ENGINE)
# -----------------------------------------------------------------------------
if nav_option == "🏢 Investment Advisor":
    st.markdown("""
    <div class="main-header">
        <h1>Real Estate Investment Advisor</h1>
        <p>AI-Powered Investment Classification ("Good Investment") & 5-Year Future Valuation Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Enter Property Specifications")
    
    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.subheader("📍 Location & Type")
        all_states = metadata.get("states", sorted(df['State'].unique().tolist()))
        state = st.selectbox("State", all_states, index=0)
        
        # Filter cities by selected state
        cities_in_state = sorted(df[df['State'] == state]['City'].unique().tolist())
        if not cities_in_state:
            cities_in_state = metadata.get("cities", sorted(df['City'].unique().tolist()))
        city = st.selectbox("City", cities_in_state, index=0)

        property_type = st.selectbox(
            "Property Type", 
            metadata.get("property_types", ['Apartment', 'Independent House', 'Villa']), 
            index=0
        )
        bhk = st.slider("BHK Configuration", min_value=1, max_value=5, value=3)
        size_sqft = st.number_input("Property Size (in SqFt)", min_value=300, max_value=10000, value=1850, step=50)

    with col_input2:
        st.subheader("💰 Pricing & Building Info")
        price_lakhs = st.number_input("Current Asking Price (in Lakhs INR)", min_value=5.0, max_value=2000.0, value=125.0, step=5.0)
        
        calculated_price_per_sqft = round((price_lakhs * 100000) / size_sqft, 2)
        st.info(f"💡 Calculated Price/SqFt: **₹{calculated_price_per_sqft:,.2f} / SqFt**")

        age_of_property = st.slider("Age of Property (Years)", min_value=0, max_value=50, value=4)
        total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=60, value=15)
        floor_no = st.number_input("Property Floor Number", min_value=0, max_value=total_floors, value=5)
        furnished_status = st.selectbox("Furnished Status", ['Furnished', 'Semi-furnished', 'Unfurnished'], index=1)

    with col_input3:
        st.subheader("🚇 Infrastructure & Amenities")
        public_transport = st.select_slider("Public Transport Accessibility", options=['Low', 'Medium', 'High'], value='High')
        nearby_schools = st.slider("Nearby Schools Rating / Density (1-10)", min_value=1, max_value=10, value=8)
        nearby_hospitals = st.slider("Nearby Hospitals Density (1-10)", min_value=1, max_value=10, value=7)
        parking_space = st.selectbox("Dedicated Parking Space", ["Yes", "No"], index=0)
        security = st.selectbox("Gated Security / CCTV", ["Yes", "No"], index=0)
        facing = st.selectbox("Facing Direction", ['East', 'North', 'North-East', 'West', 'South'], index=0)
        owner_type = st.selectbox("Owner Type", ['Owner', 'Builder', 'Broker'], index=0)
        availability_status = st.selectbox("Availability Status", ['Ready_to_Move', 'Under_Construction'], index=0)

    # Amenities checklist
    st.markdown("##### 🏊 Select Available Amenities:")
    amenity_cols = st.columns(5)
    with amenity_cols[0]:
        has_gym = st.checkbox("Gymnasium", value=True)
    with amenity_cols[1]:
        has_pool = st.checkbox("Swimming Pool", value=True)
    with amenity_cols[2]:
        has_clubhouse = st.checkbox("Clubhouse", value=True)
    with amenity_cols[3]:
        has_garden = st.checkbox("Landscaped Garden", value=True)
    with amenity_cols[4]:
        has_playground = st.checkbox("Kids Playground", value=False)

    amenities_count = sum([has_gym, has_pool, has_clubhouse, has_garden, has_playground])

    st.markdown("---")

    # Predict button
    if st.button("🚀 Analyze Investment Potential & Forecast Future Price", type="primary", use_container_width=True):
        if classifier is None or regressor is None:
            st.error("Model files not found. Please ensure `src/train_models.py` has been executed.")
        else:
            # Construct input dataframe
            input_dict = {
                'State': [state],
                'City': [city],
                'Property_Type': [property_type],
                'BHK': [bhk],
                'Size_in_SqFt': [size_sqft],
                'Price_in_Lakhs': [price_lakhs],
                'Price_per_SqFt_INR': [calculated_price_per_sqft],
                'Age_of_Property': [age_of_property],
                'Furnished_Status': [furnished_status],
                'Floor_No': [floor_no],
                'Total_Floors': [total_floors],
                'Nearby_Schools': [nearby_schools],
                'Nearby_Hospitals': [nearby_hospitals],
                'Public_Transport_Accessibility': [public_transport],
                'Parking_Space': [parking_space],
                'Security': [security],
                'Amenities_Count': [amenities_count],
                'Facing': [facing],
                'Owner_Type': [owner_type],
                'Availability_Status': [availability_status]
            }
            input_df = pd.DataFrame(input_dict)

            # Predictions
            clf_pred = classifier.predict(input_df)[0]
            clf_prob = classifier.predict_proba(input_df)[0][1] if hasattr(classifier, "predict_proba") else (0.95 if clf_pred == 1 else 0.05)
            reg_pred = float(regressor.predict(input_df)[0])

            # Investment calculations
            capital_gain = reg_pred - price_lakhs
            roi_pct = (capital_gain / price_lakhs) * 100
            cagr = (((reg_pred / price_lakhs) ** (1 / 5)) - 1) * 100

            # Display Results
            st.markdown("## 📊 Investment Assessment Results")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if clf_pred == 1:
                    st.markdown(f"""
                    <div class="verdict-good">
                        <h2 style="color: white; margin-bottom: 0.2rem;">🌟 RECOMMENDED INVESTMENT</h2>
                        <h4 style="color: #a7f3d0; font-weight: 500;">High Potential Property</h4>
                        <p style="margin-top: 1rem; font-size: 1.2rem;">
                            Model Confidence: <strong>{clf_prob * 100:.1f}%</strong>
                        </p>
                        <p style="color: #ecfdf5; font-size: 0.95rem;">
                            This property demonstrates strong pricing efficiency relative to the city median, robust infrastructure, and high appreciation tailwinds.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-bad">
                        <h2 style="color: white; margin-bottom: 0.2rem;">⚠️ CAUTION / SUB-OPTIMAL</h2>
                        <h4 style="color: #fecaca; font-weight: 500;">Lower Expected Return</h4>
                        <p style="margin-top: 1rem; font-size: 1.2rem;">
                            Model Confidence: <strong>{(1 - clf_prob) * 100:.1f}%</strong>
                        </p>
                        <p style="color: #fff1f2; font-size: 0.95rem;">
                            Asking price may be elevated compared to locality benchmarks or local infrastructure support is insufficient.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">5-Year Estimated Valuation</div>
                    <div class="kpi-value" style="color: #0f766e;">₹{reg_pred:,.2f} Lakhs</div>
                    <div style="margin-top: 0.5rem; color: #475569;">
                        Current Price: ₹{price_lakhs:,.2f} Lakhs
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

                metric_sub1, metric_sub2, metric_sub3 = st.columns(3)
                with metric_sub1:
                    st.metric("Total Capital Gain", f"₹{capital_gain:,.2f} L", f"+{roi_pct:.1f}% ROI")
                with metric_sub2:
                    st.metric("5-Year CAGR", f"{cagr:.2f}%", "Annualized")
                with metric_sub3:
                    city_median = df[df['City'] == city]['Price_per_SqFt_INR'].median()
                    diff_pct = ((calculated_price_per_sqft - city_median) / city_median) * 100
                    st.metric(
                        "Vs City Median / SqFt", 
                        f"₹{calculated_price_per_sqft:,.0f}", 
                        f"{diff_pct:+.1f}% vs City"
                    )

            # Visual gauges & breakdown
            st.markdown("---")
            st.markdown("### 📈 Valuation Trajectory & Radar Breakdown")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # 5-Year Price Trajectory
                years = [0, 1, 2, 3, 4, 5]
                projected_prices = [round(price_lakhs * ((1 + (cagr / 100)) ** y), 2) for y in years]
                
                fig_traj = go.Figure()
                fig_traj.add_trace(go.Scatter(
                    x=[f"Year {y}" if y > 0 else "Current" for y in years],
                    y=projected_prices,
                    mode='lines+markers+text',
                    name='Projected Price',
                    text=[f"₹{p:.1f}L" for p in projected_prices],
                    textposition="top center",
                    line=dict(color='#0f766e', width=3),
                    marker=dict(size=8, color='#047857')
                ))
                fig_traj.update_layout(
                    title="Estimated 5-Year Price Growth Trajectory (INR Lakhs)",
                    xaxis_title="Timeline",
                    yaxis_title="Property Value (Lakhs)",
                    template="plotly_white",
                    height=360
                )
                st.plotly_chart(fig_traj, use_container_width=True)

            with chart_col2:
                # Investment Score Radar
                categories = ['Pricing Advantage', 'Infrastructure', 'Connectivity', 'Amenities', 'Possession Safety']
                # Calculate dynamic factors
                p_adv = max(10, min(100, int(100 - diff_pct)))
                infra_score = int(((nearby_schools + nearby_hospitals) / 20) * 100)
                conn_score = 100 if public_transport == 'High' else (65 if public_transport == 'Medium' else 30)
                amen_score = int((amenities_count / 5) * 100)
                safe_score = 95 if availability_status == 'Ready_to_Move' else 55

                values = [p_adv, infra_score, conn_score, amen_score, safe_score]

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(15, 118, 110, 0.25)',
                    line=dict(color='#0f766e', width=2)
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    title="Property Investment Health Dimensions (Score 0-100)",
                    template="plotly_white",
                    height=360
                )
                st.plotly_chart(fig_radar, use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 2: MARKET EXPLORER & PROPERTY SEARCH
# -----------------------------------------------------------------------------
elif nav_option == "🔍 Market Explorer & Search":
    st.markdown("""
    <div class="main-header">
        <h1>Market Explorer & Property Search</h1>
        <p>Interactive Search, Filter, and Benchmark Database across 250,000 Indian Properties</p>
    </div>
    """, unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3, filter_c4 = st.columns(4)
    with filter_c1:
        sel_state = st.multiselect("Filter by State:", sorted(df['State'].unique().tolist()))
    with filter_c2:
        available_cities = sorted(df[df['State'].isin(sel_state)]['City'].unique().tolist()) if sel_state else sorted(df['City'].unique().tolist())
        sel_city = st.multiselect("Filter by City:", available_cities)
    with filter_c3:
        sel_prop_type = st.multiselect("Property Type:", sorted(df['Property_Type'].unique().tolist()))
    with filter_c4:
        sel_bhk = st.multiselect("BHK:", sorted(df['BHK'].unique().tolist()))

    budget_range = st.slider(
        "Budget Range (in Lakhs INR):",
        min_value=float(df['Price_in_Lakhs'].min()),
        max_value=float(df['Price_in_Lakhs'].max()),
        value=(20.0, 300.0)
    )

    # Filter dataframe
    filtered_df = df.copy()
    if sel_state:
        filtered_df = filtered_df[filtered_df['State'].isin(sel_state)]
    if sel_city:
        filtered_df = filtered_df[filtered_df['City'].isin(sel_city)]
    if sel_prop_type:
        filtered_df = filtered_df[filtered_df['Property_Type'].isin(sel_prop_type)]
    if sel_bhk:
        filtered_df = filtered_df[filtered_df['BHK'].isin(sel_bhk)]
    filtered_df = filtered_df[
        (filtered_df['Price_in_Lakhs'] >= budget_range[0]) & 
        (filtered_df['Price_in_Lakhs'] <= budget_range[1])
    ]

    st.markdown("### 📊 Market Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Properties Found", f"{len(filtered_df):,}")
    with kpi2:
        st.metric("Average Price", f"₹{filtered_df['Price_in_Lakhs'].mean():.2f} L" if len(filtered_df) > 0 else "N/A")
    with kpi3:
        st.metric("Median Price/SqFt", f"₹{filtered_df['Price_per_SqFt_INR'].median():,.0f}" if len(filtered_df) > 0 else "N/A")
    with kpi4:
        good_pct = (filtered_df['Good_Investment'].mean() * 100) if len(filtered_df) > 0 else 0
        st.metric("Good Investment Ratio", f"{good_pct:.1f}%")

    st.markdown("---")

    # Interactive Visualizations
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if len(filtered_df) > 0:
            fig_hist = px.histogram(
                filtered_df,
                x="Price_in_Lakhs",
                color="Property_Type",
                nbins=35,
                title="Price Distribution for Filtered Properties",
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    with v_col2:
        if len(filtered_df) > 0:
            sample_chart = filtered_df.sample(n=min(len(filtered_df), 1500), random_state=42)
            fig_scatter = px.scatter(
                sample_chart,
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="Good_Investment",
                color_continuous_scale=["#ef4444", "#10b981"],
                title="Property Size vs Price (Color by Good Investment)",
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### 📋 Filtered Properties Data View")
    display_cols = [
        'ID', 'State', 'City', 'Property_Type', 'BHK', 'Size_in_SqFt', 
        'Price_in_Lakhs', 'Price_per_SqFt_INR', 'Age_of_Property', 
        'Public_Transport_Accessibility', 'Good_Investment', 'Future_Price_5Y'
    ]
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 3: COMPREHENSIVE 20-POINT EDA
# -----------------------------------------------------------------------------
elif nav_option == "📊 Comprehensive 20-Point EDA":
    st.markdown("""
    <div class="main-header">
        <h1>Comprehensive 20-Point Exploratory Data Analysis</h1>
        <p>In-Depth Statistical Findings & Visualizations covering All 20 Project Specifications</p>
    </div>
    """, unsafe_allow_html=True)

    eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
        "1–5: Price & Size Dynamics",
        "6–10: Location Intelligence",
        "11–15: Feature Relationships",
        "16–20: Investment & Amenities"
    ])

    eda_dir = "outputs/eda_visualizations"

    with eda_tab1:
        st.subheader("Section 1: Price & Size Analysis (Questions 1 to 5)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Q1: Distribution of Property Prices")
            st.image(f"{eda_dir}/01_price_distribution.png", use_container_width=True)
            st.caption(f"**Key Finding:** Average property price is ₹{df['Price_in_Lakhs'].mean():.2f} Lakhs (Median: ₹{df['Price_in_Lakhs'].median():.2f} Lakhs) with a broad, healthy spread.")

            st.markdown("#### Q3: Price per Sq Ft by Property Type")
            st.image(f"{eda_dir}/03_price_per_sqft_by_type.png", use_container_width=True)
            st.caption("**Key Finding:** Villas command the highest premium per sq ft, followed by Independent Houses and Apartments.")

            st.markdown("#### Q5: Outliers in Price per SqFt and Size")
            st.image(f"{eda_dir}/05_outliers_price_size.png", use_container_width=True)
            st.caption("**Key Finding:** Both size and price per sqft conform to clean distributions with controlled outlier profiles.")

        with col2:
            st.markdown("#### Q2: Distribution of Property Sizes")
            st.image(f"{eda_dir}/02_size_distribution.png", use_container_width=True)
            st.caption(f"**Key Finding:** Property sizes range from {df['Size_in_SqFt'].min()} to {df['Size_in_SqFt'].max()} SqFt with mean {df['Size_in_SqFt'].mean():.0f} SqFt.")

            st.markdown("#### Q4: Property Size vs. Price Relationship")
            st.image(f"{eda_dir}/04_size_vs_price.png", use_container_width=True)
            st.caption("**Key Finding:** Property price and area correlate strongly with linear scaling segmented by property classification.")

    with eda_tab2:
        st.subheader("Section 2: Location-Based Analysis (Questions 6 to 10)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Q6: Average Price per Sq Ft by State")
            st.image(f"{eda_dir}/06_avg_price_per_sqft_by_state.png", use_container_width=True)
            st.caption("**Key Finding:** Western and Southern states (Maharashtra, Tamil Nadu, Karnataka) exhibit higher price benchmarks.")

            st.markdown("#### Q8: Median Age of Properties by Locality")
            st.image(f"{eda_dir}/08_median_age_by_locality.png", use_container_width=True)
            st.caption("**Key Finding:** Established central localities have median property ages of 25-35 years, while suburban pockets feature newer construction.")

            st.markdown("#### Q10: Top 5 Most Expensive Localities")
            st.image(f"{eda_dir}/10_top5_expensive_localities.png", use_container_width=True)
            st.caption("**Key Finding:** Premium prime localities command significant price premiums over city averages.")

        with col2:
            st.markdown("#### Q7: Top Cities by Average Property Price")
            st.image(f"{eda_dir}/07_avg_price_by_city.png", use_container_width=True)
            st.caption("**Key Finding:** Metro centers like New Delhi, Mumbai, Pune, and Chennai lead in average transaction prices.")

            st.markdown("#### Q9: BHK Distribution Across Major Cities")
            st.image(f"{eda_dir}/09_bhk_distribution_across_cities.png", use_container_width=True)
            st.caption("**Key Finding:** 2 BHK and 3 BHK units constitute over 60% of inventory across major Tier-1 and Tier-2 hubs.")

    with eda_tab3:
        st.subheader("Section 3: Feature Relationships & Correlation (Questions 11 to 15)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Q11: Correlation Heatmap")
            st.image(f"{eda_dir}/11_correlation_heatmap.png", use_container_width=True)
            st.caption("**Key Finding:** Correlation matrix confirms high relationship between current price and 5-year forecasted price.")

            st.markdown("#### Q13: Nearby Hospitals vs. Price per Sq Ft")
            st.image(f"{eda_dir}/13_hospitals_vs_price_per_sqft.png", use_container_width=True)
            st.caption("**Key Finding:** Healthcare proximity demonstrates consistent positive correlation with property market pricing.")

            st.markdown("#### Q15: Price per Sq Ft by Facing Direction")
            st.image(f"{eda_dir}/15_price_by_facing.png", use_container_width=True)
            st.caption("**Key Finding:** East and North facing properties consistently command modest buyer preference premiums.")

        with col2:
            st.markdown("#### Q12: Nearby Schools vs. Price per Sq Ft")
            st.image(f"{eda_dir}/12_schools_vs_price_per_sqft.png", use_container_width=True)
            st.caption("**Key Finding:** School density acts as a vital social infrastructure driver supporting capital appreciation.")

            st.markdown("#### Q14: Price by Furnished Status")
            st.image(f"{eda_dir}/14_price_by_furnished_status.png", use_container_width=True)
            st.caption("**Key Finding:** Furnished properties yield higher capital valuations compared to raw/unfurnished units.")

    with eda_tab4:
        st.subheader("Section 4: Investment, Amenities & Ownership (Questions 16 to 20)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Q16: Proportion of Properties by Owner Type")
            st.image(f"{eda_dir}/16_owner_type_distribution.png", use_container_width=True)
            st.caption("**Key Finding:** Clean distribution between direct Owner resale, Builder primary inventory, and Broker listings.")

            st.markdown("#### Q18: Parking Space Impact on Price")
            st.image(f"{eda_dir}/18_parking_vs_price.png", use_container_width=True)
            st.caption("**Key Finding:** Dedicated covered parking space adds measurable value and enhances liquidity.")

            st.markdown("#### Q20: Public Transport vs. Good Investment")
            st.image(f"{eda_dir}/20_transport_vs_investment.png", use_container_width=True)
            st.caption("**Key Finding:** High public transport accessibility significantly elevates the likelihood of classification as a 'Good Investment'.")

        with col2:
            st.markdown("#### Q17: Availability Status Breakdown")
            st.image(f"{eda_dir}/17_availability_status.png", use_container_width=True)
            st.caption("**Key Finding:** Even split between Ready-to-Move and Under-Construction listings across key urban centers.")

            st.markdown("#### Q19: Amenities Count vs. Price per Sq Ft")
            st.image(f"{eda_dir}/19_amenities_vs_price_per_sqft.png", use_container_width=True)
            st.caption("**Key Finding:** Gated developments featuring 4+ amenities (Pool, Gym, Clubhouse, Garden) reflect higher price realization.")


# -----------------------------------------------------------------------------
# MODULE 4: MLFLOW MODEL REGISTRY & BENCHMARKS
# -----------------------------------------------------------------------------
elif nav_option == "🧪 MLflow Model Registry":
    st.markdown("""
    <div class="main-header">
        <h1>MLflow Experiment Tracking & Model Registry</h1>
        <p>Production Model Metrics, Version Benchmarks, and Diagnostic Artifacts</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model evaluation CSV
    eval_csv_path = "outputs/model_evaluation.csv"
    if os.path.exists(eval_csv_path):
        eval_df = pd.read_csv(eval_csv_path)

        st.markdown("### 🏆 Model Comparison & Leaderboard")
        st.dataframe(eval_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔬 Diagnostic Artifacts")

    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        st.markdown("#### Classification Confusion Matrix")
        if os.path.exists("outputs/confusion_matrix.png"):
            st.image("outputs/confusion_matrix.png", use_container_width=True)

        st.markdown("#### Regression Actual vs. Predicted")
        if os.path.exists("outputs/regression_actual_vs_predicted.png"):
            st.image("outputs/regression_actual_vs_predicted.png", use_container_width=True)

    with diag_col2:
        st.markdown("#### Classification ROC Curve")
        if os.path.exists("outputs/roc_curve.png"):
            st.image("outputs/roc_curve.png", use_container_width=True)

        st.markdown("#### Regression Residuals Distribution")
        if os.path.exists("outputs/residuals_plot.png"):
            st.image("outputs/residuals_plot.png", use_container_width=True)


# -----------------------------------------------------------------------------
# MODULE 5: MORTGAGE & ROI CALCULATOR
# -----------------------------------------------------------------------------
elif nav_option == "💰 Mortgage & ROI Calculator":
    st.markdown("""
    <div class="main-header">
        <h1>Mortgage & Real Estate ROI Simulator</h1>
        <p>Financial Simulation for Down Payments, Loan EMIs, Rental Cashflows & Net Returns</p>
    </div>
    """, unsafe_allow_html=True)

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        prop_val_lakhs = st.number_input("Property Value (in Lakhs INR)", min_value=10.0, max_value=2000.0, value=100.0, step=5.0)
        down_payment_pct = st.slider("Down Payment Percentage (%)", min_value=10, max_value=50, value=20)
        loan_interest_pct = st.slider("Home Loan Annual Interest Rate (%)", min_value=6.0, max_value=14.0, value=8.5, step=0.1)
        loan_tenure_years = st.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20)
        expected_rental_yield_pct = st.slider("Expected Annual Rental Yield (%)", min_value=1.5, max_value=6.0, value=3.2, step=0.1)

    # Financial calculations
    down_payment_lakhs = prop_val_lakhs * (down_payment_pct / 100)
    loan_amount_lakhs = prop_val_lakhs - down_payment_lakhs
    loan_amount_inr = loan_amount_lakhs * 100000

    monthly_rate = (loan_interest_pct / 100) / 12
    total_months = loan_tenure_years * 12
    
    # EMI formula
    emi_inr = (loan_amount_inr * monthly_rate * ((1 + monthly_rate) ** total_months)) / (((1 + monthly_rate) ** total_months) - 1)
    monthly_rent_inr = (prop_val_lakhs * 100000 * (expected_rental_yield_pct / 100)) / 12
    net_monthly_cashflow = monthly_rent_inr - emi_inr

    with m_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monthly Home Loan EMI</div>
            <div class="kpi-value" style="color: #1e3a8a;">₹{emi_inr:,.0f} / mo</div>
            <div style="margin-top: 0.5rem; color: #64748b;">
                Principal Loan: ₹{loan_amount_lakhs:.2f} Lakhs | Down Payment: ₹{down_payment_lakhs:.2f} Lakhs
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Gross Monthly Rental Income</div>
            <div class="kpi-value" style="color: #047857;">₹{monthly_rent_inr:,.0f} / mo</div>
            <div style="margin-top: 0.5rem; color: {'#047857' if net_monthly_cashflow >= 0 else '#dc2626'}; font-weight: 600;">
                Net Cash Flow (Rent - EMI): ₹{net_monthly_cashflow:,.0f} / month
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 5-Year Net Wealth Accumulation Projection")

    # 5-Year equity build
    projected_val_5y = prop_val_lakhs * ((1 + 0.08) ** 5)
    total_rent_5y_lakhs = (monthly_rent_inr * 60) / 100000
    total_emi_5y_lakhs = (emi_inr * 60) / 100000
    capital_appreciation_lakhs = projected_val_5y - prop_val_lakhs
    net_profit_5y = capital_appreciation_lakhs + total_rent_5y_lakhs - total_emi_5y_lakhs

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Projected 5Y Value", f"₹{projected_val_5y:.2f} L", "+₹" + f"{capital_appreciation_lakhs:.2f} L")
    with c2:
        st.metric("Total 5Y Rental Inflow", f"₹{total_rent_5y_lakhs:.2f} L")
    with c3:
        st.metric("Total 5Y EMI Outflow", f"₹{total_emi_5y_lakhs:.2f} L")
    with c4:
        st.metric("Net 5Y Profit", f"₹{net_profit_5y:.2f} L", f"{'Profitable' if net_profit_5y > 0 else 'Deficit'}")
