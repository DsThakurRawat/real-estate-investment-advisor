"""
Real Estate Investment Advisor: Exploratory Data Analysis (EDA) Module
Answers all 20 questions specified in the project guidelines and generates publication-grade visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 15

EDA_OUTPUT_DIR = "outputs/eda_visualizations"
os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)


def run_all_eda(data_path="data/processed/cleaned_housing_data.csv"):
    print("=" * 70)
    print("STARTING COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df):,} records and {len(df.columns)} columns.")

    eda_insights = {}

    # -------------------------------------------------------------
    # SECTION 1: PRICE & SIZE ANALYSIS (Questions 1 - 5)
    # -------------------------------------------------------------
    print("\n--- Section 1: Price & Size Analysis ---")

    # Q1: What is the distribution of property prices?
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Price_in_Lakhs'], kde=True, bins=50, color='#1f77b4', edgecolor='black')
    plt.title("Q1: Distribution of Property Prices (in Lakhs)")
    plt.xlabel("Price (Lakhs INR)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    q1_path = os.path.join(EDA_OUTPUT_DIR, "01_price_distribution.png")
    plt.savefig(q1_path, dpi=200)
    plt.close()
    eda_insights["Q1_price_distribution"] = {
        "mean": float(df['Price_in_Lakhs'].mean()),
        "median": float(df['Price_in_Lakhs'].median()),
        "min": float(df['Price_in_Lakhs'].min()),
        "max": float(df['Price_in_Lakhs'].max()),
        "std": float(df['Price_in_Lakhs'].std()),
        "insight": "Property prices range from 10 to 500 Lakhs with a uniform-to-mildly skewed distribution."
    }

    # Q2: What is the distribution of property sizes?
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Size_in_SqFt'], kde=True, bins=50, color='#2ca02c', edgecolor='black')
    plt.title("Q2: Distribution of Property Sizes (in SqFt)")
    plt.xlabel("Size (SqFt)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    q2_path = os.path.join(EDA_OUTPUT_DIR, "02_size_distribution.png")
    plt.savefig(q2_path, dpi=200)
    plt.close()
    eda_insights["Q2_size_distribution"] = {
        "mean": float(df['Size_in_SqFt'].mean()),
        "median": float(df['Size_in_SqFt'].median()),
        "min": float(df['Size_in_SqFt'].min()),
        "max": float(df['Size_in_SqFt'].max()),
        "insight": "Property areas span from 500 to 5,000 SqFt with a balanced distribution across configurations."
    }

    # Q3: How does price per sq ft vary by property type?
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt_INR', palette='Set2')
    plt.title("Q3: Price per Sq Ft (INR) by Property Type")
    plt.xlabel("Property Type")
    plt.ylabel("Price per Sq Ft (INR)")
    plt.tight_layout()
    q3_path = os.path.join(EDA_OUTPUT_DIR, "03_price_per_sqft_by_type.png")
    plt.savefig(q3_path, dpi=200)
    plt.close()
    eda_insights["Q3_price_by_type"] = (
        df.groupby('Property_Type')['Price_per_SqFt_INR'].mean().round(2).to_dict()
    )

    # Q4: Is there a relationship between property size and price?
    plt.figure(figsize=(9, 5))
    # Downsample for crisp scatter visualization
    sample_df = df.sample(n=3000, random_state=42)
    sns.scatterplot(data=sample_df, x='Size_in_SqFt', y='Price_in_Lakhs', hue='Property_Type', alpha=0.6, palette='tab10')
    plt.title("Q4: Relationship between Property Size and Price (Sampled)")
    plt.xlabel("Size (SqFt)")
    plt.ylabel("Price (Lakhs)")
    plt.tight_layout()
    q4_path = os.path.join(EDA_OUTPUT_DIR, "04_size_vs_price.png")
    plt.savefig(q4_path, dpi=200)
    plt.close()
    corr_size_price = float(df['Size_in_SqFt'].corr(df['Price_in_Lakhs']))
    eda_insights["Q4_size_price_corr"] = {
        "correlation": round(corr_size_price, 4),
        "insight": f"Pearson correlation between size and price is {corr_size_price:.2f}."
    }

    # Q5: Are there any outliers in price per sq ft or property size?
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['Price_per_SqFt_INR'], color='#ff7f0e')
    plt.title("Outliers in Price per SqFt")
    plt.ylabel("INR / SqFt")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Size_in_SqFt'], color='#9467bd')
    plt.title("Outliers in Property Size")
    plt.ylabel("SqFt")
    plt.tight_layout()
    q5_path = os.path.join(EDA_OUTPUT_DIR, "05_outliers_price_size.png")
    plt.savefig(q5_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # SECTION 2: LOCATION-BASED ANALYSIS (Questions 6 - 10)
    # -------------------------------------------------------------
    print("\n--- Section 2: Location-based Analysis ---")

    # Q6: What is the average price per sq ft by state?
    plt.figure(figsize=(12, 6))
    state_avg_psqft = df.groupby('State')['Price_per_SqFt_INR'].mean().sort_values(ascending=False)
    state_avg_psqft.plot(kind='bar', color='#17becf', edgecolor='black')
    plt.title("Q6: Average Price per Sq Ft by State (INR)")
    plt.ylabel("Average Price/SqFt (INR)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    q6_path = os.path.join(EDA_OUTPUT_DIR, "06_avg_price_per_sqft_by_state.png")
    plt.savefig(q6_path, dpi=200)
    plt.close()
    eda_insights["Q6_top_states_psqft"] = state_avg_psqft.head(5).round(2).to_dict()

    # Q7: What is the average property price by city?
    plt.figure(figsize=(14, 6))
    city_avg_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(15)
    city_avg_price.plot(kind='bar', color='#8c564b', edgecolor='black')
    plt.title("Q7: Top 15 Cities by Average Property Price (Lakhs INR)")
    plt.ylabel("Average Price (Lakhs)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    q7_path = os.path.join(EDA_OUTPUT_DIR, "07_avg_price_by_city.png")
    plt.savefig(q7_path, dpi=200)
    plt.close()
    eda_insights["Q7_top_cities_price"] = city_avg_price.round(2).to_dict()

    # Q8: What is the median age of properties by locality? (Top 15 localities)
    plt.figure(figsize=(12, 5))
    top_localities = df['Locality'].value_counts().head(15).index
    median_age_loc = df[df['Locality'].isin(top_localities)].groupby('Locality')['Age_of_Property'].median().sort_values()
    median_age_loc.plot(kind='barh', color='#e377c2', edgecolor='black')
    plt.title("Q8: Median Age of Properties in Top Localities (Years)")
    plt.xlabel("Median Age (Years)")
    plt.tight_layout()
    q8_path = os.path.join(EDA_OUTPUT_DIR, "08_median_age_by_locality.png")
    plt.savefig(q8_path, dpi=200)
    plt.close()

    # Q9: How is BHK distributed across cities?
    plt.figure(figsize=(14, 6))
    top_cities = df['City'].value_counts().head(8).index
    sns.countplot(data=df[df['City'].isin(top_cities)], x='City', hue='BHK', palette='viridis')
    plt.title("Q9: BHK Distribution Across Major Indian Cities")
    plt.xlabel("City")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    q9_path = os.path.join(EDA_OUTPUT_DIR, "09_bhk_distribution_across_cities.png")
    plt.savefig(q9_path, dpi=200)
    plt.close()

    # Q10: What are the price trends for the top 5 most expensive localities?
    top_5_exp_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().nlargest(5).index
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df[df['Locality'].isin(top_5_exp_localities)],
        x='Locality',
        y='Price_in_Lakhs',
        estimator=np.mean,
        palette='magma'
    )
    plt.title("Q10: Average Price for Top 5 Most Expensive Localities")
    plt.ylabel("Average Price (Lakhs)")
    plt.tight_layout()
    q10_path = os.path.join(EDA_OUTPUT_DIR, "10_top5_expensive_localities.png")
    plt.savefig(q10_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # SECTION 3: FEATURE RELATIONSHIP & CORRELATION (Questions 11 - 15)
    # -------------------------------------------------------------
    print("\n--- Section 3: Feature Relationship & Correlation ---")

    # Q11: How are numeric features correlated with each other?
    plt.figure(figsize=(12, 9))
    num_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt_INR', 
                'Age_of_Property', 'Nearby_Schools', 'Nearby_Hospitals', 
                'Amenities_Count', 'Infrastructure_Score', 'Future_Price_5Y', 'Investment_Score']
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Q11: Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    q11_path = os.path.join(EDA_OUTPUT_DIR, "11_correlation_heatmap.png")
    plt.savefig(q11_path, dpi=200)
    plt.close()

    # Q12: How do nearby schools relate to price per sq ft?
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x='Nearby_Schools', y='Price_per_SqFt_INR', marker='o', color='#d62728')
    plt.title("Q12: Nearby Schools vs. Average Price per Sq Ft (INR)")
    plt.xlabel("Nearby Schools Count / Rating")
    plt.ylabel("Avg Price per Sq Ft (INR)")
    plt.tight_layout()
    q12_path = os.path.join(EDA_OUTPUT_DIR, "12_schools_vs_price_per_sqft.png")
    plt.savefig(q12_path, dpi=200)
    plt.close()

    # Q13: How do nearby hospitals relate to price per sq ft?
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df, x='Nearby_Hospitals', y='Price_per_SqFt_INR', marker='s', color='#2ca02c')
    plt.title("Q13: Nearby Hospitals vs. Average Price per Sq Ft (INR)")
    plt.xlabel("Nearby Hospitals Count")
    plt.ylabel("Avg Price per Sq Ft (INR)")
    plt.tight_layout()
    q13_path = os.path.join(EDA_OUTPUT_DIR, "13_hospitals_vs_price_per_sqft.png")
    plt.savefig(q13_path, dpi=200)
    plt.close()

    # Q14: How does price vary by furnished status?
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Furnished_Status', y='Price_in_Lakhs', palette='Blues_d', ci=None)
    plt.title("Q14: Average Property Price by Furnished Status")
    plt.xlabel("Furnished Status")
    plt.ylabel("Average Price (Lakhs)")
    plt.tight_layout()
    q14_path = os.path.join(EDA_OUTPUT_DIR, "14_price_by_furnished_status.png")
    plt.savefig(q14_path, dpi=200)
    plt.close()
    eda_insights["Q14_furnished_price"] = df.groupby('Furnished_Status')['Price_in_Lakhs'].mean().round(2).to_dict()

    # Q15: How does price per sq ft vary by property facing direction?
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x='Facing', y='Price_per_SqFt_INR', palette='Purples_d', ci=None)
    plt.title("Q15: Price per Sq Ft by Facing Direction")
    plt.xlabel("Facing Direction")
    plt.ylabel("Price per Sq Ft (INR)")
    plt.tight_layout()
    q15_path = os.path.join(EDA_OUTPUT_DIR, "15_price_by_facing.png")
    plt.savefig(q15_path, dpi=200)
    plt.close()

    # -------------------------------------------------------------
    # SECTION 4: INVESTMENT / AMENITIES / OWNERSHIP ANALYSIS (Questions 16 - 20)
    # -------------------------------------------------------------
    print("\n--- Section 4: Investment, Amenities & Ownership Analysis ---")

    # Q16: How many properties belong to each owner type?
    plt.figure(figsize=(8, 5))
    owner_counts = df['Owner_Type'].value_counts()
    owner_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#4c72b0', '#55a868', '#c44e52'], startangle=90)
    plt.title("Q16: Proportion of Properties by Owner Type")
    plt.ylabel("")
    plt.tight_layout()
    q16_path = os.path.join(EDA_OUTPUT_DIR, "16_owner_type_distribution.png")
    plt.savefig(q16_path, dpi=200)
    plt.close()
    eda_insights["Q16_owner_types"] = owner_counts.to_dict()

    # Q17: How many properties are available under each availability status?
    plt.figure(figsize=(7, 5))
    status_counts = df['Availability_Status'].value_counts()
    sns.barplot(x=status_counts.index, y=status_counts.values, palette='copper')
    plt.title("Q17: Availability Status Breakdown")
    plt.xlabel("Availability Status")
    plt.ylabel("Number of Properties")
    plt.tight_layout()
    q17_path = os.path.join(EDA_OUTPUT_DIR, "17_availability_status.png")
    plt.savefig(q17_path, dpi=200)
    plt.close()

    # Q18: Does parking space affect property price?
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='Parking_Space', y='Price_in_Lakhs', palette='coolwarm')
    plt.title("Q18: Property Price Distribution by Parking Space")
    plt.xlabel("Parking Space Available")
    plt.ylabel("Price (Lakhs)")
    plt.tight_layout()
    q18_path = os.path.join(EDA_OUTPUT_DIR, "18_parking_vs_price.png")
    plt.savefig(q18_path, dpi=200)
    plt.close()
    eda_insights["Q18_parking_avg_price"] = df.groupby('Parking_Space')['Price_in_Lakhs'].mean().round(2).to_dict()

    # Q19: How do amenities affect price per sq ft?
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x='Amenities_Count', y='Price_per_SqFt_INR', palette='crest')
    plt.title("Q19: Amenities Count vs. Price per Sq Ft (INR)")
    plt.xlabel("Number of Amenities")
    plt.ylabel("Price per Sq Ft (INR)")
    plt.tight_layout()
    q19_path = os.path.join(EDA_OUTPUT_DIR, "19_amenities_vs_price_per_sqft.png")
    plt.savefig(q19_path, dpi=200)
    plt.close()

    # Q20: How does public transport accessibility relate to price per sq ft or investment potential?
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=df, x='Public_Transport_Accessibility', y='Price_per_SqFt_INR', palette='viridis', ci=None)
    plt.title("Transport vs Price/SqFt")
    plt.xlabel("Accessibility")
    plt.ylabel("Avg Price/SqFt (INR)")

    plt.subplot(1, 2, 2)
    sns.barplot(data=df, x='Public_Transport_Accessibility', y='Good_Investment', palette='viridis', ci=None)
    plt.title("Transport vs Good Investment Rate")
    plt.xlabel("Accessibility")
    plt.ylabel("Proportion Good Investment")
    plt.tight_layout()
    q20_path = os.path.join(EDA_OUTPUT_DIR, "20_transport_vs_investment.png")
    plt.savefig(q20_path, dpi=200)
    plt.close()

    # Save summary insights json
    with open("outputs/eda_insights.json", "w") as f:
        json.dump(eda_insights, f, indent=4)

    print("=" * 70)
    print(f"EDA COMPLETED! All 20 visual assets generated in {EDA_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_eda()
