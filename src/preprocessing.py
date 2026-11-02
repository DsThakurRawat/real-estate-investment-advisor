"""
Real Estate Investment Advisor: Preprocessing & Feature Engineering Module
"""

import os
import numpy as np
import pandas as pd

# Tier classifications for Indian real estate appreciation
TIER_1_CITIES = {
    'New Delhi', 'Mumbai', 'Pune', 'Chennai', 'Bangalore', 
    'Hyderabad', 'Kolkata', 'Ahmedabad', 'Gurgaon', 'Noida'
}

TIER_2_CITIES = {
    'Coimbatore', 'Jaipur', 'Ludhiana', 'Chandigarh', 'Kochi', 
    'Indore', 'Lucknow', 'Nagpur', 'Surat', 'Vadodara', 
    'Visakhapatnam', 'Bhopal', 'Patna', 'Bhubaneswar', 'Dehradun',
    'Mysore', 'Nashik', 'Ranchi', 'Guwahati', 'Raipur'
}


def calculate_growth_rate(row):
    """
    Computes expected annual appreciation rate based on city tier,
    property type, connectivity, amenities, and infrastructure.
    """
    city = row['City']
    if city in TIER_1_CITIES:
        base_rate = 0.085
    elif city in TIER_2_CITIES:
        base_rate = 0.075
    else:
        base_rate = 0.065

    # Property type modifier
    if row.get('Property_Type') in ['Villa', 'Independent House']:
        base_rate += 0.005

    # Public transport modifier
    transport = str(row.get('Public_Transport_Accessibility', 'Medium'))
    if transport == 'High':
        base_rate += 0.008
    elif transport == 'Medium':
        base_rate += 0.003

    # Amenities modifier
    amenities_count = row.get('Amenities_Count', 0)
    if amenities_count >= 4:
        base_rate += 0.006
    elif amenities_count >= 2:
        base_rate += 0.003

    # Infrastructure modifier (schools & hospitals)
    if row.get('Nearby_Schools', 0) >= 7 and row.get('Nearby_Hospitals', 0) >= 7:
        base_rate += 0.006

    # Age of property modifier
    age = row.get('Age_of_Property', 10)
    if age <= 5:
        base_rate += 0.005
    elif age > 20:
        base_rate -= 0.005

    return round(base_rate, 4)


def preprocess_data(raw_filepath="data/raw/india_housing_prices.csv",
                    output_filepath="data/processed/cleaned_housing_data.csv"):
    """
    Cleans raw data, engineers financial and real estate features, 
    and constructs regression & classification targets.
    """
    print(f"Loading raw dataset from {raw_filepath}...")
    df = pd.read_csv(raw_filepath)
    print(f"Initial shape: {df.shape}")

    # Step 1: Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate records.")

    # Step 2: Handle string stripping
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # Step 3: Feature Engineering - Price per SqFt in INR
    df['Price_per_SqFt_INR'] = (
        (df['Price_in_Lakhs'] * 100000) / df['Size_in_SqFt'].replace(0, np.nan)
    ).round(2)

    # Step 4: Ensure Age_of_Property is accurate
    if 'Year_Built' in df.columns:
        df['Age_of_Property'] = 2026 - pd.to_numeric(df['Year_Built'], errors='coerce')
        df['Age_of_Property'] = df['Age_of_Property'].clip(lower=0)

    # Step 5: Parse Amenities and compute Amenity features
    def parse_amenities(val):
        if pd.isna(val) or str(val).lower() == 'nan' or str(val).strip() == '':
            return 0
        return len([a.strip() for a in str(val).split(',') if a.strip()])

    df['Amenities_Count'] = df['Amenities'].apply(parse_amenities)
    df['Has_Gym'] = df['Amenities'].str.contains('Gym', case=False, na=False).astype(int)
    df['Has_Pool'] = df['Amenities'].str.contains('Pool', case=False, na=False).astype(int)
    df['Has_Clubhouse'] = df['Amenities'].str.contains('Clubhouse', case=False, na=False).astype(int)
    df['Has_Garden'] = df['Amenities'].str.contains('Garden', case=False, na=False).astype(int)
    df['Has_Playground'] = df['Amenities'].str.contains('Playground', case=False, na=False).astype(int)

    # Step 6: Infrastructure Score (0 to 10)
    transport_map = {'High': 3, 'Medium': 2, 'Low': 1}
    transport_score = df['Public_Transport_Accessibility'].map(transport_map).fillna(2)
    school_score = (df['Nearby_Schools'] / 10.0) * 3.5
    hospital_score = (df['Nearby_Hospitals'] / 10.0) * 3.5
    df['Infrastructure_Score'] = (transport_score + school_score + hospital_score).round(2)

    # Step 7: Target Variable 1 (Regression) - Future Price in 5 Years
    df['Annual_Growth_Rate'] = df.apply(calculate_growth_rate, axis=1)
    df['Future_Price_5Y'] = (
        df['Price_in_Lakhs'] * ((1 + df['Annual_Growth_Rate']) ** 5)
    ).round(2)
    df['Expected_5Y_ROI_Pct'] = (
        ((df['Future_Price_5Y'] - df['Price_in_Lakhs']) / df['Price_in_Lakhs']) * 100
    ).round(2)

    # Step 8: Target Variable 2 (Classification) - Good Investment Label
    city_median_psqft = df.groupby('City')['Price_per_SqFt_INR'].transform('median')
    
    score = np.zeros(len(df))
    # Pricing advantage vs city median: +25
    score += np.where(df['Price_per_SqFt_INR'] <= city_median_psqft, 25, 0)
    # Expected growth rate >= 8%: +25, else +10
    score += np.where(df['Annual_Growth_Rate'] >= 0.080, 25, 10)
    # Infrastructure: +15 if both schools & hospitals >= 5
    score += np.where((df['Nearby_Schools'] >= 5) & (df['Nearby_Hospitals'] >= 5), 15, 5)
    # Public transport: +10 if High, +6 if Medium
    score += np.where(
        df['Public_Transport_Accessibility'] == 'High', 10,
        np.where(df['Public_Transport_Accessibility'] == 'Medium', 6, 0)
    )
    # Possession risk: +10 for Ready_to_Move
    score += np.where(df['Availability_Status'] == 'Ready_to_Move', 10, 3)
    # Security & Parking: +10 if both available
    score += np.where((df['Security'] == 'Yes') & (df['Parking_Space'] == 'Yes'), 10, 4)
    # Size/BHK utility: +5 for BHK >= 3
    score += np.where(df['BHK'] >= 3, 5, 2)

    df['Investment_Score'] = score
    # Balance at median (~60 score)
    df['Good_Investment'] = (df['Investment_Score'] >= 60).astype(int)

    # Clean outlier/infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Cleaned dataset successfully saved to {output_filepath}")
    print(f"Final shape: {df.shape}")
    print("Good_Investment distribution:")
    print(df['Good_Investment'].value_counts(normalize=True))
    print(f"Future_Price_5Y mean: {df['Future_Price_5Y'].mean():.2f} Lakhs (Current Price mean: {df['Price_in_Lakhs'].mean():.2f} Lakhs)")

    return df


if __name__ == "__main__":
    preprocess_data()
