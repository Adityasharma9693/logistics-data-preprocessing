import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def cap_outliers_iqr(dataframe, column):
    """Caps outliers using the Interquartile Range (IQR) method."""
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    dataframe[column] = np.where(dataframe[column] < lower_bound, lower_bound, dataframe[column])
    dataframe[column] = np.where(dataframe[column] > upper_bound, upper_bound, dataframe[column])
    return dataframe

def main():
    print("Loading raw data...")
    # Step 1: Data Loading & Initial Inspection
    df = pd.read_csv('logistics_raw_data.csv')
    df = df.drop_duplicates(subset='OrderID')

    print("Handling missing values...")
    # Step 2: Handling Missing Values
    df = df.dropna(subset=['DeliveryDate', 'OrderDate'])
    
    shipping_mode = df['ShippingMode'].mode()[0]
    df['ShippingMode'] = df['ShippingMode'].fillna(shipping_mode)
    
    freight_median = df['FreightCost'].median()
    df['FreightCost'] = df['FreightCost'].fillna(freight_median)

    print("Engineering features...")
    # Step 3: Feature Engineering & Datetime Parsing
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%Y-%m-%d', errors='coerce')
    df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%Y-%m-%d', errors='coerce')
    
    df = df.dropna(subset=['OrderDate', 'DeliveryDate'])
    df['LeadTime_Days'] = (df['DeliveryDate'] - df['OrderDate']).dt.days
    df = df[df['LeadTime_Days'] >= 0]

    print("Capping outliers...")
    # Step 4: Outlier Treatment (IQR Method)
    for col in ['FreightCost', 'ProductWeight']:
        df = cap_outliers_iqr(df, col)

    print("Normalizing data...")
    # Step 5: Data Normalization
    scaler = MinMaxScaler()
    features_to_scale = ['FreightCost', 'ProductWeight', 'LeadTime_Days']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # Save the cleaned dataset
    output_filename = 'logistics_cleaned_data.csv'
    df.to_csv(output_filename, index=False)
    print(f"Preprocessing complete! Cleaned data saved to {output_filename}")

if __name__ == "__main__":
    main()
