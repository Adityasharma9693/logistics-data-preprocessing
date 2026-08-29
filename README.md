# Logistics Data Preprocessing Pipeline

This repository contains a Python-based data preprocessing pipeline designed for supply chain and logistics datasets. The goal of this project is to prepare raw, messy logistics data for predictive modeling (e.g., forecasting delivery delays or freight costs) by handling missing values, managing outliers, and normalizing features.

## Dataset Characteristics
The pipeline is designed to work with logistics data containing the following features:
*   **OrderID**: Unique identifier for shipments
*   **OrderDate / DeliveryDate**: Used to engineer lead times
*   **FreightCost / ProductWeight**: Numerical metrics scaled and checked for outliers
*   **ShippingMode**: Categorical transit method
*   **DeliveryStatus**: Target variable

## Methodology
1.  **Handling Missing Values**: Imputes missing categorical values (e.g., `ShippingMode`) using the mode, and numerical values (e.g., `FreightCost`) using the median. Drops records with missing target dates.
2.  **Feature Engineering**: Calculates `LeadTime_Days` by measuring the difference between the order date and delivery date.
3.  **Outlier Treatment**: Uses the Interquartile Range (IQR) method to cap extreme values in continuous variables like freight cost and product weight to prevent model distortion.
4.  **Normalization**: Applies Min-Max Scaling to numerical features to transform them into a `[0, 1]` range.

## Getting Started

### Prerequisites
Install the required Python packages:
```bash
pip install pandas numpy scikit-learn
