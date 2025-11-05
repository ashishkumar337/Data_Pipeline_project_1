import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def extract_data(file_path):
    print("📥 Extracting data...")
    return pd.read_csv(file_path)

def transform_data(df):
    print("⚙️ Transforming data...")

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    df = df.fillna("Unknown")

    # Encode categorical columns
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_enc.fit_transform(df[col])

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def load_data(df, output_path):
    print("💾 Loading transformed data...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to: {output_path}")

def main():
    input_file = "data/raw_data.csv"
    output_file = "output/processed_data.csv"

    # Pipeline Flow
    df = extract_data(input_file)
    df_transformed = transform_data(df)
    load_data(df_transformed, output_file)
    print("🎉 ETL process completed successfully!")

if __name__ == "__main__":
    main()
