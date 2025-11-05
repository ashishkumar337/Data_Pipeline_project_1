import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle

def extract_data(file_path):
    print(" Extracting data...")
    return pd.read_csv(file_path)

def transform_data(df):
    print(" Transforming data...")
    
 
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
    

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, label_encoders, scaler

def load_data(df, output_path, label_encoders=None, scaler=None):
    print("Loading transformed data...")
    
  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f" Data saved to: {output_path}")
    
  
    if label_encoders:
        with open(os.path.join(os.path.dirname(output_path), 'label_encoders.pkl'), 'wb') as f:
            pickle.dump(label_encoders, f)
        print(" Label encoders saved")
    
    if scaler:
        with open(os.path.join(os.path.dirname(output_path), 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print(" Scaler saved")

def main():
    input_file = "/content/raw_data.csv"
    output_file = "output/processed_data.csv"
    
   
    df = extract_data(input_file)
    df_transformed, label_encoders, scaler = transform_data(df)
    load_data(df_transformed, output_file, label_encoders, scaler)
    
    print("\n🎉 ETL process completed successfully!")

if __name__ == "__main__":
    main()
