from src.data_loader import load_raw_data, basic_clean
from src.feature_engineering import StructuredFeatureEngineer

df = basic_clean(load_raw_data())

engineer = StructuredFeatureEngineer()

features = engineer.transform(df.head())

print(features)
print(features.shape) 