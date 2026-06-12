from src.data_loader import load_raw_data, basic_clean

df = load_raw_data()

print("Shape:", df.shape)

df = basic_clean(df)

print("\nFraud Distribution:")
print(df["fraudulent"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))