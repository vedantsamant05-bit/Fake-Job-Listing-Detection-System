from src.data_loader import load_raw_data, basic_clean
from src.preprocessing import build_preprocessor

df = basic_clean(load_raw_data())

preprocessor = build_preprocessor()

X = preprocessor.fit_transform(df.head(100))

print(type(X))
print(X.shape)