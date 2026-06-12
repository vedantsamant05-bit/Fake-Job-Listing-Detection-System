import joblib

pipeline = joblib.load("models/pipeline.joblib")

print(type(pipeline))

pre = pipeline.named_steps["preprocessor"]

print(type(pre))

cat_pipeline = pre.named_transformers_["cat"]

print(cat_pipeline)

imputer = cat_pipeline.named_steps["imputer"]

print(type(imputer))

print("Has _fill_dtype:", hasattr(imputer, "_fill_dtype"))

print("Imputer dict:")
print(imputer.__dict__.keys())
