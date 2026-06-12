from src.predict import predict_single

sample_listing = {
    "title": "Work From Home - Earn $5000 Weekly",
    "description": (
        "No experience needed. "
        "Work from anywhere. "
        "Unlimited earnings."
    ),
    "requirements": "",
    "benefits": "",
    "salary_range": "",
}

result = predict_single(sample_listing)

print(result)