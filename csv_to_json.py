import json

# Fill down the missing model names
df['Model'] = df['Model'].fillna(method='ffill')

# Group the data by Model
models_json = {}
for model, group in df.groupby('Model'):
    features = {}
    for _, row in group.iterrows():
        feature = row['Feature']
        condition = row['Condition / Tier']
        price = row['Price / 1M tokens']
        if feature not in features:
            features[feature] = []
        features[feature].append({
            'condition': condition,
            'price_per_1M_tokens': price
        })
    models_json[model] = features

# Save to JSON file
json_path = "/mnt/data/llm_pricing.json"
with open(json_path, 'w') as f:
    json.dump(models_json, f, indent=4)

json_path


























import json

# Load JSON data from the file
with open("llm_pricing.json", "r") as f:
    data = json.load(f)

def extract_model_prices(model_name):
    model_data = data.get(model_name)
    if not model_data:
        return f"Model '{model_name}' not found."

    result = {}
    for feature in ['Input', 'Output', 'Context Caching']:
        feature_entries = model_data.get(feature)
        if feature_entries:
            # Get the last entry only
            last_entry = feature_entries[-1]
            result[feature] = last_entry['price_per_1M_tokens']
        else:
            result[feature] = None  # or you could skip it

    return result

# Example usage
model_name = "Gemini 1.5 Pro"
prices = extract_model_prices(model_name)
print(prices)
