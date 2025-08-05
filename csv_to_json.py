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

