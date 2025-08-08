import pandas as pd

# Assume df is your original DataFrame
# Example index: "resnet50_run_1_perfs", "resnet50_run_2_perfs"

# Step 1: Extract model_id from index
df = df.copy()
df['model_id'] = df.index.str.extract(r'^([\w\-.]+)_run_\d+_perfs$')[0]

# Step 2: Group by model_id and compute mean and std
means = df.groupby('model_id').mean(numeric_only=True)
stds = df.groupby('model_id').std(numeric_only=True)

# Optional: Rename columns to reflect mean/std
means.columns = [f"{col}_mean" for col in means.columns]
stds.columns = [f"{col}_std" for col in stds.columns]

# Step 3: Combine into a single DataFrame
summary = pd.concat([means, stds], axis=1)

# Done!
print(summary)
