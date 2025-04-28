import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_excel('your_file.xlsx')
df['session_date'] = pd.to_datetime(df['session_date'])
df['week'] = df['session_date'].dt.strftime('%Y-W%U')

# 1. Weekly PS percentage
weekly_ps = df.groupby('week')['chat_treatment_status'].apply(
    lambda x: (x == 'PS').mean() * 100
).reset_index(name='ps_percentage')

# 2. Weekly satisfaction analysis for PS
ps_df = df[df['chat_treatment_status'] == 'PS']
weekly_satisfaction = ps_df.groupby('week')['satisfaction_status'].value_counts(
    normalize=True
).unstack().fillna(0) * 100

# 3. Prepare intention data
positive_intent = ps_df[ps_df['satisfaction_status'] == 'positive']['intention'].value_counts()
negative_intent = ps_df[ps_df['satisfaction_status'] == 'negative']['intention'].value_counts()

# Create visualization grid
plt.figure(figsize=(14, 12))

# Weekly PS Percentage Plot
plt.subplot(3, 1, 1)
plt.plot(weekly_ps['week'], weekly_ps['ps_percentage'], 
         marker='o', color='blue', linestyle='--')
plt.title('Weekly Percentage of PS Treatments')
plt.xticks(rotation=45)
plt.ylabel('Percentage (%)')
plt.grid(True, alpha=0.3)

# Weekly Satisfaction Plot
plt.subplot(3, 1, 2)
weekly_satisfaction.plot(kind='bar', stacked=True, ax=plt.gca(), 
                        color=['#4CAF50', '#FF5722', '#9E9E9E'])
plt.title('Weekly Satisfaction Distribution for PS Treatments')
plt.xlabel('Week')
plt.ylabel('Percentage (%)')
plt.legend(title='Satisfaction', bbox_to_anchor=(1, 1))
plt.grid(axis='y', alpha=0.3)

# Intention Distribution Horizontal Bar Charts
plt.subplot(3, 1, 3)
max_count = max(positive_intent.max(), negative_intent.max())
plt.barh(positive_intent.index, positive_intent.values, 
         height=0.4, color='#4CAF50', label='Positive')
plt.barh(negative_intent.index, negative_intent.values + max_count*0.05, 
         height=0.4, color='#FF5722', label='Negative')
plt.title('Intention Distribution by Satisfaction (Horizontal Comparison)')
plt.xlabel('Count')
plt.xlim(0, max_count*1.1)
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout(pad=2.0)

plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
df = pd.read_excel('your_file.xlsx')  # Replace with your file path

# Convert date column to datetime
df['session_date'] = pd.to_datetime(df['session_date'])

# 1. Percentage of lines with chat_treatment_status = "PS"
total_rows = len(df)
ps_count = len(df[df['chat_treatment_status'] == 'PS'])
ps_percentage = (ps_count / total_rows) * 100
print(f"Percentage of PS treatments: {ps_percentage:.2f}%")

# 2. Satisfaction analysis for PS treatments
ps_df = df[df['chat_treatment_status'] == 'PS']

# Global satisfaction percentages
global_satisfaction = ps_df['satisfaction_status'].value_counts(normalize=True) * 100
print("\nGlobal satisfaction percentages:")
print(global_satisfaction)

# Weekly satisfaction percentages
ps_df['week'] = ps_df['session_date'].dt.strftime('%Y-W%U')  # Year-week format
weekly_satisfaction = ps_df.groupby('week')['satisfaction_status'].value_counts(normalize=True).unstack() * 100
print("\nWeekly satisfaction percentages:")
print(weekly_satisfaction)

# 3. Intention histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Positive satisfaction histogram
positive_df = ps_df[ps_df['satisfaction_status'] == 'positive']
ax1.hist(positive_df['intention'], bins=20, color='green', alpha=0.7)
ax1.set_title('Positive Satisfaction Intention Distribution')
ax1.set_xlabel('Intention Score')
ax1.set_ylabel('Count')

# Negative satisfaction histogram
negative_df = ps_df[ps_df['satisfaction_status'] == 'negative']
ax2.hist(negative_df['intention'], bins=20, color='red', alpha=0.7)
ax2.set_title('Negative Satisfaction Intention Distribution')
ax2.set_xlabel('Intention Score')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()






import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
df = pd.read_excel('your_file.xlsx')  # Replace with your file path
df['session_date'] = pd.to_datetime(df['session_date'])

# 1. PS percentage
total_rows = len(df)
ps_count = len(df[df['chat_treatment_status'] == 'PS'])
ps_percentage = (ps_count / total_rows) * 100
print(f"Percentage of PS treatments: {ps_percentage:.2f}%")

# 2. Satisfaction analysis for PS
ps_df = df[df['chat_treatment_status'] == 'PS'].copy()

# Global satisfaction
global_satisfaction = ps_df['satisfaction_status'].value_counts(normalize=True) * 100
print("\nGlobal satisfaction percentages:")
print(global_satisfaction)

# Time-based analysis
ps_df['week'] = ps_df['session_date'].dt.strftime('%Y-W%U')
ps_df['month'] = ps_df['session_date'].dt.to_period('M')
ps_df['day'] = ps_df['session_date'].dt.date

# Weekly satisfaction
weekly_satisfaction = ps_df.groupby('week')['satisfaction_status'].value_counts(normalize=True).unstack() * 100

# Monthly satisfaction
monthly_satisfaction = ps_df.groupby('month')['satisfaction_status'].value_counts(normalize=True).unstack() * 100

# Daily satisfaction
daily_satisfaction = ps_df.groupby('day')['satisfaction_status'].value_counts(normalize=True).unstack() * 100

# Plotting satisfaction trends
fig, axes = plt.subplots(3, 1, figsize=(12, 15))

# Daily satisfaction plot
daily_satisfaction.plot(kind='line', ax=axes[0], marker='o', markersize=3)
axes[0].set_title('Daily Satisfaction Trends')
axes[0].xaxis.set_major_locator(mdates.AutoDateLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Weekly satisfaction plot
weekly_satisfaction.plot(kind='line', ax=axes[1], marker='o', markersize=3)
axes[1].set_title('Weekly Satisfaction Trends')
axes[1].set_xticks(range(len(weekly_satisfaction)))
axes[1].set_xticklabels(weekly_satisfaction.index, rotation=45)

# Monthly satisfaction plot
monthly_satisfaction.plot(kind='line', ax=axes[2], marker='o')
axes[2].set_title('Monthly Satisfaction Trends')
axes[2].set_xticks(range(len(monthly_satisfaction)))
axes[2].set_xticklabels([str(period) for period in monthly_satisfaction.index], rotation=45)

plt.tight_layout()

# 3. Intention histograms
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

positive_df = ps_df[ps_df['satisfaction_status'] == 'positive']
negative_df = ps_df[ps_df['satisfaction_status'] == 'negative']

ax1.hist(positive_df['intention'], bins=20, color='green', alpha=0.7)
ax1.set_title('Positive Satisfaction Intention')
ax1.set_xlabel('Intention Score')
ax1.set_ylabel('Count')

ax2.hist(negative_df['intention'], bins=20, color='red', alpha=0.7)
ax2.set_title('Negative Satisfaction Intention')
ax2.set_xlabel('Intention Score')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Print results
print("\nDaily satisfaction percentages:")
print(daily_satisfaction)
print("\nWeekly satisfaction percentages:")
print(weekly_satisfaction)
print("\nMonthly satisfaction percentages:")
print(monthly_satisfaction)






