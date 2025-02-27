import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from your specified path
df = pd.read_csv("dataset/Indian Liver Patient Dataset (ILPD).csv")

# Assign column names since the dataset doesn't include headers
column_names = ['Age', 'Gender', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio', 'Selector']
df.columns = column_names

# Convert Gender to numerical (Female: 0, Male: 1)
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Basic dataset information
print(f"Dataset shape: {df.shape}")
print(f"Number of patients with liver disease: {sum(df['Selector'] == 1)}")
print(f"Number of patients without liver disease: {sum(df['Selector'] == 2)}")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Set Seaborn style
sns.set_theme(style="whitegrid", palette="pastel")

# Distribution of target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Selector', data=df)
plt.title('Distribution of Liver Disease Cases', fontsize=14)
plt.xlabel('Liver Disease (1: Yes, 2: No)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Age distribution by disease status
plt.figure(figsize=(10, 6))
sns.histplot(df, x='Age', hue='Selector', bins=20, kde=True, element="step", stat="density", common_norm=False)
plt.title('Age Distribution by Disease Status', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.show()

# Disease distribution by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Selector', data=df)
plt.title('Disease Distribution by Gender', fontsize=14)
plt.xlabel('Gender (0: Female, 1: Male)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Features', fontsize=14)
plt.show()


# Create distribution plots for all numerical features
numerical_features = ['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio']
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.histplot(df, x=feature, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# Create box plots to identify outliers in numerical features
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_features])
plt.title('Box Plots of Numerical Features', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Create pair plots to visualize relationships between features
sns.pairplot(df, hue='Selector', diag_kind='kde', plot_kws={'alpha': 0.6},
             vars=numerical_features[:4])  # Using first 4 features for clarity
plt.suptitle('Pair Plot of Features by Disease Status', y=1.02, fontsize=16)
plt.show()

# Compare distributions of features between disease and non-disease groups
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

for i, feature in enumerate(numerical_features):
    sns.violinplot(x='Selector', y=feature, data=df, ax=axes[i])
    axes[i].set_title(f'{feature} by Disease Status')
    axes[i].set_xlabel('Liver Disease (1: Yes, 2: No)')

plt.tight_layout()
plt.show()

# Create joint plots for key relationships
plt.figure(figsize=(10, 8))
sns.jointplot(data=df, x='TB', y='DB', hue='Selector', kind='scatter')
plt.suptitle('Relationship between Total and Direct Bilirubin by Disease Status', y=1.02)
plt.show()

sns.jointplot(data=df, x='Sgpt', y='Sgot', hue='Selector', kind='scatter')
plt.suptitle('Relationship between SGPT and SGOT by Disease Status', y=1.02)
plt.show()

# Apply log transformation to skewed features
skewed_features = ['TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot']
df_log = df.copy()

for feature in skewed_features:
    df_log[f'{feature}_log'] = np.log1p(df_log[feature])

# Plot distributions before and after transformation for a sample feature
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(df['Sgpt'], kde=True, ax=axes[0])
axes[0].set_title('Original SGPT Distribution')
sns.histplot(df_log['Sgpt_log'], kde=True, ax=axes[1])
axes[1].set_title('Log-transformed SGPT Distribution')
plt.show()

# Analyze age groups
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, 100], labels=['0-20', '21-40', '41-60', '60+'])

plt.figure(figsize=(10, 6))
sns.countplot(x='Age_Group', hue='Selector', data=df)
plt.title('Disease Distribution by Age Group', fontsize=14)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()

# Create swarm plots for key features
plt.figure(figsize=(12, 8))
sns.swarmplot(x='Selector', y='TB', data=df)
plt.title('Total Bilirubin by Disease Status', fontsize=14)
plt.xlabel('Liver Disease (1: Yes, 2: No)', fontsize=12)
plt.ylabel('Total Bilirubin', fontsize=12)
plt.show()

# Create strip plots with statistical annotations
plt.figure(figsize=(12, 8))
ax = sns.stripplot(x='Selector', y='DB', data=df, jitter=True)
plt.title('Direct Bilirubin by Disease Status with Statistical Summary', fontsize=14)
plt.xlabel('Liver Disease (1: Yes, 2: No)', fontsize=12)
plt.ylabel('Direct Bilirubin', fontsize=12)

# Add statistical annotations
for i, selector in enumerate([1, 2]):
    subset = df[df['Selector'] == selector]['DB']
    plt.text(i, subset.max() + 0.5, f'Mean: {subset.mean():.2f}\nStd: {subset.std():.2f}',
             ha='center', va='bottom', fontweight='bold')
plt.show()

# Create facet grids to analyze relationships across multiple dimensions
g = sns.FacetGrid(df, col='Gender', row='Selector', height=4, aspect=1.5)
g.map_dataframe(sns.scatterplot, x='Age', y='TB')
g.add_legend()
plt.suptitle('Relationship between Age and Total Bilirubin by Gender and Disease Status', y=1.02)
plt.show()

# Create a cluster map to visualize feature correlations
plt.figure(figsize=(12, 10))
sns.clustermap(df[numerical_features].corr(), annot=True, cmap='coolwarm',
               linewidths=0.5, figsize=(12, 10))
plt.title('Clustered Correlation Matrix', fontsize=14, y=1.02)
plt.show()

# Create KDE plots for key features
plt.figure(figsize=(12, 8))
for selector in df['Selector'].unique():
    subset = df[df['Selector'] == selector]
    sns.kdeplot(data=subset, x='Sgpt', y='Sgot', fill=True, alpha=0.5,
                levels=5, label=f'Selector {selector}')
plt.title('2D Density Plot of SGPT vs SGOT by Disease Status', fontsize=14)
plt.legend()
plt.show()
