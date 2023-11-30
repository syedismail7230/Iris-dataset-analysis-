import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

print(iris_df.info())

print(iris_df.describe())

sns.pairplot(iris_df, hue='target', palette='viridis')
plt.title('Pairplot of Iris Dataset')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_df.drop('target', axis=1), palette='Set2')
plt.title('Boxplot of Iris Features')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=iris_df.drop('target', axis=1), palette='Pastel1')
plt.title('Violin Plot of Iris Features')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=iris_df, palette='Set3')
plt.title('Distribution of Iris Target Classes')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
