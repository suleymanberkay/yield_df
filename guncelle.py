import seaborn as sns
import matplotlib.pyplot as plt

# Korelasyon matrisi
corr_matrix = data.corr()
print("Korelasyon Matrisi:")
print(corr_matrix['hg/ha_yield'].sort_values(ascending=False))

# Isı haritası
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()
