import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# Dosyaları listeleyin
file_names = ['pesticides.csv', 'rainfall.csv', 'temp.csv', 'yield_df.csv', 'yield.csv']

# Dosyaları işleyin ve üzerine yazdırın
for file_name in file_names:
    # Veri setini oku
    df = pd.read_csv(file_name)

    # Yıl sütununa göre filtrele (Year veya year olmasına dikkat edin)
    year_column = 'Year' if 'Year' in df.columns else 'year'
    filtered_data = df[(df[year_column] >= 2010) & (df[year_column] <= 2015)]
    
    # Boş satırları ve sütunları temizle
    filtered_data = filtered_data.dropna(how='all')
    filtered_data = filtered_data.dropna(axis=1, how='all')

    # Eğer dosya 'yield_df.csv' ise sıralama işlemi yap
    if file_name == 'yield_df.csv':
        filtered_data.insert(0, '', range(0, len(filtered_data)))

    # Filtrelenmiş veriyi aynı dosyanın üzerine yaz
    filtered_data.to_csv(file_name, index=False)
    print(f"Processed data written back to {file_name}")
