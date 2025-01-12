import pandas as pd

# CSV dosyasını yükle
df = pd.read_csv('yield_df.csv')
df.rename(columns={'Unnamed: 0': ''}, inplace=True)

# Yeni bir sıralama yapmak için 'Area' ve 'Year' sütunlarına göre sıralama yapılır
df_sorted = df.sort_values(by=['Area', 'Year']).reset_index(drop=True)

# Yeni sıralanmış veri ile yeni CSV dosyasını kaydedin
df_sorted.to_csv('sorted_yield_df.csv', index=False)

# Sıralı verileri ekrana yazdır
print(df_sorted.head())
