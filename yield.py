import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# CSV dosyasını oku
df = pd.read_csv('yield_df.csv')
df.rename(columns={'Unnamed: 0': ''}, inplace=True)

# Güncellenmiş veri çerçevesini kaydet
df.to_csv('yield_df.csv', index=False)

# İşlem tamamlandıktan sonra kontrol
print(df.head())

# Veri tipleri hakkında bilgi
df.info()

print(df.isnull().sum())

print("Number of duplicates before dropping:", df.duplicated().sum())

# Kopyaları düşürme ve son kontrol
df.drop_duplicates(inplace=True)
print("Number of duplicates after dropping:", df.duplicated().sum())

# Veri çerçevesi boyutu
print("Dataframe shape:", df.shape)

print("Dataframe describe:", df.describe)

print("Corr:", df.corr)

# Data Visualization

print(len(df['Area'].unique()))

print(len(df['Item'].unique()))


plt.figure(figsize=(20, 10))  # Genişletilmiş boyut
sns.countplot(x=df['Area'], palette='muted')
plt.xticks(rotation=90, fontsize=6)  # X eksenindeki etiketleri döndür
plt.title('Count of Areas')
plt.xlabel('Area')
plt.ylabel('Count')
plt.show() # Y ekseni yazı tipi küçültüldü

plt.tight_layout()  # Etiketlerin taşmasını engeller
plt.show()


plt.figure(figsize=(15,20))
sns.countplot(y=df['Item'], palette='muted')
plt.show()

print((df['Area'].value_counts() <400).sum())
country = df['Area'].unique()
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['Area'] == state]['hg/ha_yield'].sum())
print(df['hg/ha_yield'].sum())
print(yield_per_country)

plt.figure(figsize=(20, 12))  # Genişletilmiş boyut
sns.barplot(x=country, y=yield_per_country, palette='muted')

plt.xticks(rotation=90, fontsize=10)

plt.title("Yield per Country", fontsize=16)
plt.xlabel("Country", fontsize=14)
plt.ylabel("Yield", fontsize=14)

plt.tight_layout()

plt.show()


crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item'] == crop]['hg/ha_yield'].sum())

plt.figure(figsize=(15,20))
sns.barplot(y = crops, x = yield_per_crop,palette='muted')
plt.show()

print(df.head())
print(df.columns)

col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp','Area', 'Item', 'hg/ha_yield']
df = df[col]
print(df.head())

X = df.drop('hg/ha_yield', axis = 1)
y = df['hg/ha_yield']

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, shuffle=True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ohe = OneHotEncoder(handle_unknown='ignore')
scale = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']),
        ('OneHotEncode', ohe, ['Area', 'Item'])    ], 
    remainder = 'passthrough'
)

X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy  = preprocessor.fit_transform(X_test)

print(preprocessor.get_feature_names_out(col[:-1]))

from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score



X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy  = preprocessor.transform(X_test)

print(f"X_train_dummy shape: {X_train_dummy.shape}")
print(f"X_test_dummy shape: {X_test_dummy.shape}")

models = {
    'Linear Regression': LinearRegression(),
    'Lasso' : Lasso(),
    'Ridge' : Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}

for name, md in models.items():
    md.fit(X_train_dummy,y_train)
    y_pred = md.predict(X_test_dummy)
    print(f"{name}: mae : {mean_absolute_error(y_test, y_pred)} score : {r2_score(y_test, y_pred)}")

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr.predict(X_test_dummy)

print(df.columns)
print(df.head())

dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
print(dtr.predict(X_test_dummy))

print(df.columns)
print(df.head())


# Predictive System
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = pd.DataFrame([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                            columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
    print(features)
    print(features.dtypes)
    transform_features = preprocessor.transform(features)
    predicted_yeild = dtr.predict(transform_features).reshape(-1,1)
    return predicted_yeild[0][0]
    print("Features Columns:", features.columns)
    print("Training Data Columns:", X_train.columns)
    print("Preprocessor Feature Names:", preprocessor.get_feature_names_out())

result = prediction(2015, 950.0, 38554.69, 33.0, 'Turkey', 'Sorghum')

print("Predicted Yield:", result)


import pickle
pickle.dump(dtr, open("dtr.pkl","wb"))
pickle.dump(preprocessor, open("preprocessor.pkl","wb"))
