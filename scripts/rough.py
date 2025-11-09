from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target

import pandas as pd
df = pd.DataFrame(X, columns=wine.feature_names)
df['target'] = y

print(df.head())
print(df.shape)