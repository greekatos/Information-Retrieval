import pandas as pd
import matplotlib.pyplot as plt

df_reviews = pd.read_csv(r"BX-Book-Ratings.csv")
print(df_reviews['uid'].value_counts())
array = [str(x) for x in df_reviews['uid'].value_counts().head(10).index.tolist()]
array1 = [str(x) for x in df_reviews['rating'].value_counts().head(11).index.tolist()]


plt.bar(array,df_reviews['uid'].value_counts().head(10).values.tolist())
plt.xlabel('frequency')
plt.ylabel('users')
plt.title('rating count')
plt.show()

plt.bar(array1,df_reviews['rating'].value_counts().head(11).values.tolist())
plt.xlabel('ratings')
plt.ylabel('count')
plt.title('rating distribution')
plt.show()