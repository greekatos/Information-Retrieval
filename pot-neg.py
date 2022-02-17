import pandas as pd
import matplotlib.pyplot as plt

df_reviews = pd.read_csv(r"BX-Book-Ratings.csv")
print(df_reviews['uid'].value_counts())
df_filtered_0 = df_reviews[df_reviews['uid'] == 98391]
df_filtered_1 = df_reviews[df_reviews['uid'] == 11676]

# df_filtered_0['rating'] = [1 if x > 7 else 0 for x in df_filtered_0.rating]
# df_filtered_1['rating'] = [1 if x > 5 else 0 for x in df_filtered_1.rating]

print(df_filtered_1['rating'].value_counts())
print(df_filtered_0['rating'].value_counts())

array = [str(x) for x in df_filtered_1['rating'].value_counts().index.tolist()]
array1 = [str(x) for x in df_filtered_0['rating'].value_counts().index.tolist()]


plt.bar(array,df_filtered_1['rating'].value_counts().values.tolist())
plt.xlabel('rating')
plt.ylabel('count')
plt.title('rating count 11676')
plt.show()

plt.bar(array1,df_filtered_0['rating'].value_counts().values.tolist())
plt.xlabel('rating')
plt.ylabel('count')
plt.title('rating count 98391')
plt.show()