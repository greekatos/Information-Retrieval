import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r'BX-Book-Ratings.csv', on_bad_lines='warn')

print(df)


plt.rc("font", size=10)
df['rating'].value_counts(sort=False, dropna=False).plot(kind='bar')#value_counts used to count the unique values of a dataframe
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('system1.png', bbox_inches='tight')
plt.show()



