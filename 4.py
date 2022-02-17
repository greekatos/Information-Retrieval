import nltk
import time
import sys
import numpy as np
import pandas as pd
from spherical_kmeans import SphericalKMeans
from sklearn.metrics.pairwise import linear_kernel
# from scipy.cluster import  hierarchy
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import sklearn.cluster as cluster
# from sklearn.pipeline import Pipeline
# from sklearn.decomposition import PCA
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
# np.set_printoptions(threshold=sys.maxsize)



# Remove the warning about the method we create the column cleared summaries into the df_filtered
pd.options.mode.chained_assignment = None

sw = set(stopwords.words('english'))# We want to remove stopwords because by removing them we remove the low-level information from our text in order to give more focus to the important information
#In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task.


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (("'", '&#39;'), ('"', '&quot;'), ('>', '&gt;'), ('<', '&lt;'), ('&', '&amp;'))

    for code in htmlCodes:
        s = s.replace(code[1], code[0])
    return s


def clean_text(sample):
    sample = sample.lower()
    sample = html_decode(sample)
    sample = sample.split()
    sample = [s for s in sample if s not in sw]
    sample = ' '.join(sample)
    return sample

def current_milli_time():
    return round(time.time() * 1000)

# Import the dataframes
df_books = pd.read_csv(r"BX-Books.csv")
df_reviews = pd.read_csv(r"BX-Book-Ratings.csv")

# Merge the dataframes
df = pd.merge(df_books, df_reviews, on='isbn')

# df_filtered_1 Keep the rows which have rating are for the user with id=11676
df_filtered_1 = df[df['uid'] == 11676]

df_filtered_1['cleared_summaries'] = df_filtered_1['summary'].apply(clean_text)

corpus_1 = df_filtered_1['cleared_summaries'].values


#Transforms text into a sparse matrix of n-gram counts.
cv = CountVectorizer(max_df=0.6)
tfidf = TfidfTransformer()


X_1 = cv.fit_transform(corpus_1)
X_1 = tfidf.fit_transform(X_1)
X_1_dense = X_1.todense()
X_1_list = X_1_dense.tolist()
skm = SphericalKMeans(n_clusters=3)
skm.fit(X_1)
print(X_1)
print(skm)
# print('cosine similarities:', linear_kernel(X_1_dense[0], X_1_dense[0]).flatten())
# print(X_1)


# kmeans = KMeans(n_clusters=3, max_iter=600, algorithm='auto', random_state=200)
# fitted = kmeans.fit(X_1)
# sns.scatterplot(data=X_1.toarray())
# plt.show()
# prediction = kmeans.predict(X_1)


# X_1 = csr_matrix.sorted_indices(X_1)

