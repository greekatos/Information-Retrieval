import sys
import numpy as np
import pandas as pd
from keras import models
from keras.layers import Embedding, LSTM, Dense, Dropout
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

np.set_printoptions(threshold=sys.maxsize)



# Remove the warning about the method we create the column cleared summaries into the df_filtered
pd.options.mode.chained_assignment = None

sw = set(stopwords.words('english'))# We want to remove stopwords because by removing them we remove the low-level information from our text in order to give more focus to the important information
#In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task.
ps = PorterStemmer()#https://stackoverflow.com/a/14525214/14749665(We might not include it to our final code)

batch_size = 64

def html_decode(s):

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


def model_construction(data_train):
    model = models.Sequential()
    model.add(Dense(256, input_shape=[data_train.shape[1]], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


# Import the dataframes
df_books = pd.read_csv(r"BX-Books.csv")
df_reviews = pd.read_csv(r"BX-Book-Ratings.csv")

# Merge the dataframes
df = pd.merge(df_books, df_reviews, on='isbn')


# df_filtered_0 Keep the rows which have rating something else upon 0 and are for the user with id=98391
df_filtered = df[df['rating'] != 0]
df_filtered_0 = df_filtered[df_filtered['uid'] == 98391]


# df_filtered_1 Keep the rows which have rating are for the user with id=11676
df_filtered_1 = df[df['uid'] == 11676]


df_filtered_0['cleared_summaries'] = df_filtered_0['summary'].apply(clean_text)
df_filtered_1['cleared_summaries'] = df_filtered_1['summary'].apply(clean_text)


corpus_0 = df_filtered_0['cleared_summaries'].values# the input
corpus_1 = df_filtered_1['cleared_summaries'].values


df_filtered_0['rating'] = [1 if x > 7 else 0 for x in df_filtered_0.rating] #we want to balance the data
df_filtered_1['rating'] = [1 if x > 5 else 0 for x in df_filtered_1.rating]


Y_0 = df_filtered_0['rating'].values # the rating we make it np array
Y_1 = df_filtered_1['rating'].values


#Transforms text into a sparse matrix of n-gram counts.
cv = CountVectorizer(max_df=0.6) #Convert a collection of text documents to a matrix of token counts.
tfidf = TfidfTransformer()  #Transform a count matrix to a normalized tf or tf-idf representation.


X_0 = cv.fit_transform(corpus_0)
X_0 = tfidf.fit_transform(X_0)


X_1 = cv.fit_transform(corpus_1)

X_1 = tfidf.fit_transform(X_1)

X_1 = csr_matrix.sorted_indices(X_1)
X_0 = csr_matrix.sorted_indices(X_0)

X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X_0, Y_0, test_size=0.3, random_state=1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, Y_1, test_size=0.3, random_state=1)


model1 = model_construction(X_train_1)


model1.fit(X_train_1, y_train_1, validation_data=(X_test_1, y_test_1), batch_size=batch_size, epochs=10)


model1.save('MyModel_tf', save_format='tf')
