import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



# Remove the warning about the method we create the column cleared summaries into the df_filtered
pd.options.mode.chained_assignment = None

sw = set(stopwords.words('english'))# We want to remove stopwords because by removing them we remove the low-level information from our text in order to give more focus to the important information
#In order words, we can say that the removal of such words does not show any negative consequences on the model we train for our task.
ps = PorterStemmer()#https://stackoverflow.com/a/14525214/14749665(We might not include it to our final code)


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


# Import the dataframes
df_books = pd.read_csv(r"BX-Books.csv")
df_reviews = pd.read_csv(r"BX-Book-Ratings.csv")

# Merge the dataframes
df = pd.merge(df_books, df_reviews, on='isbn')

# Keep the rows which have rating something else upon 0
df_filtered = df[df['rating'] != 0]

df_filtered_0['cleared_summaries'] = df_filtered_0['summary'].apply(clean_text)
df_filtered_1['cleared_summaries'] = df_filtered_1['summary'].apply(clean_text)

array = [str(i) for i in df_filtered['uid'].value_counts().head(10).index.tolist()]
plt.bar(array, df_filtered['uid'].value_counts().head(10).values.tolist())

plt.xlabel('frequency')

plt.ylabel('Users')

plt.title('Ratings count')

plt.show()