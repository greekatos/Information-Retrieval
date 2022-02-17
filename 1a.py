from elasticsearch import Elasticsearch, helpers
import csv
import pandas as pd




HOST_URLS = ["http://127.0.0.1:9200"]
es_conn = Elasticsearch(HOST_URLS)


with open(r'BX-Book-Ratings.csv', encoding="utf8") as df:
    reader = csv.DictReader((l.replace('\0', '') for l in df))
    helpers.bulk(es_conn, reader, index='ratings')


with open(r'BX-Books.csv', encoding="utf8") as df:
    reader = csv.DictReader(df)
    helpers.bulk(es_conn, reader, index='books')