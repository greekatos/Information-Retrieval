import tkinter as tk
from tkinter import simpledialog
from elasticsearch import Elasticsearch

HOST_URLS = ["http://127.0.0.1:9200/"]
es_conn = Elasticsearch(HOST_URLS)

book_title=[]

root = tk.Tk()#ROOT is used for the blank window
root.withdraw()
user_input = simpledialog.askstring(title="Name", prompt="Type the book title:")
print("Book Title:", user_input)


elastic_query = {
    "query_string": {
        "query": user_input,
        "default_field": "book_title"
    }
}

result = es_conn.search(index="books", query=elastic_query)
for i in result['hits']['hits']:
    book_title.append(i['_source']['book_title'])
print(book_title)
