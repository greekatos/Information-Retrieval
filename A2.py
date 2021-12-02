from tkinter import *
from tkinter import simpledialog
import tkinter.messagebox as MessageBox
from elasticsearch import Elasticsearch
import json


def submit_value():
    data_set = {"book_title": [], "score": [], "user_rating": [], "average_rating": [], "isbn": []}#dictionary
    book_title_info = book_title_box.get()
    user_info = user_box.get()

    if(book_title_info=="" or user_info==""):
        MessageBox.showinfo("Error 404", "All Fields are required")
    else:
        print(book_title_info, user_info)
        elastic_query_user = {
            "query_string":{
                "query":user_info,
                "default_field":"uid"
            }
        }

        user_result = es_conn.search(index="ratings",query= elastic_query_user)
        print(user_result)
        if(len(user_result['hits']['hits'])==0):
            MessageBox.showinfo("Error 404", "User ID not found")
        else:
            # elastic_query = json.dumps({
            #     "query": {
            #         "match_phrase": {
            #             "book_title": book_title_info
            #         }
            #     }
            # }) old-fashioned way with body parameter instead of query in result parameter

            elastic_query = {
                "query_string": {
                    "query": book_title_info,
                    "default_field": "book_title"
                }
            }#query for book_title with the user's answer

            result = es_conn.search(index="books", query=elastic_query)#search api at elasticsearch with the above query

            for i in range(len(result['hits']['hits'])):
                data_set['book_title'].append(result['hits']['hits'][i]['_source']['book_title'])#maybe it is not needed
                data_set['score'].append(result['hits']['hits'][i]['_score'])
                data_set['isbn'].append(result['hits']['hits'][i]['_source']['isbn'])


            print(result['hits']['hits'])
            print(data_set['isbn'][0])
            MessageBox.showinfo("Execution Status","Query successfully executed")


HOST_URLS = ["http://127.0.0.1:9200"]
es_conn = Elasticsearch(HOST_URLS)


root = Tk()
root.geometry("500x500")

book_title_text = Label(root, text="Book Title:")
user_text = Label(root, text="User Id:")
book_title_text.place(x=15, y=70)
user_text.place(x=15, y=140)


book_title_box = Entry()#Entry it will create a box that the user can write'''textvariable=book_title_value, width = "30"'''
user_box = Entry()#'''textvariable=user_value, width = "30"'''
book_title_box.place(x=15,y=100)
user_box.place(x=15,y=200)

register = Button(root, text="Submit", width = "30",height="2", command=submit_value)
register.place(x=15,y=290)

root.mainloop()


