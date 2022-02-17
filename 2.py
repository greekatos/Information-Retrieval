from tkinter import *
import tkinter.messagebox as MessageBox
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import math


def submit_value():
    # Initialize the variables
    data_set = {"book_title": [], "score": [], "user_rating": [], "average_rating": [], "isbn": [], "new_metric": []}#dictionary
    # Get the values that the user's have entered at the form
    book_title_info = book_title_box.get()
    user_info = user_box.get()

    if book_title_info == "" or user_info == "":
        MessageBox.showinfo("Error 404", "All Fields are required")
    else:
        print(book_title_info, user_info)
        elastic_query_user = {
            "query_string": {
                "query": user_info,
                "default_field": "uid"
            }
        }

        user_result = es_conn.search(index="ratings", query=elastic_query_user)# The value of the index parameter has to be normally users(index) but we put it ratings for the continue of the project

        if len(user_result['hits']['hits']) == 0:
            MessageBox.showinfo("Error 404", "User ID not found")
        else:
            elastic_query_book_title = {
                "query_string": {
                    "query": book_title_info,
                    "default_field": "book_title"
                }
            }# query for book_title with the user's answer

            result = es_conn.search(index="books", query=elastic_query_book_title, size=10)#search api at elasticsearch with the above query

            for i in range(len(result['hits']['hits'])):
                Ratings = []
                count = 0
                data_set['book_title'].append(result['hits']['hits'][i]['_source']['book_title'])#maybe it is not needed, it appends the book_titles of the hits
                data_set['score'].append(result['hits']['hits'][i]['_score'])#it appends the book_titles of the hits
                data_set['isbn'].append(result['hits']['hits'][i]['_source']['isbn'])#it appends the book_titles of the hits

                # print(data_set['isbn'][i])
                elastic_query_ratings = {
                    "query_string": {
                        "query": data_set['isbn'][i],
                        "default_field": "isbn"
                    }
                }
                isbn_result = es_conn.search(index="ratings", query=elastic_query_ratings,size=9999)#If the size is more than 9999 it will appear the first 9999 results. If we need more, we need to do sth else.

                for j in range(len(isbn_result['hits']['hits'])):
                    Ratings.append(isbn_result['hits']['hits'][j]['_source']['rating'])
                    if int(user_info) == isbn_result['hits']['hits'][j]['_source']['uid']:#This can be more fast in the terms of time if we put another boolean counter as a second parameter at the if statement
                        data_set['user_rating'].append(isbn_result['hits']['hits'][j]['_source']['rating'])
                        count += 1
                        # break We put it to reduce the cost of the time, because of note number 6

                if count == 0:
                    data_set['user_rating'].append(math.nan)

                Ratings = [x for x in Ratings if x != 0]# We remove the 0 from the list Ratings, because we consider 0 as a missing value

                if len(Ratings)==0:
                    data_set['average_rating'].append(math.nan)
                else:
                    data_set['average_rating'].append(np.mean(Ratings))



            for i in range(len(data_set['isbn'])):#Can we make it a function?
                Average = [data_set['score'][i], data_set['average_rating'][i], data_set['user_rating'][i]]
                if math.isnan(Average[-1]) or Average[-1] == 0:# Because we know that the user_rating can be nan we put -1 to Average, because we want to check only the last element of the list. If it is nan we pop it.
                    Average.pop()
                if math.isnan(Average[1]):
                    Average.pop()

                data_set['new_metric'].append(np.mean(Average))

            df = pd.DataFrame.from_dict(data_set)
            df = df.sort_values('new_metric', ascending=False)
            print(df)
           

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
