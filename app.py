#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 07:24:57 2022

@author: venkateshchandra
"""
from flask import Flask, request, render_template, escape
from summary import summarize

app = Flask(__name__)

@app.route('/')
def home():
    #yes it works
    return render_template('submit.html')

@app.route('/summarizer',methods=['POST'])
def predict2():
    '''
    Render results on HTML GUI
    '''
    
    query = request.form['note']

    trim_value = request.form['trim_val']

    try:
        if int(trim_value)> 20 or int(trim_value)<0:
            trim_value = 7
        else:
            trim_value = int(trim_value)

    except:
        trim_value = 7
    
    lang = 'english'
    #Summarize
    result_summary = summarize(lang, query, trim_value)
    
    return render_template('submit.html', summarized_text='{}'.format(result_summary))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=33333)
    

