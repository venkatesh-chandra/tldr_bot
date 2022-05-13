#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 07:11:32 2022

@author: venkateshchandra
"""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import SnowballStemmer
import re
import numpy as np
from operator import itemgetter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#----------Section 1 - Tokenize and preprocess----------
def sent_tokenizer(txt):

        txt_list = []
        for line in txt.splitlines():
            line = re.sub(r"\.'|\.\"", "'.", line)
            if line and not re.search(r"\.$", line):
                line += "."
            txt_list.append(line)
        txt_string = " ".join(txt_list)
        tokens = sent_tokenize(txt_string)
        sentences = []
        for s in tokens:
            s = re.sub("\n", " ", s)
            s = re.sub(" +", " ", s)
            s.strip()
            sentences.append(s)
        return sentences
    
def tokenize(corpus):
        tokens = [word_tokenize(t) for t in sent_tokenize(corpus)]
        return tokens

#----------Section 2 - Preprocess: POS tagger and stem---------

def preprocess(lang, sentences):
        """
        Preprocesses sentence tokens: 1) removes stopwords, 2) removes special characters, 3) removes and leading and
        trailing spaces, and 4) transforms all words to lowercase.
        Returns: 2D List of words per sentence: [[w1, w2, w3], [w4, w5, w6], ... [wi, wi, wi]]
        @params:
            sentences   -Required  : 2D list of words per sentence (Lst)
        """
        tokens = sentences
        sw = stopwords.words(lang)
        preprocessed_tokens = []
        for index, s in enumerate(tokens):
            preprocessed_tokens.append([])
            terms = word_tokenize(s)
            for t in terms:
                t = re.sub("\n", " ", t)
                t = re.sub("[^A-Za-z]+", " ", t)
                t = re.sub(" +", " ", t)
                t = t.strip()
                t = t.lower()
                if t and t not in sw:
                    preprocessed_tokens[index].append(t)
        return preprocessed_tokens

def tag_pos(preprocessed_tokens, pos=['NN', 'ADJ']):

        tagged_tokens = []
        for index, s in enumerate(preprocessed_tokens):
            tagged_tokens.append([])
            for t in s:
                t = pos_tag([t])
                if t[0][1] in pos:
                    tagged_tokens[index].append(t[0][0])
        return tagged_tokens
    
def stem(lang, tagged_tokens):

        stemmer = SnowballStemmer(lang)
        stemmed_tokens = []
        for index, s in enumerate(tagged_tokens):
            stemmed_tokens.append([])
            for t in s:
                t = stemmer.stem(t)
                stemmed_tokens[index].append(t)
        return stemmed_tokens

#----------Section 3 - Textrank and similarity----------

def textrank(A, eps=0.0001, d=0.85):
        """
        Applies TextRank algorithm to pairwise similarity matrix.
        Returns: Ranked sentences unsorted (numpy.ndarray)
        @params:
            A       -Required  : Pairwise similarity matrix (Lst)
            eps     -Optional  : stop the algorithm when the difference between 2 consecutive iterations is smaller or
                                 equal to eps. eps=0.0001 by default (Flt)
            d       -Optional  : damping factor: With a probability of 1-d the user will simply pick a web page at random
                                 as the next destination, ignoring the link structure completely. d=0.85 by defaul (Flt)
        """
        P = np.ones(len(A)) / len(A)
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                return new_P
            P = new_P
            
def build_similarity_matrix(stemmed_tokens):

    token_strings = [" ".join(sentence) for sentence in stemmed_tokens]
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.50)
    X = vectorizer.fit_transform(token_strings)
    cosine_similarities = linear_kernel(X, X)
    for index1 in range(len(cosine_similarities)):
        for index2 in range(len(cosine_similarities)):
            if index1 == index2:
                cosine_similarities[index1][index2] = 0
    for index in range(len(cosine_similarities)):
        if cosine_similarities[index].sum() == 0:
            continue
        else:
            cosine_similarities[index] /= cosine_similarities[index].sum()
    return cosine_similarities

#----------Section 4 - main---------

def summarize(lang, corpus, length=7):

    sentences = sent_tokenizer(corpus)
    preprocessed_tokens = preprocess(lang, sentences)
    tagged_tokens = tag_pos(preprocessed_tokens)
    stemmed_tokens = stem(lang, tagged_tokens)
    sentence_ranks = textrank(build_similarity_matrix(stemmed_tokens))
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    selected_sentences = sorted(ranked_sentence_indexes[:length])
    summary = itemgetter(*selected_sentences)(sent_tokenizer(corpus))
    str_summary = "\n\n".join(summary)#' '.join(summary)#

    str_summary = str_summary.replace('\n', '<br />')
    return str_summary

