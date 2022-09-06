from os import walk, path
from lxml import etree
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from tqdm import tqdm
import pickle
from collections import Counter
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words(InvertedIndex.LANGUAGE))
        self.lemmatizer = WordNetLemmatizer()
    
    def count_terms(self, text):
        if not text:
            return Counter()
        f = lambda word: InvertedIndex.TERM_FILTER_REGEX.match(word) and word not in self.stop_words
        lemmatized = map(self.lemmatizer.lemmatize, text.lower().split())
        return Counter(filter(f, lemmatized))

class FileLoader:
    ELMNT_DOCID = 'DOCID'
    ELMNT_HEADLINE = 'HEADLINE'
    ELMNT_TEXT = 'TEXT'
    ELMNT_QUERYID = 'QUERYID'
    ELMNT_TITLE = 'TITLE'
    
    def load_document(self, filepath):
        tree = etree.parse(filepath)
        return (tree.find(FileLoader.ELMNT_DOCID).text, tree.find(FileLoader.ELMNT_HEADLINE).text, tree.find(FileLoader.ELMNT_TEXT).text)

    def load_query_file(self, filepath):
        tree = etree.parse(filepath)
        return (tree.find(FileLoader.ELMNT_QUERYID).text, tree.find(FileLoader.ELMNT_TITLE).text)
                           
class Posting:
    def __init__(self, docid, count):
        self.docid = docid
        self.count = count

class InvertedIndex:
    LANGUAGE = 'english'
    TERM_FILTER_REGEX = re.compile('^[a-z][a-z_\\-]*[a-z]$') # length>1, no punctuation-only
    
    def __init__(self):
        self._document_count = 0
        self._documents_total_length = 0
        self._index = {}
        self._document_files = {}
        self._file_loader = FileLoader()
        self._test_proc = TextProcessor()
        
    def _add_postings(self, docid, text):
        tcounts = self._test_proc.count_terms(text)
        for t in tcounts:
            if t not in self._index:
                self._index[t] = []
            self._index[t].append(Posting(docid, tcounts[t]))
        return tcounts

    def add_files(self, document_dir):
        all_documents_length = 0
        basepath, _, files = next(walk(document_dir))
        for each in tqdm(files):
            filepath = path.join(document_dir, each)
            docid, _, text = self._file_loader.load_document(filepath)
            self._document_files[docid] = filepath
            term_counts = self._add_postings(docid, text)
            self._documents_total_length += sum(term_counts.values())
            self._document_count += 1
    
    def get_avg_document_length(self):
        return self._documents_total_length / self._document_count
    
    def get_documents_total_length(self):
        return self._documents_total_length
            
    def keys(self):
        return self._index.keys()
    
    def get_posting_list(self, term):
        return self._index[term]
    
    def get_document_file(self, docid):
        return self._document_files[docid]
    
    def get_document_count(self):
        return self._document_count
