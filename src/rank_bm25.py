from os import walk, path
from collections import Counter
import numpy as np
from datetime import datetime
from inverted_index import *
from rank import *

class Bm25Ranking(RankingStrategy):
    PARAM_B = 'bm25.b'
    PARAM_K1 = 'bm25.k1'
    
    def __init__(self, index):
        self._index = index
        self._config = { Bm25Ranking.PARAM_B: 0.75, Bm25Ranking.PARAM_K1: 1.25 }
    
    def set_param(self, name, value):
        self._config[name] = value
        
    def update_rank(self, search_result, query_terms, qt):
        if qt not in search_result.terms:
            return
        
        tf = search_result.terms[qt]
        N = self._index.get_document_count()
        n = len(self._index.get_posting_list(qt))
        dl = sum(search_result.terms.values())
        avdl = self._index.get_avg_document_length()
        k1 = self._config[Bm25Ranking.PARAM_K1]
        b = self._config[Bm25Ranking.PARAM_B]
        search_result.relevance += tf / ((k1 * (1 - b + (b * dl / avdl))) + tf) * np.log((N - n + 0.5) / (n + 0.5))
