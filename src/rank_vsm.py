from os import walk, path
from collections import Counter
import numpy as np
from datetime import datetime
from inverted_index import *
from rank import *


class VsmRanking(RankingStrategy):
    TERM_WEIGHTS = 'vsm.term_weights'
    QUERY_WEIGHTS = 'vsm.query_weights'
    DOC_NORM = 'vsm.doc_norm'
    QUERY_NORM = 'vsm.query_norm'
    
    def __init__(self, index):
        self._index = index
        self._config = {}
    
    def set_param(self, name, value):
        self._config[name] = value
        
    def _calc_norm(self, term_counts):
        return np.sqrt(np.sum(np.square(np.array(list(term_counts.values())))))
    
    def _calc_tfidf(self, term_counts):
        n = self._index.get_document_count()
        def calc_weight(kv):
            t = kv[0]
            tf = kv[1]
            df = len(self._index.get_posting_list(t))
            return t, (1 + np.log(tf)) * np.log(1 + (n / df))
        return Counter(dict(map(calc_weight, term_counts.items())))

    def update_rank(self, search_result, query_terms, qt):
        if qt not in search_result.terms:
            return
        
        sr_data = search_result.custom_data
        
        if VsmRanking.TERM_WEIGHTS not in sr_data:
            sr_data[VsmRanking.TERM_WEIGHTS] = self._calc_tfidf(search_result.terms)
            
        if VsmRanking.DOC_NORM not in sr_data:
            sr_data[VsmRanking.DOC_NORM] = self._calc_norm(sr_data[VsmRanking.TERM_WEIGHTS])
        
        if VsmRanking.QUERY_WEIGHTS not in sr_data:
            sr_data[VsmRanking.QUERY_WEIGHTS] = self._calc_tfidf(query_terms)
            
        if VsmRanking.QUERY_NORM not in sr_data:
            sr_data[VsmRanking.QUERY_NORM] = self._calc_norm(query_terms)

        # calculate cosine similarity with the query term vector, then add to relevance score
        search_result.relevance += sr_data[VsmRanking.TERM_WEIGHTS][qt] * sr_data[VsmRanking.QUERY_WEIGHTS][qt] / sr_data[VsmRanking.DOC_NORM] * sr_data[VsmRanking.QUERY_NORM]
