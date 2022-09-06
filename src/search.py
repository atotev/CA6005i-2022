from os import walk, path
from tqdm import tqdm
from collections import Counter
import numpy as np
from datetime import datetime
from inverted_index import *
from rank import *
from rank_vsm import *
from rank_bm25 import *
from rank_lm_jms import *


class Search:
    PARAM_ACTIVE = 'active'

    IR_VSM = 'vsm'
    IR_BM25 = 'bm25'
    IR_LM = 'lm'
    
    RESULT_LIST_SIZE = 1000
    
    def __init__(self, index):
        self._index = index
        self._irModels = {
            Search.IR_VSM: VsmRanking(self._index),
            Search.IR_BM25: Bm25Ranking(self._index),
            Search.IR_LM: LmJmsRanking(self._index)
        }
        self._ACTIVE_RANKING = self._irModels[Search.IR_VSM]
        self._file_loader = FileLoader()
        self._text_proc = TextProcessor()
        
    def _remove_unkown(self, term_counts):
        return Counter({x: count for x, count in term_counts.items() if x in self._index.keys()})
    
    def execute(self, query_text):
        search_results = {}
        query_terms = self._text_proc.count_terms(query_text)
        query_terms = self._remove_unkown(query_terms)
        for qt in query_terms:
            posting_list = self._index.get_posting_list(qt)
            for p in posting_list:
                if p.docid not in search_results:
                    sr = SearchResult(p.docid)
                    sr.filepath = self._index.get_document_file(p.docid)
                    _, sr.headline, text = self._file_loader.load_document(sr.filepath)
                    sr.terms = self._text_proc.count_terms(text)
                    search_results[p.docid] = sr
        for qt in query_terms:
            for docid in search_results:
                self._ACTIVE_RANKING.update_rank(search_results[docid], query_terms, qt)
        result = list(search_results.values())
        result.sort(reverse=True, key=lambda sr: sr.relevance)
        return result[:Search.RESULT_LIST_SIZE]

    def configure(self, param_name, param_value):
        if Search.PARAM_ACTIVE!=param_name:
            self._ACTIVE_RANKING.set_param(param_name, param_value)
        elif param_value in self._irModels:
            self._ACTIVE_RANKING = self._irModels[param_value]
        else:
            raise Exception('Unrecognized ranking: %s' % (param_value))
        return self
