from os import walk, path
from collections import Counter
import numpy as np
from datetime import datetime
from inverted_index import *
from rank import *

class LmJmsRanking(RankingStrategy):
    PARAM_LAMBDA = 'lmjms.lambda'
    
    def __init__(self, index):
        self._index = index
        self._config = { LmJmsRanking.PARAM_LAMBDA: 0.25 }
    
    def set_param(self, name, value):
        self._config[name] = value
        
    def update_rank(self, search_result, query_terms, qt):
        Cd = search_result.terms[qt]
        sumCd = sum(search_result.terms.values())
        CD = sum(p.count for p in self._index.get_posting_list(qt))
        sumCD = self._index.get_documents_total_length()
        lmbda = self._config[LmJmsRanking.PARAM_LAMBDA]
        search_result.relevance += np.log(((1 - lmbda) * Cd / sumCd) + (lmbda * CD / sumCD))
