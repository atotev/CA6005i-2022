from os import walk, path
from tqdm import tqdm
from collections import Counter
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from inverted_index import *

class SearchResult:
    def __init__(self, docid):
        self.docid = docid
        self.filepath = None
        self.headline = None
        self.terms = Counter()
        self.relevance = 0.
        self.custom_data = {}

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__)

class RankingStrategy(ABC):
    @abstractmethod
    def set_param(self, name, value):
        raise Exception('Abstract method call attempted')

    @abstractmethod
    def update_rank(self, search_result, query_terms, qt):
        raise Exception('Abstract method call attempted')
