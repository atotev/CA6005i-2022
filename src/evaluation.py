from inverted_index import *
from search import Search
import subprocess
from os import walk, path
import numpy as np
from datetime import datetime

collection_dir = './COLLECTION'
topics_dir = './topics'

index = InvertedIndex()
index.add_files(collection_dir)

class DocRank:
    def __init__(self, queryid, search_result):
        self.queryid = queryid
        self.sr = search_result
        
    def to_qrel(self):
        return '%s Q0 %s rank %.6f STANDARD\n' % (self.queryid, self.sr.docid, self.sr.relevance)

def evaluate(search):
    queryids = set()
    ranks = []
    _, _, topic_files = next(walk(topics_dir))
    test_topic_files = np.array(topic_files)
    file_loader = FileLoader()
    for each in tqdm(test_topic_files):
        queryid, query_text = file_loader.load_query_file(path.join(topics_dir, each))
        queryids.add(queryid)
        ranks.extend(map(lambda sr: DocRank(queryid, sr), search.execute(query_text)))
            
        
    def results_filepath():
        return './results_%s.txt' % (datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))
    results_file = results_filepath()
    with open(results_file, 'w') as fp:
        fp.writelines(r.to_qrel() for r in ranks)

    qrels_file = '%s.qrels' % (results_file)
    with open(qrels_file, 'w') as fp:
        with open('./test_qrels.txt') as qrelf:
            fp.writelines(line for line in qrelf if any(qid in line for qid in queryids))

    result = subprocess.run(['./trec_eval-9.0.7/trec_eval', '-m', 'ndcg', '-m', 'map', '-m', 'P', qrels_file, results_file], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

search = Search(index).configure(Search.PARAM_ACTIVE, Search.IR_VSM)
print('Evaluating vector space model')
evaluate(search)

search = Search(index).configure(Search.PARAM_ACTIVE, Search.IR_BM25)
print('Evaluating BM25 probabilistic model')
evaluate(search)

search = Search(index).configure(Search.PARAM_ACTIVE, Search.IR_LM)
print('Evaluating languauge model with Jelinek-Mercer smoothing')
evaluate(search)
