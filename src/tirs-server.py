from flask import Flask
from inverted_index import InvertedIndex
from search import Search
import json
from flask import Response
from json import JSONEncoder
from flask import request

collection_dir = './COLLECTION'
index = InvertedIndex()

print('Indexing collection:')
index.add_files(collection_dir)

search = Search(index).configure(Search.PARAM_ACTIVE, Search.IR_LM)

app = Flask(__name__)

@app.route('/')
def index():
    class SearchResultEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__
    sr = search.execute(request.args.get('query'))
    return Response(json.dumps(sr, cls=SearchResultEncoder),  mimetype='application/json')

@app.route('/configure', methods = ['PUT'])
def configure():
    data = request.get_json()
    search.configure(data['param_name'], data['param_value'])
    return {}

app.run(host='0.0.0.0', port=8081)
