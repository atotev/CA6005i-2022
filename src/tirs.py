import fire
import requests
import json

BASE_URL = 'http://localhost:8081'
HEADLINE_MAX_LEN = 80

class Search(object):
    def search(self, query):
        r = requests.get(BASE_URL, params={ 'query': query })
        for sr in r.json():
            print('%s\t%s' % (sr['filepath'], sr['headline'][:HEADLINE_MAX_LEN]))

    def configure(self, parameter, value):
        r = requests.put(BASE_URL + '/configure', json={ 'param_name': parameter, 'param_value': value })
        print('Success: %s' % (r.status_code == requests.codes.ok))

if __name__ == '__main__':
  fire.Fire(Search)
