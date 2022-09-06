# Setup
### Configure virtual environment and install dependencies
```
$ python3 -m venv ./venv
$ . ./venv/bin/activate
(venv) $ pip install -r requirements.txt
```
### Copy document collection to current directory as _COLLECTION_
### Start the search service
```
$ . ./venv/bin/activate
(venv) $ python src/tirs-server.py
[nltk_data] Downloading package stopwords to /home/atotev/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to /home/atotev/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to /home/atotev/nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
Indexing collection:
100%|████████████████████████████████████████████████████████████████████████████| 12208/12208 [00:47<00:00, 254.40it/s]
 * Serving Flask app 'tirs-server' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://172.28.194.129:8081/ (Press CTRL+C to quit)
```
### Execute a query
```
(venv) $ python src/tirs.py search "NETWORKING AMONG REBEL GROUPS TROUBLES MEXICAN GOVERNMENT"
./COLLECTION/LA012694-0168.xml   NETWORKING AMONG REBEL GROUPS TROUBLES MEXICAN GOVERNMENT ; REVOLT : CHIAPAS RE
./COLLECTION/LA011294-0018.xml   SALINAS -- BOTH THE DOVE AND THE HAWK
./COLLECTION/LA120994-0269.xml   GOVERNOR TAKES OATH PEACEFULLY IN CHIAPAS ; RIVAL DOES TOO ; MEXICO : BOTH PROM
./COLLECTION/LA022294-0243.xml   MORE FRICTION SEEN AS MEXICO TALKS BEGIN
./COLLECTION/LA012694-0031.xml   CONFRONTING THE STAIN OF CHIAPAS ; WITH ARMY ' S BRUTALITY NEWLY DOCUMENTED , S
./COLLECTION/LA031994-0017.xml   NEWS ANALYSIS ; MEXICO POISED FOR POLITICAL REFORM ; ELECTIONS : SPECIAL_SESSIO
```
### View search client help
```
(venv) $ python src/tirs.py configure -h # press Q to exit help
```
### Change the active language model and configure parameter. (The parameter names are available in each model's implementation file, e.g. rank_lm_jms.py)
```
(venv) $ python src/tirs.py configure active lm
127.0.0.1 - - [22/Feb/2022 22:58:23] "PUT /configure HTTP/1.1" 200 -
Success: True

(venv) $ python src/tirs.py configure lmjms.lambda 0.7
127.0.0.1 - - [22/Feb/2022 22:59:37] "PUT /configure HTTP/1.1" 200 -
Success: True
```
### Search using the new activated model
```
(venv) $ python src/tirs.py search "NETWORKING AMONG REBEL GROUPS TROUBLES MEXICAN GOVERNMENT"
./COLLECTION/LA012694-0168.xml   NETWORKING AMONG REBEL GROUPS TROUBLES MEXICAN GOVERNMENT ; REVOLT : CHIAPAS RE
./COLLECTION/LA011594-0113.xml   WORLD IN BRIEF ; MEXICO ; U.S. COPTERS USED AGAINST UPRISING
./COLLECTION/LA022294-0243.xml   MORE FRICTION SEEN AS MEXICO TALKS BEGIN
./COLLECTION/LA011294-0018.xml   SALINAS -- BOTH THE DOVE AND THE HAWK
./COLLECTION/LA020294-0046.xml   GLOBAL MARKETS AND ECONOMIC UPDATE ; FOREIGN STOCK_MARKETS
./COLLECTION/LA011794-0017.xml   WORLD IN BRIEF : MEXICO ; PRESIDENT PUSHES AMNESTY FOR REBELS
```

# Model evaluation
### Copy topics to current directory as _topics_
### Copy ground truth file to current directory as _test_qrels.txt_
### Copy trec_eval utility directory to current directory as _trec_eval-9.0.7_
### Execute the evaluation script
```
$ python3 -m venv ./venv
$ . ./venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python src/evaluation.py
[nltk_data] Downloading package stopwords to /home/atotev/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to /home/atotev/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package omw-1.4 to /home/atotev/nltk_data...
[nltk_data]   Package omw-1.4 is already up-to-date!
100%|████████████████████████████████████████████████████████████████████████████| 12208/12208 [01:48<00:00, 112.20it/s]
Evaluating vector space model
100%|███████████████████████████████████████████████████████████████████████████████| 162/162 [1:01:26<00:00, 22.75s/it]
map                     all     0.1848
P_5                     all     0.1712
...
```