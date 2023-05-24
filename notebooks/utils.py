from typing import List
import tomotopy as tp
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed




##### HELPRER FN
def preprocess_docs(text: str, sp, user_data = None) -> List[str]:
    """ 
    Preprocess text: removes punctuation, digits, stop words, lemmatizes words
    Args:
        text: a string (e.g. sentence), not a list (lowercase is expected)
        user_data: placeholder for the Tomotopy input (not used)
    """
    lemmas = [word.lemma_ for word in sp(text) if word.is_alpha and (not word.is_stop)]  
    return lemmas

def chunker(iterable, total_length, chunksize):
    """
    Chunk the iterable object for multithread processing
    """
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists: List) -> List:
    """
    Flatten a list of lists to a combined list
    """
    return [item for sublist in list_of_lists for item in sublist]

def preprocess_pipe(texts: list, sp):
    """
    Helper function for multithread processing of texts
    """
    preproc_pipe = []
    texts = [text.lower() for text in texts]
    for doc in sp.pipe(texts, batch_size=48):
        preproc_pipe.append(preprocess_docs(doc, sp))
    return preproc_pipe


def preprocess_parallel(texts, num_docs: int, sp, chunksize: int =100) -> List:
    """
    Multithread processing of texts (e.g. sentences)
    Args:
        texts: list of strings (e.g. sentences)
        num_docs: total number of strings in the list
        chunksize: specifies how to split data across the threads
        
    """
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(preprocess_pipe)
    tasks = (do(chunk, sp) for chunk in chunker(texts, num_docs, chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

def preprocess_text_sentence(text:str):
    return " ".join(text)


### WRAPPER TO FINETUNE THE MODEL
class TopicWrapper(BaseEstimator):
    """ 
    Wrapper for the Tomotopy HPA model. It simplifies the hyperparamaeter search with Sklearn.
    """
    def __init__(self,  
                k1: int, 
                k2: int, 
                top_n:int = 25, 
                train_iter: int = 500,
                random_state: int = 0,
                num_workers: int = 1,
                ) -> None:
        super().__init__()
        self.random_state = random_state
        self.k1 = k1
        self.k2 = k2
        self.train_iter = train_iter
        self.top_n = top_n
        self.num_workers = num_workers
        self.model = None

    def __init_model__(self):
        """Initialisez the HPA model with specific parameters"""
        return tp.PAModel(tw=tp.TermWeight.PMI, min_cf=10, rm_top=1, 
                          k1=self.k1, k2=self.k2, seed=self.random_state)
    def fit(self, X, **kwargs):
        corpus = tp.utils.Corpus()
        for doc in X:
            if doc: 
                corpus.add_doc(doc)
        self.model = self.__init_model__()
        self.model.add_corpus(corpus)
        self.model.burn_in = 100
        self.model.train(self.train_iter, workers=self.num_workers)
        return self

    def predict(self, X):
        infered_corpus, ll = self.model.infer(X)
        return infered_corpus, ll
    def score(self, *args, **kwargs) -> float:
        """Returns the coherence score"""
        return -tp.coherence.Coherence(self.model,coherence="u_mass").get_score()
    def set_params(self, **params):
        self.model = None
        return super().set_params(**params)


#### Sentiment Analyser (Multilingual)
class SentimentAnalyser():
    def __init__(self, lang, path = "sentiments/") -> None:
        self.lang = lang
        self.path = path
        self.load_lang()
        self.fit()
        print("Sentiment Analyser is fit for %s" %self.lang.upper())
    def load_lang(self):
        neg = pd.read_csv(self.path + "negative_words_%s.txt" %self.lang)
        neg = neg.rename(columns={neg.columns[0]: "word"})
        neg['polarity'] = 0
        self.neg = neg.copy()
        pos = pd.read_csv(self.path + "positive_words_%s.txt" %self.lang)
        pos = pos.rename(columns={pos.columns[0]: "word"})
        pos['polarity'] = 1
        self.pos = pos.copy()
        self.s = pd.concat([self.pos, self.neg], ignore_index=True).set_index("word")
    def fit(self):
        self.vectorizer = CountVectorizer()
        words = self.vectorizer.fit_transform([k for k in self.s["polarity"].keys()])
        self.model = LogisticRegression().fit(words, [v for v in self.s["polarity"].values])
    def get_sentiment(self, x: list):
        """
        Predict for single sentence/document
        """
        assert len(x) == 1
        return self.model.predict_proba(self.vectorizer.transform(x)).flatten()[1]
        