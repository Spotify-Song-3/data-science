import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')
sid = SentimentIntensityAnalyzer()

import spacy
from spacy.tokenizer import Tokenizer
nlp = spacy.load("en_core_web_lg")
tokenizer = Tokenizer(nlp.vocab)
STOP_WORDS = nlp.Defaults.stop_words

import pandas as pd 
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def processed_score(data=lyrics):
    tokens = []
    df = pd.DataFrame(data.split(), columns=['words'])

    for doc in tokenizer.pipe(df['words']):
        doc_tokens = []
        for token in doc:
            if (token.is_stop == False) & (token.is_punct == False):
                doc_tokens.append(token.text.lower())
        tokens.append(doc_tokens)

    df['tokens'] = tokens
    word_list = sum(list([item for item in df['tokens'] if len(item) != 0]), [])
    lyrics_processed = ' '.join(word_list)
    scores = sid.polarity_scores(lyrics_processed)

    return scores

def stemmed_score(data=lyrics):
    """ Processes the text via spacy """
    tokens = []
    words = []

    # Stemming
    for word in data.split():
        words.append(ps.stem(word))

    df = pd.DataFrame(words, columns=['words'])

    for doc in tokenizer.pipe(df['words']):
        doc_tokens = []
        for token in doc:
            if (token.is_stop == False) & (token.is_punct == False):
                doc_tokens.append(token.text.lower())
        tokens.append(doc_tokens)

    df['tokens'] = tokens
    word_list = sum(list([item for item in df['tokens'] if len(item) != 0]), [])
    lyrics_processed = ' '.join(word_list)
    scores = sid.polarity_scores(lyrics_processed)

    return scores

def get_lemmas(data):
    """ Gets lemmas for text """
    lemmas = []
    doc = nlp(data)
    
    for token in doc: 
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_!= 'PRON'):
            lemmas.append(token.lemma_)
    
    return lemmas

def lemma_score(data=lyrics):
    """ Processes the text via spacy """
    tokens = []
    words = []

    # Lemmatization
    words = get_lemmas(data)

    df = pd.DataFrame(words, columns=['words'])

    for doc in tokenizer.pipe(df['words']):
        doc_tokens = []
        for token in doc:
            if (token.is_stop == False) & (token.is_punct == False):
                doc_tokens.append(token.text.lower())
        tokens.append(doc_tokens)

    df['tokens'] = tokens
    word_list = sum(list([item for item in df['tokens'] if len(item) != 0]), [])
    lyrics_processed = ' '.join(word_list)
    scores = sid.polarity_scores(lyrics_processed)

    return scores
