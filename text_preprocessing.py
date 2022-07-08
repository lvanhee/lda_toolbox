import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import spacy
import csv
import os
from csv import reader


stop_words = stopwords.words('english')
stop_words.extend(
    ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
     'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'high',
     'show',
     'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

def text_list_to_preprocessed_word_list(data):

    if os.path.exists("results/preprocessed_lists_of_words.csv"):
        with open("results/preprocessed_lists_of_words.csv", 'r') as read_obj:
            # pass the file object to reader() to get the reader object
            csv_reader = reader(read_obj)
            # Pass reader object to list() to get a list of lists
            list_of_rows = list(csv_reader)
            return list_of_rows


    def word_list_from_text_list(texts):
        for text in texts:
            # deacc=True removes punctuations
            yield (gensim.utils.simple_preprocess(str(text), deacc=True))

    data_words = list(word_list_from_text_list(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50)  # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=50)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc))
                 if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    print(">loading spacy")
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    print(">string to word lists")
    data_words = list(word_list_from_text_list(data))
    print(">filtering stopwords")
    data_words_nostops = remove_stopwords(data_words)
    print(">lemmatizing")
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print(">replacing bigrams")
    data_words_bigrams = make_bigrams(data_lemmatized)
    print(">replacing trigrams")
    data_words_bigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words_bigrams]

    data_words_bigrams = remove_stopwords(data_words_bigrams)

    print(">preprocessing done, caching")
    #return remove_stopwords(data_lemmatized)

    with open("results/preprocessed_lists_of_words.csv", "w", newline='') as f:
        wr = csv.writer(f)
        wr.writerows(data_words_bigrams)
    print(">preprocessing finished")
    return data_words_bigrams