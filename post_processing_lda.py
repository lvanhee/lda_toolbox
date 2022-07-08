
def get_topic_distribution_of_document(lda_model,doc):
    bow = lda_model.id2word.doc2bow(doc)
    return lda_model.get_document_topics(bow,minimum_probability=0)

def get_dominant_topic_of_document(lda_model, doc):

    topic_percs = get_topic_distribution_of_document(lda_model,doc)
    # topic_percs, wordid_topics, wordid_phivalues = lda_model[corp]
    return sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]

def get_topic_in_which_the_word_is_the_most_frequently_used(lda_model,w):
    if isinstance(w, int):
        w = lda_model.id2word[w]

    topic_distr = get_frequency_in_topics_given_a_word(lda_model,w)
    res = max(topic_distr, key=topic_distr.get)
    return res


def dominant_topic_and_topic_distribution_per_document_in_corpus(lda_model, corpus, texts_as_word_lists):
        corpus_sel = corpus
        dominant_topics = []
        topic_percentages = []
        for i, corp in enumerate(corpus_sel):
            topic_percs = get_topic_distribution_of_document(lda_model, texts_as_word_lists[i])
            #topic_percs, wordid_topics, wordid_phivalues = lda_model[corp]
            dominant_topic = get_dominant_topic_of_document(lda_model,texts_as_word_lists[i])
            dominant_topics.append((i, dominant_topic))
            topic_percentages.append(topic_percs)
        return (dominant_topics, topic_percentages)

def get_frequency_of_words_per_topic(lda_model, words):
    return [(lda_model.id2word[word_index], lda_model.get_term_topics(word_index, 0)) for (word_index, _) in words]

#Given a word as input, this function determines how frequent it is for every topic
def get_frequency_in_topics_given_a_word(lda_model, w):
    if isinstance(w, int):
        w = lda_model.id2word[w]

    return dict(map(lambda x:x,lda_model.get_term_topics(w, 0)))


def get_relative_frequency_in_topics_for_every_word(lda_model,bow):
    frequency_per_topic_per_word = get_frequency_of_words_per_topic(lda_model,bow)
    relative_frequency_per_word = dict(map(lambda x: (x[0], sum([v[1] for v in x[1]])), frequency_per_topic_per_word))

    relative_frequency_per_topic_per_word = dict(
        map(lambda x: (x[0], ([v[1] * 1 / relative_frequency_per_word.get(x[0]) for v in x[1]])),
            frequency_per_topic_per_word))
    return relative_frequency_per_topic_per_word

def get_relative_frequency_in_topics_given_a_word(lda_model, w):
    freq = get_frequency_in_topics_given_a_word(lda_model,w)
    ratio = sum([v[1] for v in freq])
    res = dict(map(lambda x: (x[0],x[1]/ratio),freq))

    return res

def get_relative_frequency_in_topics_given_a_word_and_a_topic(lda_model, w,t):
    freq = get_frequency_in_topics_given_a_word(lda_model,w)
    ratio = sum(freq.values())
    res = dict(map(lambda x: (x,freq.get(x,0)/ratio),freq.keys()))

    if not t in res:
        return 0

    return res.get(t)
