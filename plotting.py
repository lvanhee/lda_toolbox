from gensim import similarities
from wordcloud import WordCloud
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import os
import pandas as pd
import text_preprocessing
from csv import reader
import post_processing_lda
from nltk.corpus import stopwords
import csv


def raw_text_to_wordcloud(long_string, filename):
    # Join all the text in one string
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    wordcloud.to_file(filename)
    wordcloud.to_image()


def export_lda_results_as_html(lda_model, corpus, file_name):
    # Visualize the topics
    # pyLDAvis.enable_notebook()
    #visualisation = pyLDAvis.gensim_models.prepare(lda_model, corpus, lda_model.id2word)
    #pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

    LDAvis_data_filepath = os.path.join('results/tmp_' + str(lda_model.num_topics))

    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, lda_model.id2word, mds='mmds')
        with open(LDAvis_data_filepath, 'wb') as f:
            pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, file_name)

    LDAvis_prepared

def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    # Format
    df_dominant_topic = sent_topics_df.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.head(10)
    return(sent_topics_df)

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

def generate_wordclouds_lda(lda_model):


    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=text_preprocessing.stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


# Sentence Coloring of N Sentences
from matplotlib.patches import Rectangle

def sentences_chart(lda_model=None, corpus=None, texts = None, start = 0, end = 13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i-1]
            bow = lda_model.id2word.doc2bow(texts[i-1])
            #original_call = lda_model[corp_cur]
            updated_call = lda_model.get_document_topics(bow, per_word_topics=True)
            topic_percs, wordid_topics, wordid_phivalues = updated_call
            #original formula; failed as it was returning only one output. Tried adding per_word_topics=True in the
            # constructor but then all went bananas: lda_model[corp_cur]

            lda_model.get_term_topics(1)

            word_dominanttopic = [(lda_model.id2word[word_index],post_processing_lda.get_topic_in_which_the_word_is_the_most_frequently_used(lda_model,word_index)) for (word_index, _) in bow]
            ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
    plt.tight_layout()
    plt.show()



def cluster_test(corpus, model):
    docs_with_1_topic = 0
    docs_with_multiple_topics = 0
    docs_with_no_topics = 0
    total_docs = 0
    for doc in corpus:
        topics = model.get_document_topics(doc, minimum_probability=0.20)
        total_docs += 1
        if len(topics) == 1:
            docs_with_1_topic += 1
        elif len(topics) > 1:
            docs_with_multiple_topics += 1
        else:
            docs_with_no_topics += 1
    print('Corpus assigned to a single topic:', (docs_with_1_topic / total_docs) * 100, '%')
    print('Corpus assigned to multiple topics:', (docs_with_multiple_topics / total_docs) * 100, '%')
    print('corpus assigned to no topics:', (docs_with_no_topics / total_docs) * 100, '%')


# define retrieve similar documents
def retrieval_test(new_doc, lda, dictionary,corpus):
    new_bow = dictionary.doc2bow(new_doc)  # change new document to bag of words representation
    new_vec = lda[new_bow]  # change new bag of words to a vector
    transformed_corpus = lda[corpus]
    index = similarities.MatrixSimilarity(transformed_corpus)
    index.num_best = 10  # set index to generate 10 best results
    matches = (index[new_vec])
    scores = []
    for match in matches:
        score = (match[1])
        scores.append(score)
        score = str(score)
        key = 'doc_' + str(match[0])
        #article_dict = doc2metadata[key]
        #author = article_dict['author']
        #title = article_dict['title']
        #year = article_dict['pub_year']
        #print(key + ': ' + author.title() + ' (' + year + '). ' + title.title() + '\n\tsimilarity score -> ' + score + '\n')
        print(key)

def export_topic_distribution_per_document(lda_model, corpus, texts_as_word_lists):
    if os.path.exists("results/topic_distribution_per_document.csv"):
        return

    with open("results/topic_distribution_per_document.csv", "w", newline='') as f:
        writer = csv.writer(f, dialect='excel')
        num_topic = lda_model.num_topics
        header = list(map(lambda x: "topic "+ str(x), list(range(0, num_topic))))
        header.insert(0,"id")
        writer.writerow(header)
        lines = [[idx]+item for idx, item in enumerate(list(map(lambda x: [y[1] for y in post_processing_lda.get_topic_distribution_of_document(lda_model,x)],texts_as_word_lists)))]
        writer.writerows(lines)


def export_topic_per_word_distribution(lda_model, words):
    if os.path.exists("results/topic_per_word_distribution.csv"):
        return


    num_topic = lda_model.num_topics
    header = ["word"]+list(map(lambda x: "proportion in topic " + str(x), list(range(0, num_topic))))+\
             list(map(lambda x: "p(topic" + str(x)+"|word)", list(range(0, num_topic))))
    frequency_per_word =  dict(map(lambda w: (w,
                              [
                                  post_processing_lda.get_frequency_in_topics_given_a_word(
                                      lda_model, w).get(t,0) for t in range(0, num_topic)]),
                   words))
    #dict(map(lambda x: (w,[y[1] for y in post_processing_lda.get_frequency_in_topics_given_a_word(lda_model,w)]),words))


    relative_frequency_per_word = dict(map(lambda w: (w,
                                                      [
                                                          post_processing_lda.get_relative_frequency_in_topics_given_a_word_and_a_topic(
                                                              lda_model, w, t) for t in range(0, num_topic)]),
                                           words))



    lines = [[w]+frequency_per_word.get(w)+relative_frequency_per_word.get(w) for w in words]
    with open("results/topic_per_word_distribution.csv", "w", newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerow(header)
        writer.writerows(lines)


def export_word_distribution_per_topic(lda_model):
    if os.path.exists("results/word_per_topic.csv"):
        return

    res = lda_model.show_topics()

    with open("results/word_per_topic.csv", "w", newline='') as f:
        writer = csv.writer(f, dialect='excel')
        writer.writerows(res)
        #writer.writerows(lines)