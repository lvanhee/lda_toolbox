import gensim.corpora as corpora
import gensim
import tqdm
import text_preprocessing
from gensim.models import CoherenceModel
import numpy as np
import pandas as pd
import os.path
from os import path
import plotting

#Mallet:
#https://github.com/maria-antoniak/little-mallet-wrapper/blob/master/demo.ipynb


mallet_path = "C:\\Users\\loisv\\Desktop\\Code\\class_code\\ai2\pylda\\resources\\mallet-2.0.8\\bin\\mallet"
def compute_coherence_values(corpus, dictionary, k, a, b, passes,input_data):
   # lda_model2 = gensim.models.LdaMulticore(corpus=corpus,
   #                                        id2word=dictionary,
   #                                        num_topics=k,
   #                                        random_state=100,
                                          # chunksize=100,
                                          # passes=10,
   #                                        alpha=a,
   #                                        eta=b)
    lda_model = compute_lda(passes,corpus = corpus, dictionary = dictionary, num_topics=k, a=a, b=b)


    print(lda_model.print_topics())
    #print(lda_model2.print_topics())

    coherence_model_lda_c_v = CoherenceModel(model=lda_model, corpus= corpus, texts = input_data, dictionary=dictionary, coherence='c_v').get_coherence()
    coherence_model_lda_u_mass = CoherenceModel(model=lda_model, corpus=corpus,coherence='u_mass').get_coherence()



    return (coherence_model_lda_c_v, coherence_model_lda_u_mass)


def term_frequency_from_text(texts):
    id2word = corpora.Dictionary(texts)
    return [id2word.doc2bow(text) for text in texts]


def compute_lda(passes, texts_as_word_lists=None, corpus = None, dictionary=None, num_topics=10, a=0.05, b=0.05):
    if(corpus ==None):
        corpus = term_frequency_from_text(texts_as_word_lists)
    # Create Dictionary
    if(dictionary==None):
        id2word = corpora.Dictionary(texts_as_word_lists)
    else:
        id2word = dictionary

    # Term Document Frequency

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           # chunksize=100,
                                           passes=passes,
                                           alpha=a,
                                           eta=b)

    #lda_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=k, id2word=dictionary)
    return lda_model


def compute_optimal_hyperparameters(texts_as_word_lists):

    if os.path.exists("results/best_lda_model.obj"):
        return gensim.models.ldamodel.LdaModel.load("results/best_lda_model.obj")

    id2word = corpora.Dictionary(texts_as_word_lists)
    corpus = [id2word.doc2bow(text) for text in texts_as_word_lists]

    # Topics range
    min_topics = 2
    max_topics = 20
    step_size = 1
    topics_range = range(min_topics, max_topics, step_size)

    num_of_docs = len(corpus)
    corpus_sets = [#gensim.utils.ClippedCorpus(corpus, int(num_of_docs * 0.75)),
                   corpus]
    corpus_title = [#'75% Corpus',
                    '100% Corpus']

    # Alpha parameter
    alpha = list(np.arange(0.05, 1, 1))
    alpha.append('symmetric')
    #alpha.append('asymmetric')

    # Beta parameter
    beta = list(np.arange(0.05, 1, 1))
    beta.append('symmetric')

    passes = list(np.arange(10, 50, 10))


    pbar = tqdm.tqdm(total=(len(beta) * len(alpha) * len(topics_range) * len(corpus_title) * len(passes)))



    model_results = pd.DataFrame.from_dict({'Validation_Set': [],
                     'Topics': [],
                     'Alpha': [],
                     'Beta': [],
                     'Coherence': []
                     })

    max_coherence = None
    best_parameters = None

    if(path.exists('./results/lda_tuning_results.csv')):
        model_results = pd.read_csv('./results/lda_tuning_results.csv')
        if(len(model_results.index)>0):
            index_max_coherence = model_results['coherence_umass'].idxmax()
            a = model_results['Alpha'][index_max_coherence]
            try:
                a = float(a)
            except ValueError:
                print("")

            b = model_results['Beta'][index_max_coherence]
            try:
                b = float(b)
            except ValueError:
                print("")

            max_coherence = float(model_results['coherence_umass'][index_max_coherence])

            nt = model_results['Topics'][index_max_coherence]
            best_parameters = gensim.models.LdaModel(corpus=corpus,#it disregards whether it was on a smaller corpus, but whatever
                                                         id2word=id2word,
                                                         num_topics= nt,
                                                         random_state=100,
                                                         #chunksize=100,
                                                         #passes=10,
                                                         alpha=a,
                                                         eta=b
                                                         )




    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through alpha values
        for a in alpha:
            # iterare through beta values
            for b in beta:
                # iterate through number of topics
                for k in topics_range:
                    # iterate through number of topics
                    for p in passes:
                        if(len(model_results['Validation_Set'])>0):
                            #print(corpus_title[i]+" "+str(k)+" "+str(a)+" "+str(b))
                            fitting_result = model_results[(model_results['Validation_Set'] == corpus_title[i]) &
                                                             (model_results['Topics'] == k) &
                                                             (model_results['Passes'] == p) &
                                                           ((model_results['Alpha'] == a) | (model_results['Alpha'] == str(a))) &
                                                           ((model_results['Beta'] == b) | (model_results['Beta'] == str(b)))]
                            list_of_matches = (fitting_result.index.tolist())
                            if 0 < len(list_of_matches):
                                pbar.update(1)
                                continue
                        # get the coherence score for the given parameters
                        (cv, umass) = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, passes = p,
                                                      k=k, a=a, b=b, input_data=texts_as_word_lists)
                        # Save the model results
                        model_results = model_results.append({'Validation_Set': corpus_title[i], 'Topics': k, 'Alpha': a,
                                                              'Beta': b,'Passes': p, 'coherence_cv': cv,
                                                              'coherence_umass':(umass)}, ignore_index=True)

                        print(" "+corpus_title[i]+" "+str(k)+" "+str(a)+" "+str(b)+" passes:"+str(p)
                              +"->"+str(cv)+" "+str(umass))
                        #model_results['Validation_Set'].append(corpus_title[i])
                        #model_results['Topics'].append(k)
                        #model_results['Alpha'].append(a)
                        #model_results['Beta'].append(b)
                        #model_results['Coherence'].append(cv)

                        if(max_coherence == None or umass > max_coherence):
                            print("New best!")
                            best_parameters = compute_lda(corpus=corpus, dictionary=id2word, num_topics=k, a=a, b=b, passes=p)

                            max_coherence = umass
                            plotting.export_lda_results_as_html(best_parameters, [best_parameters.id2word.doc2bow(text)
                                                                                  for text in texts_as_word_lists],
                                                                "results/ongoing-optimization" +
                                                                # str(lda_model)+" "+str(lda_model.eta)+
                                                                ".html")
                        pd.DataFrame(model_results).to_csv('./results/lda_tuning_results.csv', index=False)


                        pbar.update(1)
    pbar.close()

    best_parameters.save("results/best_lda_model.obj")

    # Show graph
    import matplotlib.pyplot as plt

    for p in passes:
        data_coherence = []
        data_coherence2 = []

        for x in topics_range:
            fitting_result = model_results[(model_results['Topics'] == x) & (model_results['Passes'] == p)]['coherence_umass']
            data_coherence.append(sum(fitting_result) / len(fitting_result))

            fitting_result_cv = model_results[((model_results['Topics'] == x)& (model_results['Passes'] == p))]['coherence_cv']
            data_coherence2.append(sum(fitting_result_cv) / len(fitting_result_cv))

        plt.plot(topics_range, data_coherence, label="umass "+str(p))

        plt.plot(topics_range, data_coherence2, label="cv "+str(p))


    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')

    plt.savefig("results/coherence.png")
    #plt.show()


    return best_parameters
