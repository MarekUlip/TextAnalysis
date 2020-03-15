import os
import sys
import random
import numpy
import time

from methods.Lda import Lda
from methods.Lsa import Lsa

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from preprocessor.text_preprocessor import TextPreprocessor
from tests.log_writer import LogWriter
from tests.model_tester import ModelTester

"""
Old test used to find best parameters for models LSA and LDA.
"""

def create_variations(depth, field, all_vars, possibilities):
    if depth == len(all_vars):
        possibilities.append(field)
        return

    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth + 1, f, all_vars, possibilities)

def get_time_in_millis():
    return int(round(time.time())*1000)


log_writer = LogWriter("log.txt")
base_path = os.getcwd()
csv_folder = base_path+"\\csv_folder\\"
data_sets = [(csv_folder+"4"+"\\train.csv",csv_folder+"4"+"\\test.csv",20,"-20newsgroups-"),(csv_folder+"1"+"\\train.csv",csv_folder+"1"+"\\test.csv",10,"-reuters-")]#,(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]
#data_sets = [(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]


strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params]
preproces_variations = []
create_variations(0, [], preproces_all_vals, preproces_variations)

lda_kappa = [0.51]
lda_tau = [2.0]
lda_minimum_probability = [0.0,0.01,0.1]
lda_passes = [50]
lda_iterations = [50]
lda_all_vals = [lda_kappa,lda_tau, lda_passes,lda_iterations]
lda_variations = []
create_variations(0,[],lda_all_vals,lda_variations)

lsa_one_pass = [False]
lsa_power_iter = [2]
lsa_use_tfidf = [True]
lsa_topic_nums = [data_sets[0][2]]
lsa_extra_samples = [100,200]
lsa_decay = [0.5,1.0,2.0]
lsa_all_vals = [lsa_one_pass, lsa_power_iter, lsa_use_tfidf,lsa_topic_nums]
lsa_variations = []
create_variations(0,[],lsa_all_vals,lsa_variations)

hdp_variations = []
num_of_test = 15

test_model = [True, True]
start_time = get_time_in_millis()

for i in range(len(data_sets)):
    lsa_topic_nums = [data_sets[i][2], data_sets[i][2] * 4]
    lsa_all_vals = [lsa_one_pass, lsa_power_iter, lsa_use_tfidf, lsa_topic_nums]
    lsa_variations = []
    create_variations(0, [], lsa_all_vals, lsa_variations)
    statistics_to_merge = []
    for index, preproces_settings in enumerate(preproces_variations):
        seed = 5
        settings = {'strip_nums': preproces_settings[0],
            'use_stemmer': preproces_settings[1],
            'use_lemmatizer': preproces_settings[2],
            'strip_short': preproces_settings[3]
        }
        log_writer.add_log("Initializing text preprocessor with strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}.".format(preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3]))
        text_preprocessor = TextPreprocessor(settings)

        log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
        texts_for_train = text_preprocessor.load_and_prep_csv([data_sets[i][0]], "eng", False, 1, ';')
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
        texts_for_topic_asses = text_preprocessor.load_and_prep_csv([data_sets[i][0]], "eng", True, 1, ';')
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for testing".format(data_sets[i][1]))
        texts_for_testing = text_preprocessor.load_and_prep_csv([data_sets[i][1]], "eng", True, 1, ';')
        log_writer.add_log("Preprocessing finished")

        statistics = []
        if test_model[0]:
            # For every preprocesing add line that descripbes methods used
            statistics.append(["strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}.".format(preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3])])
            accuracies = []
            for model_settings_index, model_settings in enumerate(lda_variations):
                # every column means one settings variation

                if model_settings_index == 0:
                    statistics.append([])
                    statistics.append(["LDA"])
                    statistics.append([x for x in range(num_of_test)])
                    statistics.append([])
                """for j in range(num_of_test):
                    if model_settings_index == 0:
                        lda_statistics.append([])"""
                for j in range(num_of_test):
                    # every row means jth test
                    test_checker_lda = ModelTester(texts_for_topic_asses, texts_for_testing, data_sets[i][2], log_writer) #"""numpy.asfarray(test_checker_lda.topic_distributions)"""
                    lda = Lda(data_sets[i][2], 15, kappa=model_settings[0], tau=model_settings[1], passes=model_settings[2], iterations=model_settings[3]) #TODO remember random state
                    log_writer.add_log("Starting training LDA model")
                    lda.train(texts_for_train)
                    log_writer.add_log("Starting testing LDA model")
                    accuracy = test_checker_lda.test_model(lda, "\\results\\results{}{}\\lda\\{}\\{}\\{}\\{}".format(data_sets[i][3], start_time, i, model_settings_index, index, 0))#j))
                    accuracies.append(accuracy)
                    statistics[len(statistics)-1].append(accuracy)
                    log_writer.add_log("Testing LDA model done with {}% accuracy".format(accuracy * 100))
                    log_writer.add_log("\n\n")
            total_accuracy = sum(accuracies)/len(accuracies)
            statistics[2].append(accuracies)
            log_writer.add_log("Total accuracy is: {}".format(total_accuracy))
            #statistics_to_merge.append(lda_statistics)

        if test_model[1]:
            statistics.append([])
            statistics.append(["LSA"])
            statistics.append([x for x in range(num_of_test)])
            statistics.append([])
            for model_settings_index, model_settings in enumerate(lsa_variations):
                for j in range(num_of_test):
                    test_checker_lsa = ModelTester(texts_for_topic_asses, texts_for_testing, data_sets[i][2], log_writer)
                    lsa = Lsa(model_settings[3], 15, one_pass=model_settings[0],power_iter=model_settings[1], use_tfidf=model_settings[2])
                    lsa.train(texts_for_train)
                    log_writer.add_log("Starting testing LSA model")
                    accuracy = test_checker_lsa.test_model(lsa, "\\results\\results{}{}\\lsa\\{}\\{}\\{}\\{}".format(data_sets[i][3], start_time, i, model_settings_index, index, j))
                    statistics[len(statistics)-1].append(accuracy)
                    log_writer.add_log("Testing LSA model done with {}% accuracy".format(accuracy * 100))
                    log_writer.add_log("\n\n\n")

            statistics.append([])
        statistics_to_merge.append(statistics)

        """for model_settings_index, model_settings in enumerate(hdp_variations):
            for j in range(num_of_test):
                test_checker_hdp = TestChecker(texts_for_testing, data_sets[i][2], log_writer)
                hdp = Hdp(4, 15)
                hdp.train(texts_for_train)
                log_writer.add_log("Starting testing HDP model")
                accuracy = test_checker_hdp.test_model(hdp, "\\results\\hdp\\{}\\{}\\{}\\{}".format(i, model_settings_index, index, j))
                log_writer.add_log("Testing HDP model done with {}% accuracy".format(accuracy * 100))
                log_writer.add_log("\n\n\n")"""

    output_lda_csv = []
    for item in statistics_to_merge:
        for statistic in item:
            output_lda_csv.append(statistic)
    log_writer.write_2D_list("\\results\\results-stats\\stats{}{}".format(data_sets[i][3],start_time), output_lda_csv)

log_writer.end_logging()
