import os
import sys
import time
from preprocessor.helper_functions import Dataset_Helper
from tests.ModelType import ModelType
from tests.general_tester import GeneralTester

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from preprocessor.text_preprocessor import TextPreprocessor
from tests.log_writer import LogWriter



def create_variations(depth, field, all_vars, possibilities):
    if depth == len(all_vars):
        possibilities.append(field)
        return

    for item in all_vars[depth]:
        f = [a for a in field]
        f.append(item)
        create_variations(depth + 1, f, all_vars, possibilities)


def get_time_in_millis():
    return int(round(time.time()) * 1000)


log_writer = LogWriter("log.txt")
base_path = os.getcwd()
csv_folder = base_path + "\\csv_folder\\"
data_sets = [(csv_folder + "1", 10, "-reuters-"),
             (csv_folder + "2", 11, "-DBpedia-"),
             (csv_folder + "3", 4, "-AGsNews-"),
             (csv_folder+"4",20,"-20newsgroups-"),
             (csv_folder + "5", 14, "-DBpediaReal-"),
             (csv_folder + "6", 4, "-AGsNewsReal-"),
             (csv_folder + "7", 8, "-Yelp-"),
             (csv_folder + "8", 7, "-20newsgroupsReduced-")]  # , (csv_folder+"4",20,"-20newsgroups-")]#,(csv_folder+"1",10,"-reuters-")]#,(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]
# data_sets = [(csv_folder+"2"+"\\train.csv",csv_folder+"2"+"\\test.csv",14)]


strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = remove_stop_words = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params, remove_stop_words]
#preproces_variations = [[False,True,False,True,True]]
#preproces_variations = [[False,False,False,False], [True,True,False,True], [True,False,True,False]]  # [[False,False,False,False]]#[[True,True,True,True],[False,False,False,False],[True,False,True,False],[True,False,True,True]]
preproces_variations = []
create_variations(0, [], preproces_all_vals, preproces_variations)

lda_kappa = [0.51]
lda_tau = [2.0]
lda_minimum_probability = [0.0, 0.01, 0.1]
lda_passes = [50]
lda_iterations = [50]
lda_all_vals = [lda_kappa, lda_tau, lda_passes, lda_iterations]
# lda_variations = []
# create_variations(0,[],lda_all_vals,lda_variations)

lsa_one_pass = [False]
lsa_power_iter = [2]
lsa_use_tfidf = [True]
lsa_topic_nums = [data_sets[0][1]]
lsa_extra_samples = [100, 200]
lsa_decay = [0.5, 1.0, 2.0]
lsa_all_vals = [lsa_one_pass, lsa_power_iter, lsa_use_tfidf, lsa_topic_nums]
# lsa_variations = []
# create_variations(0,[],lsa_all_vals,lsa_variations)

hdp_variations = []
num_of_tests = 1

test_model = {ModelType.LDA: False,
              ModelType.LSA: False,
              ModelType.LDA_Sklearn: False,
              ModelType.NB: False,
              ModelType.SVM: True
              }
is_stable = {ModelType.LDA: False,
              ModelType.LSA: True,
              ModelType.LDA_Sklearn: False,
              ModelType.NB: True
              }
start_time = get_time_in_millis()

models_for_test = [ModelType.LDA, ModelType.LSA, ModelType.NB, ModelType.LDA_Sklearn]

tester = GeneralTester(log_writer, start_time)
datasets_helper = Dataset_Helper(preprocess=False)
#array to iterate should contain valid indexes (ranging from 0 to length of data_sets) of datasets that are present in list data_sets
for i in [6]:#range(len(data_sets)):
    topic_names = TextPreprocessor.load_csv([data_sets[i][0] + "\\topic-names.csv"])
    tester.set_new_dataset(data_sets[i][1], topic_names)
    statistics_to_merge = []
    models_params = {
        ModelType.LDA: {
            "topic_count": data_sets[i][1],
            "topic_word_count": 15,
            "kappa": 0.51,
            "tau": 2.0,
            "passes": 25,
            "iterations": 25
        },
        ModelType.LSA: {
            "topic_count": data_sets[i][1],
            "topic_word_count": 15,
            "one_pass": False,
            "power_iter": 2,
            "use_tfidf": True
        },
        ModelType.LDA_Sklearn: {
            "topic_count": data_sets[i][1],
            "passes": 25,
            "iterations": 25
        },
        ModelType.NB: {

        }
    }
    log_writer.write_model_params("\\results\\results{}{}\\model-settings".format(data_sets[i][2],start_time),models_params)
    for preprocess_index, preproces_settings in enumerate(preproces_variations):
        seed = 5
        settings = {'strip_nums': preproces_settings[0],
                    'use_stemmer': preproces_settings[1],
                    'use_lemmatizer': preproces_settings[2],
                    'strip_short': preproces_settings[3],
                    'remove_stop_words': preproces_settings[4],
                    'use_alternative': False
                    }
        log_writer.add_log(
            "Initializing text preprocessor with strip_nums: {}, use_stemmer: {}, use_lemmatizer {}, strip_short: {}, remove_stop_words: {}.".format(
                preproces_settings[0], preproces_settings[1], preproces_settings[2], preproces_settings[3], preproces_settings[4]))
        preprocessor = TextPreprocessor(settings)

        log_writer.add_log("Starting preprocessing texts of {} for training".format(data_sets[i][0]))
        texts_for_train = preprocessor.load_and_prep_csv([data_sets[i][0] + "\\train.csv"], "eng", True, 1, ';')
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for testing".format(data_sets[i][0]))
        texts_for_testing = preprocessor.load_and_prep_csv([data_sets[i][0] + "\\test.csv"], "eng", True, 1, ';')
        log_writer.add_log("Preprocessing finished")

        # Lda(data_sets[i][1], 15, kappa=lda_kappa[0], tau=lda_tau[0], passes=lda_passes[0], iterations=lda_iterations[0])]
        # Lsa(data_sets[i][1], 15, one_pass=lsa_one_pass[0],power_iter=lsa_power_iter[0], use_tfidf=lsa_use_tfidf[0])]
        # LdaSklearn(data_sets[i][1], passes=lda_passes[0],iterations=lda_iterations[0]),

        statistics = []
        preprocess_style = "{} No nums: {}, Stemmer: {}, Lemmatize {}, No short: {}, Rm stopwords: {}".format(preprocess_index,
                                                                                             preproces_settings[0],
                                                                                             preproces_settings[1],
                                                                                             preproces_settings[2],
                                                                                             preproces_settings[3],
                                                                                             preproces_settings[4])
        statistics.append([preprocess_style])
        tester.set_new_preprocess_docs(texts_for_train, texts_for_testing, preprocess_style)
        test_params = {"preprocess_index": preprocess_index, "dataset_name": data_sets[i][2]}
        for m_index, model in enumerate(models_for_test):
            if test_model[model]:
                # For every preprocesing add line that descripbes methods used
                """accuracies = []
                statistics.append([])
                statistics.append([model.name])
                statistics.append([x for x in range(num_of_tests)])
                statistics[len(statistics) - 1].append("Average")
                statistics.append([])  # TODO dont forget to add test params"""
                tester.do_test(model, num_of_tests, statistics, models_params[model], test_params, is_stable[model])
        tester.output_model_comparison(data_sets[i][2])
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
    log_writer.write_2D_list("\\results\\results-stats\\stats{}{}".format(data_sets[i][2], start_time), output_lda_csv)
    tester.output_preprocess_comparison(data_sets[i][2])

log_writer.end_logging()
