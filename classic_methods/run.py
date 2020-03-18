import os
import sys
import time
from dataset_loader.dataset_helper import Dataset_Helper, DatasetType
from classic_methods.tests.ModelType import ModelType
from classic_methods.tests.general_tester import GeneralTester
from results_saver import LogWriter

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

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


log_writer = LogWriter(result_desc='Classic')

strip_nums_params = use_stemmer_params = use_lemmatizer_params = strip_short_params = remove_stop_words = [True, False]
preproces_all_vals = [strip_nums_params, use_stemmer_params, use_lemmatizer_params, strip_short_params, remove_stop_words]
preproces_variations = [[False,False,True,True,True]]

hdp_variations = []
num_of_tests = 1

test_model = {ModelType.LDA: False,
              ModelType.LSA: False,
              ModelType.LDA_Sklearn: False,
              ModelType.NB: True,
              ModelType.SVM: False,
              ModelType.DT: False,
              ModelType.RF: False
              }
is_stable = {ModelType.LDA: False,
              ModelType.LSA: True,
              ModelType.LDA_Sklearn: False,
              ModelType.NB: True,
             ModelType.SVM: True,
              ModelType.DT: True,
              ModelType.RF: True
              }
start_time = get_time_in_millis()

models_for_test = test_model.keys()#[ModelType.LDA, ModelType.LSA, ModelType.NB, ModelType.LDA_Sklearn, ModelType.SVM, ModelType.RF, ModelType.DT]

tester = GeneralTester(log_writer, start_time)
datasets_helper = Dataset_Helper(preprocess=True)
datasets_helper.set_wanted_datasets([3])
#array to iterate should contain valid indexes (ranging from 0 to length of data_sets) of datasets that are present in list data_sets
while datasets_helper.next_dataset():#range(len(data_sets)):
    topic_names = [(index, item) for index, item in enumerate(datasets_helper.get_dataset_topic_names())]#TextPreprocessor.load_csv([datasets_helper.get_dataset_folder_path() + "\\topic-names.csv"])
    tester.set_new_dataset(datasets_helper.get_num_of_topics(), topic_names)
    statistics_to_merge = []
    models_params = {
        ModelType.LDA: {
            "topic_count": datasets_helper.get_num_of_topics(),
            "topic_word_count": 15,
            "kappa": 0.51,
            "tau": 2.0,
            "passes": 25,
            "iterations": 25
        },
        ModelType.LSA: {
            "topic_count": datasets_helper.get_num_of_topics(),
            "topic_word_count": 15,
            "one_pass": False,
            "power_iter": 2,
            "use_tfidf": True
        },
        ModelType.LDA_Sklearn: {
            "topic_count": datasets_helper.get_num_of_topics(),
            "passes": 25,
            "iterations": 25
        },
        ModelType.NB: {
            'alpha':0.1
        },
        ModelType.SVM:{
            'c':100,
            'kernel':'rbf',
            'degree':3,
            'gamma':1
        },
        ModelType.RF:{
            'n_estimators':20,
            'max_features':10000
        },
        ModelType.DT:{
            'max_features':10000
        }
    }
    for key,value in test_model.items():
        if not value:
            models_params.pop(key)
    log_writer.write_model_params("\\results{}{}\\model-settings".format(datasets_helper.get_dataset_name(),start_time),models_params)
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
        #preprocessor = TextPreprocessor(settings)

        log_writer.add_log("Starting preprocessing texts of {} for training".format(datasets_helper.get_dataset_name()))
        texts_for_train = datasets_helper.get_dataset(DatasetType.TRAIN)#preprocessor.load_and_prep_csv([datasets_helper.get_train_file_path()], "cz", True, 1, ';')
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for testing".format(datasets_helper.get_dataset_name()))
        texts_for_testing = datasets_helper.get_dataset(DatasetType.TEST)#preprocessor.load_and_prep_csv([datasets_helper.get_test_file_path()], "cz", True, 1, ';')
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
        test_params = {"preprocess_index": preprocess_index, "dataset_name": datasets_helper.get_dataset_name(), 'dataset_helper': datasets_helper}
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
        tester.output_model_comparison(datasets_helper.get_dataset_name())
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
    log_writer.write_2D_list("\\results-stats\\stats{}{}".format(datasets_helper.get_dataset_name(), start_time), output_lda_csv)
    tester.output_preprocess_comparison(datasets_helper.get_dataset_name())

log_writer.end_logging()
