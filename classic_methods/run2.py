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
models_params = {
        ModelType.LDA: {
            "topic_count": None,
            "topic_word_count": 15,
            "kappa": 0.51,
            "tau": 2.0,
            "passes": 25,
            "iterations": 25
        },
        ModelType.LSA: {
            "topic_count": None,
            "topic_word_count": 15,
            "one_pass": False,
            "power_iter": 2,
            "use_tfidf": True
        },
        ModelType.LDA_Sklearn: {
            "topic_count": None,
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
            'max_features':15000
        },
        ModelType.DT:{
            'max_features':15000
        }
    }
start_time = get_time_in_millis()

models_for_test = test_model.keys()
for model in models_for_test:
    if not test_model[model]:
        continue
    log_writer = LogWriter(log_file_desc='_{}'.format(model.name),result_desc='Classic')
    tester = GeneralTester(log_writer, start_time)
    datasets_helper = Dataset_Helper(preprocess=True)
    datasets_helper.set_wanted_datasets([0,2,3])
    while datasets_helper.next_dataset():
        if 'topic_count' in models_params[model]:
            models_params[model]['topic_count'] = datasets_helper.get_num_of_topics()
        topic_names = [(index, item) for index, item in enumerate(datasets_helper.get_dataset_topic_names())]
        tester.set_new_dataset(datasets_helper.get_num_of_topics(), topic_names)
        statistics_to_merge = []

        """for key,value in test_model.items():
            if not value:
                models_params.pop(key)"""
        log_writer.write_model_params("model-settings",models_params)
        seed = 5

        log_writer.add_log("Starting preprocessing texts of {} for training".format(datasets_helper.get_dataset_name()))
        texts_for_train = datasets_helper.get_dataset(DatasetType.TRAIN)
        log_writer.add_log("Preprocessing finished")

        log_writer.add_log("Starting preprocessing texts of {} for testing".format(datasets_helper.get_dataset_name()))
        texts_for_testing = datasets_helper.get_dataset(DatasetType.TEST)
        log_writer.add_log("Preprocessing finished")

        statistics = []
        tester.set_new_preprocess_docs(texts_for_train, texts_for_testing)
        test_params = {"dataset_name": datasets_helper.get_dataset_name(), 'dataset_helper': datasets_helper}
        tester.do_test(model, num_of_tests, statistics, models_params[model], test_params, is_stable[model])
        #tester.output_model_comparison(datasets_helper.get_dataset_name())
        statistics.append([])
        statistics_to_merge.append(statistics)


        output_lda_csv = []
        for item in statistics_to_merge:
            for statistic in item:
                output_lda_csv.append(statistic)
        log_writer.write_2D_list("stats".format(datasets_helper.get_dataset_name(), start_time), output_lda_csv)
        log_writer.add_log('Done testing {} dataset.'.format(datasets_helper.get_dataset_name()),True)
        #tester.output_preprocess_comparison(datasets_helper.get_dataset_name())

    log_writer.end_logging()
