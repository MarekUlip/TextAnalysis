from results_saver import LogWriter
from .ModelType import ModelType
from .lda_lsa_model_tester import LModelTester
from .naive_bayes_model_tester import NBModelTester
from .lsa_tester import LSAModelTester
from .svm_model_tester import SVMModelTester
from ..methods.Lda import Lda
from ..methods.Lsa import Lsa
from ..methods.Lda_sklearn import LdaSklearn
from ..methods.Naive_bayes import NaiveBayes
from ..methods.SVM import SupportVectorMachines
from ..methods.Decision_tree import DecisionTree
from ..methods.Random_forest import RandomForest
from results_saver import plot_confusion_matrix
import numpy as np


class GeneralTester:
    def __init__(self, log_writer, start_time):
        self.testing_docs = None
        self.training_docs = None
        self.num_of_topics = None
        self.log_writer:LogWriter = log_writer
        self.start_time = start_time
        self.topic_names = None
        self.model_results = []
        self.preprocess_style = ""
        self.preproces_results = {}
        self.num_of_tests = 1

    def set_new_dataset(self, num_of_topics, topic_names):
        """
        Notifies that new dataset has been set and updates num_of_topics and topic_names atribtes
        :param num_of_topics:
        :param topic_names:
        """
        self.num_of_topics = num_of_topics
        self.topic_names = topic_names

    def set_new_preprocess_docs(self, training_docs, testing_docs):
        """
        Sets new dataset documents to be tested
        :param training_docs:
        :param testing_docs:
        :param preprocess_style:
        """
        self.testing_docs = testing_docs
        self.training_docs = training_docs

    def do_test(self, model_type, num_of_tests, statistics, params, test_params, stable=False):
        """
        Do test on provided model type. Also sets things up before the test.
        :param model_type: ModelType enum for model that should be tested
        :param num_of_tests: number of tests to be performed on this model
        :param statistics: list to which accuracy and other information will be written
        :param params: Parameters for tested model
        :param test_params: Parameters for test
        :param stable: Indicates whether algorithm is deterministic. If True only one test will be commited and the rest of results will be padded with same result (for charts comparisons).
        """
        self.num_of_tests = num_of_tests
        accuracies = []
        statistics.append([])
        statistics.append([model_type.name])
        statistics.append([x for x in range(num_of_tests)])
        statistics[len(statistics) - 1].append("Average")
        statistics.append([])
        for i in range(num_of_tests):
            accuracy = self.test_model(model_type,
                                       test_params.get("dataset_name", "none"),
                                       params,test_params)
            accuracies.append(accuracy)
            statistics[len(statistics) - 1].append(accuracy)
            self.log_writer.add_log("Testing {} model done with {}% accuracy".format(model_type, accuracy * 100))
            self.log_writer.add_log("\n\n")
            if stable:
                for j in range(num_of_tests - 1):
                    accuracies.append(accuracy)
                    statistics[len(statistics) - 1].append(accuracy)
                break
        total_accuracy = sum(accuracies) / len(accuracies)
        self.log_writer.add_to_plot(model_type.name, accuracies)
        self.log_writer.draw_plot(model_type.name + " " + test_params.get("dataset_name", "none"),
                                  '{}_model_accuracy'.format(test_params.get("dataset_name", "none")), num_of_tests)
        self.model_results.append((model_type.name, accuracies))
        if model_type in self.preproces_results:
            self.preproces_results[model_type].append((self.preprocess_style, accuracies))
        else:
            self.preproces_results[model_type] = [(self.preprocess_style, accuracies)]
        statistics[len(statistics) - 1].append(total_accuracy)
        self.log_writer.add_log("Total accuracy is: {}".format(total_accuracy))

    def test_model(self, model_type, test_name, params, test_params):
        """
        Runs actual test on a model
        :param model_type:  ModelType enum for model that should be tested
        :param test_name: name that will be used for creating output folder
        :param params: Parameters for tested model
        :return: Accuracy of provided model
        """
        model = None
        tester = None
        if model_type == ModelType.LDA:
            model = Lda(self.num_of_topics, params=params)
        elif model_type == ModelType.LDA_Sklearn:
            model = LdaSklearn(self.num_of_topics, params=params)
        if model is not None:
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs)  # TODO watch out for rewrites
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = LModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer,
                                  self.topic_names)
            

        if model_type == ModelType.LSA:
            model = Lsa(self.num_of_topics, params=params)
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs)  # TODO watch out for rewrites
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = LSAModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer,
                                    self.topic_names)
            

        if model_type == ModelType.NB:
            model = NaiveBayes(params)
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs, self.testing_docs)
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = NBModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer,
                                   self.topic_names)
            

        if model_type == ModelType.SVM or model_type == ModelType.DT or model_type == ModelType.RF:
            if model_type == ModelType.SVM:
                model = SupportVectorMachines(params)
            elif model_type == ModelType.DT:
                model = DecisionTree(params)
            elif model_type == ModelType.RF:
                model = RandomForest(params)
            self.log_writer.add_log("Starting training {} model".format(model_type))
            model.train(self.training_docs)
            self.log_writer.add_log("Starting testing {} model".format(model_type))
            tester = SVMModelTester(self.training_docs, self.testing_docs, self.num_of_topics, self.log_writer,
                                    self.topic_names)
        accuracy = tester.test_model(model,test_name)
        cm:np.ndarray = np.array(tester.confusion_matrix)
        cm = cm[1:,1:]
        cm = cm.transpose()
        cm = cm.astype(np.uint32)
        dataset_helper = test_params.get('dataset_helper',None)
        plot_confusion_matrix(cm,dataset_helper.get_num_of_topics(),dataset_helper.get_dataset_name(),self.log_writer)
        return accuracy
            

    def create_test_name(self, dataset_name, start_time, model_name, preprocess_index, test_num):
        """
        Helper function to create path to a current test folder
        :param dataset_name: name of a tested dataset
        :param start_time: can be any unique number. (if number was already used in past test results will rewrite those past test results)
        :param model_name: name of a tested model
        :param preprocess_index: Index of a preprocess settings
        :param test_num: number of a test (if multiple tests are conducted on a single model)
        :return: path to test folder
        """
        return "\\results\\results{}{}\\{}\\preprocess{}\\test_num{}".format(dataset_name, start_time, model_name,
                                                                             preprocess_index, test_num)



