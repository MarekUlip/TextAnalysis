import operator

class LModelTester:
    """
        class for testing LDA and LSA models from gensim and sklearn implementation of LDA
    """
    def __init__(self, training_docs, testing_docs, num_of_topics, log_writer, topic_names):
        """

        :param training_docs: docs in form of list containing tuples (topic_index, doc) used for training
        :param testing_docs: docs in form of list containing tuples (topic_index, doc) used in test
        :param num_of_topics: number of topics occuring in provided texts
        :param log_writer: instance of log writer to write output
        :param topic_names: list of tuples in form of (topic id, topic name)
        """
        self.testing_docs = testing_docs
        self.training_docs = training_docs
        self.num_of_topics = num_of_topics
        self.log_writer = log_writer
        self.representants = {}
        self.topic_indexes = {}
        self.topics_of_index = {}
        self.topic_distributions = []
        self.topic_numbers = []
        self.topic_names = {}
        #count_of_filled_reps = 0
        self.prep_train_docs_for_assesment()
        self.count_topic_dist()
        self.confusion_matrix = [[0 for y in range(num_of_topics)] for x in range(num_of_topics)]
        self.confusion_matrix_true = [[0 for y in range(num_of_topics)] for x in range(num_of_topics)]
        self.create_topic_names_dict(topic_names)

    def create_topic_names_dict(self, topic_names_list):
        """
        Creates dictionary where key is topic number and value is its name
        :param topic_names_list: list of tuples containing topic number[0] and its name[1]
        """
        for item in topic_names_list:
            self.topic_names[int(item[0])] = item[1]

    def prep_train_docs_for_assesment(self, training_docs=None):
        """
        Sorts training docs into groups by their respective topics. Used for "gueesing: which model topic index belongs
        to which real topic id.
        :param training_docs: if not provided training docs from initialization will be used otherwise no action
        will be performed
        """
        if training_docs is not None:
            self.training_docs = training_docs
        for i in range(len(self.training_docs)):
            if self.training_docs[i][0] not in self.representants:
                self.representants[self.training_docs[i][0]] = [self.training_docs[i]]
            else:
                self.representants[self.training_docs[i][0]].append(self.training_docs[i])

    def count_topic_dist(self):
        """
        Counts relative topic distribution. This can be used with LDA alpha parameter.
        counted value is saved in topic_distribution property
        """
        if len(self.representants) == 0:
            self.log_writer("Representants not set. Cannot make topic dist.")
            return
        for key, value in self.representants.items():
            self.topic_distributions.append(len(value)/len(self.training_docs))
            self.topic_numbers.append(key)

    def add_descriptions_to_confusion_matrix(self):
        """
        Adds topic names into confusion matrix
        """
        topic_names = []
        for topic_num in self.topic_numbers:
            topic_names.append(self.topic_names[topic_num])
        for index, row in enumerate(self.confusion_matrix):
            row.insert(0,topic_names[index])
        topic_names_for_matrix = topic_names.copy()
        topic_names_for_matrix.insert(0,"")
        self.confusion_matrix.insert(0,topic_names_for_matrix)
        self.confusion_matrix_true.insert(0,topic_names_for_matrix)


    def test_model(self, model, test_name):
        """
        Tests provided instance of a model and outputs results using provided test_name
        :param model: model to be tested
        :param test_name: name which will be used for output
        :return: accuracy in range (0 - 1)
        """
        statistics = []
        stats = []
        for item in model.get_topics():
            statistics.append(item)
        statistics.append(["Article topic", "Model topic index"])
        self.connect_topic_id_to_topics(model)

        for article in self.testing_docs:
            analysis_res = model.analyse_text(article[1])
            if len(analysis_res) == 0:
                print("nothing found")
                continue
            res = max(analysis_res, key=lambda item: item[1])
            statistics.append([article[0], res[0]])
            if res[0] not in self.topics_of_index:
                self.topics_of_index[res[0]] = [article[0]]
                self.topic_indexes[article[0]] = res[0]
                print("continuing")
                continue

            stats.append(1 if article[0] in self.topics_of_index[res[0]] else 0)
            topic_number_index = self.topic_numbers.index(article[0])

            if article[0] in self.topics_of_index[res[0]]:
                guessed_topic_number_index = self.topic_numbers.index(article[0])
            else:
                guessed_topic_number_index = self.topic_numbers.index(self.topics_of_index[res[0]][0])
            self.confusion_matrix[guessed_topic_number_index][topic_number_index] += 1
            self.confusion_matrix_true[res[0]][topic_number_index] += 1
            #self.log_writer.add_log("Article with topic {} was assigned {} with {} certainty.".format(article[0], "correctly" if res[0] == self.topic_positions[article[0]] else "wrong", res[1]))

        self.log_writer.write_2D_list(test_name, statistics)
        self.add_descriptions_to_confusion_matrix()
        self.log_writer.write_2D_list(test_name+"-confusion-matrix", self.confusion_matrix)
        self.log_writer.write_2D_list(test_name+"-confusion-matrix-true", self.confusion_matrix_true)
        return sum(stats)/len(stats)

    def connect_topic_id_to_topics(self, model):
        """
            Connects model topic indexes to topic ids from dataset. Every model index is assigned and all topic ids are used. Note that this method should be used
            on balanced datasets because it would prefer smaller topic groups due to higher precentage confidence.
            :param model: model containing topics to connect
        """
        confidence = []
        for key, value in self.representants.items():
            connection_results = {}
            for article in value:
                try:
                    # get most possible index
                    topic_index = max(model.analyse_text(article[1]), key=lambda item: item[1])[0]
                except ValueError:
                    print("No topic index returned continuing")  # TODO replace with if
                    continue
                # add most possible index for this article to counter
                if topic_index not in connection_results:
                    connection_results[topic_index] = 1
                else:
                    connection_results[topic_index] += 1
            # find index that occured mostly
            print(connection_results)
            for tp_num, val in connection_results.items():
                confidence.append([key, tp_num, val / len(value)])
        confidence = sorted(confidence, key=operator.itemgetter(2), reverse=True)
        associated_indexes = []
        associated_topics = []
        for conf in confidence:
            if conf[1] in associated_indexes or conf[0] in associated_topics:
                continue
            associated_indexes.append(conf[1])
            associated_topics.append(conf[0])
            self.log_writer.add_log(
                'Connecting topic {} to model index {} based on highest unused confidence of {}'.format(conf[0],
                                                                                                        conf[1],
                                                                                                        conf[2]))
            self.topic_indexes[conf[0]] = conf[1]

        for key, value in self.topic_indexes.items():
            self.topics_of_index[value] = [key]

    def connect_topic_id_to_topics_old(self, model):
        """
        Connects topic indexes from model to topic ids from dataset. Note that some ids might not get connected due various reasons.
        :param model: model containing topics to connect
        """
        #t = model.get_topics()
        for key, value in self.representants.items():
            connection_results = {}
            for article in value:
                try:
                    #get most possible index
                    topic_index = max(model.analyse_text(article[1]), key=lambda item: item[1])[0]
                except ValueError:
                    print("No topic index returned continuing")#TODO replace with if
                    continue
                #add most possible index for this article to counter
                if topic_index not in connection_results:
                    connection_results[topic_index] = 1
                else:
                    connection_results[topic_index] += 1
            #find index that occured mostly
            best_candidates = max(connection_results.items(), key=operator.itemgetter(1))
            print(best_candidates)
            self.log_writer.add_log("Best candidate with index {} is connected to topic {} with {}% accuracy".format(best_candidates[0], key, (connection_results[best_candidates[0]]/len(value))*100))
            #create connection between topic id and model topic index
            self.topic_indexes[key] = best_candidates[0]
            #creat connection in opposite direction if there already is some connection add found index to that connection (some model topic index can represent more than one real topic)
            if best_candidates[0] not in self.topics_of_index:
                self.topics_of_index[best_candidates[0]] = [key]
            else:
                self.topics_of_index[best_candidates[0]].append(key)

        self.log_writer.add_log("Out of {} real topics only {} were learned".format(len(self.representants), len(self.topics_of_index)))



