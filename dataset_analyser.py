from helper_functions import Dataset_Helper
from collections import Counter

helper = Dataset_Helper(False)
helper.set_wanted_datasets([3])
while helper.next_dataset():
    labels = helper.get_labels(helper.get_train_file_path())
    labels_test = helper.get_labels(helper.get_test_file_path())
    print(Counter(labels))
    print(Counter(labels_test))