import errno
import json
import os
import sys


class Settings:
    def __init__(self):
        self.main_frame = None
        self.settings_file = os.path.join(sys.path[0], 'settings.json')
        self.settings = {
            'file_folder': "",
            'strip_nums': False,
            'use_stemmer': False,
            'use_lemmatizer': True,
            'strip_short': False,
            'use_alternative': False,
            'remove_stop_words': True
        }
        self.load_settings()

    def load_settings(self):
        if os.path.isfile(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding="utf8") as f:
                    self.settings = json.loads(f.read())

            except IOError as exc:
                if exc.errno != errno.EISDIR:
                    raise
