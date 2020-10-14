import os
import sys
import re

class GetLabels():
    def __init__(self, path, files):
        self.path = path
        self.files = files

    def get_labels(self):
        labels = set()
        for file in self.files:
            full_path = os.path.join(self.path, file)
            with open(full_path, "rt", encoding='utf-8') as file_in:
                for line in file_in:
                    chunks = re.split(' ', line.strip())
                    if len(chunks) == 2:
                        label = chunks[1]
                        labels.add(label)

        labels = list(labels)
        labels = sorted(labels)
        return labels
