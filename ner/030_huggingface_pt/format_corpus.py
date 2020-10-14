import os
import sys
import re


class FormatCorpus():
    def __init__(self, corpus_name, path_in, file_in, path_out, file_out):
        self.corpus_name = corpus_name
        self.path_in = path_in
        self.file_in = file_in
        self.full_path_in = os.path.join(path_in, file_in)
        self.path_out = path_out
        self.file_out = file_out
        self.full_path_out = os.path.join(path_out, file_out)
        if corpus_name == 'conll2003':
            self.delimiter = ' '

    def format_corpus(self):
        with open(self.full_path_in, "rt", encoding="utf-8") as file_in:
            with open(self.full_path_out, "wt", encoding="utf-8", newline='\n') as file_out:
                for line_number, line in enumerate(file_in):
                    if line_number == 2940:
                        xxxxxxx = 0

                    if line_number in [0, 1]:
                        continue
                    elif re.match(r'[#]', line):
                        continue
                    elif not line.strip():
                        file_out.write('' + '\n')
                    else:
                        # Get column #0 and column #3
                        columns = re.split(self.delimiter, line)
                        del columns[1]
                        del columns[1]
                        if len(columns) != 2:
                            continue
                        line_out = self.delimiter.join(columns)
                        file_out.write(line_out)

        print('Done writing {0}'.format(self.full_path_out))
