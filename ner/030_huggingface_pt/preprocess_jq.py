# wget https://raw.githubusercontent.com/stefan-it/fine-tuned-berts-seq/master/scripts/preprocess.py

import sys
import os
import ntpath

from transformers import AutoTokenizer

class ner_preprocess():

    def __init__(self, path, file_in, file_out, model_name_or_path, max_len):
        self.path = path
        self.file_in = file_in
        self.file_out = file_out
        self.model_name_or_path = model_name_or_path
        self.max_len = max_len

    def do_preprocess(self):
        subword_len_counter = 0

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.max_len -= tokenizer.num_special_tokens_to_add()

        #with open(os.path.dirname(self.dataset) + os.path.l) as file_out:
        with open(os.path.join(self.path, self.file_in), 'rt') as file_in:
            with open(os.path.join(self.path, self.file_out), 'wt', encoding='utf-8') as file_out:
                for line in file_in:
                    line = line.rstrip()

                    if not line:
                        #print(line)
                        file_out.write(line + '\n')
                        subword_len_counter = 0
                        continue

                    token = line.split()[0]

                    current_subwords_len = len(tokenizer.tokenize(token))

                    # Token contains strange control characters like \x96 or \x95
                    # Just filter out the complete line
                    if current_subwords_len == 0:
                        continue

                    if (subword_len_counter + current_subwords_len) > self.max_len:
                        #print("")
                        file_out.write('' + '\n')
                        #print(line)
                        file_out.write(line + '\n')
                        subword_len_counter = current_subwords_len
                        continue

                    subword_len_counter += current_subwords_len

                    # print(line)
                    file_out.write(line + '\n')