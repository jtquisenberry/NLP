# https://colab.research.google.com/drive/1b8S5lzxst5lRpGvHesu1TEams_ndXqpp#scrollTo=pztmAgaS-DIt
# https://huggingface.co/transformers/v2.4.0/examples.html#named-entity-recognition

import os

# pip install transformers
import transformers

PREPROCESS1 = True
PREPROCESS2 = True

# ORIGINAL CORPUS
# These are German-language files.
# They are no longer available from the sites listed below.
'''
!curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-train.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > train.txt.tmp
!curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-dev.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > dev.txt.tmp 
!curl -L 'https://sites.google.com/site/germeval2014ner/data/NER-de-test.tsv?attredirects=0&d=1' \
| grep -v "^#" | cut -f 2,3 | tr '\t' ' ' > test.txt.tmp
'''

# CoNLL-2003 CORPUS
# https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003
# https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data

# STEP 1: Preprocess with command line
# 1. Get lines that do not contain "#".
# 2. Treat space as delimiter.
# 3. Remove fields 2 and 3, leaving
#   a. The token; and
#   b. The NER annotation
# 4. Replace tab with space

# Use msys2 on Windows 10
# grep -v "^#" test_orig.txt| cut -d " " -f 1,4 | tr '\t' ' ' > test_orig.txt.tmp
# grep -v "^#" train_orig.txt| cut -d " " -f 1,4 | tr '\t' ' ' > train_orig.txt.tmp
# grep -v "^#" valid_orig.txt| cut -d " " -f 1,4 | tr '\t' ' ' > valid_orig.txt.tmp

# STEP 2: Manually delete the header lines like this:
# -DOCSTART- O
#

if PREPROCESS1:
    # An alternative to grep, cut, tr
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path_in = os.path.abspath('../datasets/conll2003/en')
    from format_corpus import FormatCorpus
    fc = FormatCorpus(corpus_name='conll2003', path_in=path_in, file_in='test_orig.txt', path_out=path_in, file_out='test_orig.txt.tmp')
    fc.format_corpus()
    fc = FormatCorpus(corpus_name='conll2003', path_in=path_in, file_in='train_orig.txt', path_out=path_in, file_out='train_orig.txt.tmp')
    fc.format_corpus()
    fc = FormatCorpus(corpus_name='conll2003', path_in=path_in, file_in='valid_orig.txt', path_out=path_in, file_out='valid_orig.txt.tmp')
    fc.format_corpus()
    print('Done preprocessing CONLL2003 - Part 1')


if PREPROCESS2:
    from preprocess_jq import ner_preprocess
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.abspath('../datasets/conll2003/en')
    files = [('test_orig.txt.tmp', 'test.txt'), ('train_orig.txt.tmp', 'train.txt'), ('valid_orig.txt.tmp', 'valid.txt')]
    for file_in, file_out in files:
        # full_path = os.path.join(path, file)
        np = ner_preprocess(path=path, file_in=file_in, file_out=file_out,
                            model_name_or_path='bert-base-multilingual-cased', max_len=128)
        np.do_preprocess()

    print('Done preprocessing CONLL2003 - Part 2')


# Get distinct labels
# Using Msys2 terminal
# cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
from get_labels import GetLabels
os.chdir(os.path.dirname(os.path.abspath(__file__)))
path = os.path.abspath('../datasets/conll2003/en')
files = ['test.txt', 'train.txt', 'valid.txt']
gl = GetLabels(path=path, files=files)
labels = gl.get_labels()
a = 1
