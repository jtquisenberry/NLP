import os
import sys
from urllib.request import urlopen
from zipfile import ZipFile

CORPUS = 'sttng6'

if not os.path.exists(r'./data'):
    os.mkdir(r'./data')

a = 1


def get_zip(url, path, zip_name):
    zipresp = urlopen(url)
    if not os.path.exists(r'./zips'):
        os.mkdir(r'./zips')
    if not os.path.exists(os.path.join(r'./zips', os.path.basename(path))):
        os.mkdir(os.path.join(r'./zips', os.path.basename(path)))
    zip_path = os.path.join(r'./zips', os.path.basename(path), zip_name)
    tempzip = open(zip_path, 'wb')
    tempzip.write(zipresp.read())
    tempzip.close()
    zf = ZipFile(zip_path)
    zf.extractall(path=path)
    zf.close()


class Corpus():
    def __init__(self, name='', url='', description='', subdirectory='', first_file=''):
        self.name = name
        self.url = url
        self.description = description
        self.subdirectory = subdirectory
        self.first_file = first_file

corpus_dict = {}
corpus_dict['sttng6'] = \
    Corpus(name='Star Trek: The Next Generation',
           url=r'https://github.com/jtquisenberry/NLP/raw/master/corpora/sttng6.zip',
           description="All scripts of Star Trek: The Next Generation, preprocessed by ' + '\n' +"
           "(1) concatenating scripts, (2) conversion on lowercase.",
           subdirectory='sttng6',
           first_file='sttng6.txt')
corpus_dict['spampot_hpsl_concat'] = \
    Corpus(name='Spampot',
           url=r'https://github.com/jtquisenberry/NLP/raw/master/corpora/spampot_hpsl_concat.zip',
           description='Email spam collection',
           subdirectory='spampot_hpsl_concat',
           first_file='spampot_hpsl_concat.txt')
corpus_dict['princess_bride_orig'] =  \
    Corpus(name='The Princess Bride Script Original',
           url=r'https://github.com/jtquisenberry/NLP/raw/master/corpora/princess_bride_orig.zip',
           description='The Princess Bride Script without text preprocessing',
           subdirectory='princess_bride_orig',
           first_file='princess_bride.txt')

corpus = corpus_dict[CORPUS]
download_directory = os.path.join(r'./data', corpus.subdirectory)
zip_name = os.path.basename(corpus.url)

if os.path.exists(download_directory):
    if os.path.exists(os.path.join(download_directory, corpus.first_file)):
        print("CORPUS EXISTS")
    else:
        get_zip(corpus.url, download_directory, zip_name)
else:
    os.mkdir(download_directory)
    get_zip(corpus.url, download_directory, zip_name)

print('DONE')

