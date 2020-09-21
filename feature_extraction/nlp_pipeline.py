import pandas as pd
import os
from talon.signature.bruteforce import extract_signature
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class NlpPipeline:

    def __init__(self):
        self.content = None
        self.paragraphs = None
        self.tokens_all = []

    def process_text(self, content):

        self.paragraphs = None
        self.tokens_all = []

        self.content = content

        # To lower-case
        self.content = self.convert_to_lower(self.content)

        # Split to paragraphs
        self.paragraphs = self.content.split('\n')

        # Trim white space
        self.paragraphs = map(str.strip, self.paragraphs)

        # Remove junk lines
        self.remove_junk_lines()

        # Split into words
        self.split_into_words()

        # Remove stop words
        self.remove_stop_words()

        # Perform stemming
        # self.perform_stemming()

        # Perform lemmatization
        self.perform_lemmatization()


        #self.get_signature(content)
        return self.tokens_all
        #return "".join(self.tokens_all)


    def convert_to_lower(self, content):
        return content.lower()


    def get_signature(self, content):
        text, signature = extract_signature(content)
        a = 1


    def remove_junk_lines(self):

        def evaluate_paragraph(paragraph):
            if '-original message-' in paragraph or '********' in paragraph:
                return False
            if '@' in paragraph and ('.gov' in paragraph or '.edu' in paragraph or '.com' in paragraph):
                return False
            if 'to:' in paragraph[:10] or 'from:' in paragraph[:10] or 'cc:' in paragraph[:10] or 'bcc:' in \
                paragraph[:10] or 'date:' in paragraph[:10] or 'sent:' in paragraph[:10]:
                return False
            if len(paragraph) < 2:
                return False
            return True

        self.paragraphs = [paragraph for paragraph in self.paragraphs if evaluate_paragraph(paragraph)]
        a = 1


    def split_into_words(self):

        def evaluate_words(token):
            if len(token) < 2:
                return False
            if str.strip(token) in ['re', 'fwd', 'fw']:
                return False
            return True

        tokens_outer = []

        for paragraph in self.paragraphs:
            tokens = word_tokenize(paragraph)
            tokens = [token for token in tokens if evaluate_words(token) ]
            tokens_outer.append(tokens)
            self.tokens_all.extend(tokens)

        a  = 1



    def remove_stop_words(self):
        stop_words = stopwords.words('english')
        self.tokens_all = [token for token in self.tokens_all if token not in stop_words]


    def perform_stemming(self):
        porter = PorterStemmer()
        self.tokens_all = [porter.stem(word) for word in self.tokens_all]
        a = 1

    def perform_lemmatization(self):
        lemmatizer = WordNetLemmatizer()

        tokens_all2 = [lemmatizer.lemmatize(token) for token in self.tokens_all]

        a = 1
