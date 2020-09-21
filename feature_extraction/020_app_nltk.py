import os
from pickle_handler import PickleHandler
from nltk_extractor import NltkExtractor


if __name__ == '__main__':

    in_directory = r'E:\Corpora\PII_Directory_20190507'
    in_pickled_df = os.path.join(in_directory, 'df_pickle_transformed_ids_001.pkl')
    out_pickled_df = None
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)

    nltk_extractor = NltkExtractor(pickle_handler.df)
    nltk_extractor.start_extraction()