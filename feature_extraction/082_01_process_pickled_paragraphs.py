import pandas as pd
import os
from pickle_handler import PickleHandler


if __name__ == '__main__':
    in_directory = r'E:\Corpora\PII_Jeb_20190507'
    in_pickled_df = os.path.join(in_directory, 'pickled_paragraphs.pkl')
    out_pickled_df = os.path.join(in_directory, 'pickled_paragraphs_transformed_001.pkl')
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)
    pickle_handler.perform_nlp()
    #pickle_handler.set_category_ids()
    pickle_handler.save_pickle()
    print('DONE')
