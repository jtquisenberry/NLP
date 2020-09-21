import os
from pickle_handler import PickleHandler
from sklearn_wrapper_extractor import SklearnWrapperExtractor

if __name__ == '__main__':
    in_directory = r'E:\Corpora\PII_Jeb_20190507'
    in_pickled_df = os.path.join(in_directory, 'df_pickle_transformed_ids_001.pkl')
    out_pickled_df = None
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)

    sklearn_wrapper_extractor = SklearnWrapperExtractor(pickle_handler.df)
    sklearn_wrapper_extractor.prepare_data()
    sklearn_wrapper_extractor.vectorize_documents_tokenized()
    #sklearn_wrapper_extractor.get_features_with_chi2()
    sklearn_wrapper_extractor.get_features_with_model()
    #sklearn_wrapper_extractor.save_features()



