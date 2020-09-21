import os
from pickle_handler import PickleHandler
from sklearn_wrapper_extractor import SklearnWrapperExtractor
from dataframe_manipulator import DataframeManipulator




if __name__ == '__main__':
    in_directory = r'E:\Corpora\PII_Directory_20190507'
    in_pickled_df = os.path.join(in_directory, 'pickled_lines_transformed_001.pkl')
    out_pickled_df = None
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)

    dfm = DataframeManipulator(pickle_handler.df)
    labels = ['PII|Health|condition_treatment', 'PII|Health|health_payment',
              'PII|Health|applications_and_claims', 'PII|Employment|performance_review']
    pickle_handler.df = dfm.equalize_rows_by_label(labels)

    #Plot

    df_plot = pickle_handler.df.groupby(['minimum_label'], as_index=False).count()[['minimum_label', 'tag']]
    df_plot['label'] = df_plot.apply(lambda x: x['minimum_label'].split('|')[2], axis=1)
    df_plot.columns = ['minimum_label', 'count', 'label']
    plt = df_plot.plot.barh(x='label', y='count')




    sklearn_wrapper_extractor = SklearnWrapperExtractor(pickle_handler.df)
    sklearn_wrapper_extractor.prepare_data()
    sklearn_wrapper_extractor.vectorize_documents_tokenized()
    #sklearn_wrapper_extractor.get_features_with_chi2()
    sklearn_wrapper_extractor.get_features_with_model()

    sklearn_wrapper_extractor.features_csv = os.path.join(in_directory, 'pickled_lines_features_20190619.csv')
    sklearn_wrapper_extractor.features_pkl = os.path.join(in_directory, 'pickled_lines_features_20190619.pkl')
    sklearn_wrapper_extractor.save_features()
    print('DONE')



