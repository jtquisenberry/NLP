# It is necessary to import findspark before pyspark.
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer

import pandas as pd
import os
from pickle_handler import PickleHandler
from spark_extractor import SparkExtractor

if __name__ == '__main__':

    in_directory = r'E:\Corpora\PII_Directory_20190507'
    in_pickled_df = os.path.join(in_directory, 'df_pickle_transformed_001.pkl')
    out_pickled_df =  None
    pickle_handler = PickleHandler(in_pickled_df, out_pickled_df)

    spark_extractor = SparkExtractor(pickle_handler.df)
    spark_extractor.start_extraction()







