# It is necessary to import findspark before pyspark.
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer
from pyspark import SparkContext
from pyspark.sql import SQLContext
import datetime

import pandas as pd

class SparkExtractor:

    def __init__(self, pdf):
        self.pdf = pdf

    def start_extraction(self):

        # Create a new pandas dataframe containing only the required columns.
        pdf2 = self.pdf[['minimum_label', 'content_list']]
        pdf2.columns = ['id', 'words']

        # Create Spark session
        spark = SparkSession.builder.appName("CountVectorizerExample").getOrCreate()

        # Convert form pandas DataFrame to spark DataFrame
        # https://stackoverflow.com/questions/37513355/converting-pandas-dataframe-into-spark-dataframe-error
        # sc = SparkContext() # SparkSession already makes a context
        sqlCtx = SQLContext(spark)
        sdf = sqlCtx.createDataFrame(pdf2)

        '''
        sdf = spark.createDataFrame([
            (0, "a b c".split(" ")),
            (1, "a b b c a".split(" ")),
            (1, "a b b e a".split(" ")),
            (1, "a b b c e".split(" ")),
            (1, "a b d c a".split(" ")),
            (1, "e e e e e e e e e e".split(" ")),
            (0, "a f f g a f".split(" "))
        ], ["id", "words"])
        '''

        sdf.show(5)

        # fit a CountVectorizerModel from the corpus.
        cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=1000, minDF=2.0)

        model = cv.fit(sdf)

        print('Min TF', model.getMinTF())
        print('Min DF', model.getMinDF())
        print('Max DF', model.getMaxDF())

        print(model.vocabulary[:10])

        # It appears that the default order of model.vocabulary is the order of significance of the features in descending order. Notice that 'e' appears first and it is most represented in the DataFrame.

        result = model.transform(sdf)

        result.show(5)

        id_feature_counts_dictionary = dict()


        # It appears that indexes in the features vector refer to indices in the `model.vocabulary` list. Note that model.vocabulary[0] is 'e'. Counts of [0] are equal to counts of 'e'.
        row_num = 0
        feature_counts_dictionary = dict()
        for row in result.rdd.collect():


            if row.id not in id_feature_counts_dictionary:
                id_feature_counts_dictionary[row.id] = dict()






            dt_start = datetime.datetime.now()
            # print(row)
            # print(row.features.indices)
            # print(row.features)
            # print(row.features[1])

            x = 1

            for i in range(0, len(row.features.indices)):



                feature_id = row.features.indices[i]

                # feature_name = model.vocabulary[feature_id]
                # print(feature_name)
                feature_count = row.features[int(feature_id)]
                # print(feature_count)
                # print("feature_id {0}, feature_count {1}".format(row.features.indices[i], row.features[i]))

                if feature_id in id_feature_counts_dictionary[row.id]:
                    id_feature_counts_dictionary[row.id][feature_id] += feature_count
                else:
                    id_feature_counts_dictionary[row.id][feature_id] = feature_count





                if feature_id in feature_counts_dictionary:
                    feature_counts_dictionary[feature_id] += feature_count
                else:
                    feature_counts_dictionary[feature_id] = feature_count

                '''
                if feature_name in feature_counts_dictionary:
                    feature_counts_dictionary[feature_name] += feature_count
                else:
                    feature_counts_dictionary[feature_name] = feature_count
                '''


                a = 1

            if row_num % 100 == 0:

                print(datetime.datetime.now() - dt_start)
                print(row_num)

            row_num += 1

            u = 1

        print('Most Informative Features')

        '''
        for feature in model.vocabulary:
            print(feature, feature_counts_dictionary[feature])
        '''

        '''
        for item in feature_counts_dictionary.items():

            print(item[0], model.vocabulary[item[0]], int(item[1]))
        '''

        out_pdf = pd.DataFrame(columns=['tag', 'feature_id', 'feature_name', 'feature_count'])

        for outer_item in id_feature_counts_dictionary.items():
            tag = outer_item[0]
            inner_dict = outer_item[1]

            for inner_item in inner_dict.items():
                item_id = inner_item[0]
                item_name = model.vocabulary[item_id]
                item_count = inner_item[1]
                out_pdf.loc[len(out_pdf)] = [tag,item_id,item_name,int(item_count)]

                #print(tag, item_id, item_name, item_count)






            a = 1



        out_pdf.to_pickle(r'e:\projects\id_features_jeb.pkl')
        a = 1


