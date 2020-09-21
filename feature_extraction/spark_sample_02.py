# It is necessary to import findspark before pyspark.
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer


if __name__ == '__main__':

    spark = SparkSession.builder.appName("CountVectorizerExample").getOrCreate()

    df = spark.createDataFrame([
        (0, "a b c".split(" ")),
        (1, "a b b c a".split(" ")),
        (1, "a b b e a".split(" ")),
        (1, "a b b c e".split(" ")),
        (1, "a b d c a".split(" ")),
        (1, "e e e e e e e e e e".split(" ")),
        (0, "a f f g a f".split(" "))
    ], ["id", "words"])

    df.show()

    # fit a CountVectorizerModel from the corpus.
    cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=4, minDF=2.0)


    model = cv.fit(df)


    print('Min TF', model.getMinTF())
    print('Min DF', model.getMinDF())
    print('Max DF', model.getMaxDF())


    model.vocabulary

    # It appears that the default order of model.vocabulary is the order of significance of the features in descending order. Notice that 'e' appears first and it is most represented in the DataFrame.




    result = model.transform(df)




    result.show(truncate=False)

    # It appears that indexes in the features vector refer to indices in the `model.vocabulary` list. Note that model.vocabulary[0] is 'e'. Counts of [0] are equal to counts of 'e'.



    feature_counts_dictionary = dict()
    for row in result.rdd.collect():
        # print(row)
        # print(row.features.indices)
        # print(row.features)
        # print(row.features[1])
        for i in range(0, len(row.features.indices)):
            feature_id = row.features.indices[i]

            feature_name = model.vocabulary[feature_id]
            # print(feature_name)
            feature_count = row.features[int(feature_id)]
            # print(feature_count)
            # print("feature_id {0}, feature_count {1}".format(row.features.indices[i], row.features[i]))
            if feature_name in feature_counts_dictionary:
                feature_counts_dictionary[feature_name] += feature_count
            else:
                feature_counts_dictionary[feature_name] = feature_count




    print('Most Informative Features')
    for feature in model.vocabulary:
        print(feature, feature_counts_dictionary[feature])






