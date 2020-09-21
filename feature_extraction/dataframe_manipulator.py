

import pandas as pd
import random


class DataframeManipulator():

    def __init__(self, df):
        self.df = df
        self.labels = []
        self.df_equalized = self.df.loc[self.df['guid'] == 'xxx']

    def equalize_rows_by_label(self, labels=None):

        # If labels is not defined, then use all the labels in the dataframe.
        if labels == None:
            labels = list(self.df['minimum_label'].drop_duplicates())

        # The new dataframe should include only those rows meeting the labels condition.
        self.df = self.df.loc[self.df['minimum_label'].isin(labels)]

        # Determine the smallest number of rows corresponding to a given label.
        min_rows = self.df.groupby(['minimum_label'], as_index=False).count()['guid'].min()

        for label in labels:
            # Get the rows corresponding to the label
            df_temp = self.df.loc[self.df['minimum_label'] == label]

            # Randomize the rows
            # Notice that DataFrame.temp does not have an inplace parameter.
            df_temp = df_temp.sample(frac=1).reset_index(drop=True)

            # Get the top min_rows rows
            df_temp = df_temp[0:min_rows]

            # Append to equalized DataFrame
            self.df_equalized = self.df_equalized.append(df_temp, ignore_index=True)

        return self.df_equalized


