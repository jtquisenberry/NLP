from pickle_handler import PickleHandler

N_best_features = 10

ph = PickleHandler(in_pickled_df=r'E:\Corpora\PII_Directory_20190507\sklearn_features.pkl')
best_features_dict = ph.df

targets = [target for target in best_features_dict.keys()]
print(targets)

features = best_features_dict['PII|Tax|itin_tax_id'][0][-N_best_features:]

print(features)


a = 1