import file_handler
import pandas as pd
import os

def get_text_from_row(row):
    guid = row['guid']
    text_start = row['paragraph_position']
    text_length = row['paragraph_length']
    path = fh.file_dictionary[guid]
    text = ''
    with open(path, encoding='utf8', newline='') as f:
        text = f.read()
        text = text[text_start:text_start + text_length]

    return text




if __name__ == '__main__':

    in_directory = r"E:\Corpora\PII_Jeb_20190507"
    out_directory = r"E:\Corpora\PII_Jeb_20190507\export_lines"

    label_dictionary = {'PII|Health|condition_treatment':0,
                        'PII|Health|health_payment': 1,
                        'PII|Health|applications_and_claims':2,
                        'PII|Employment|performance_review':3}

    fh = file_handler.FileHandler(in_directory=in_directory)
    # Guid / path dictionary.
    fh.make_file_dictionary()

    csv_file = r"E:\Corpora\PII_Jeb_20190507\Paragraph_Reports\health_payment\combined.csv"
    df = pd.read_csv(csv_file,usecols=['guid','paragraph_position','paragraph_length','tag'])
    df.drop_duplicates(inplace=True)
    df = df.loc[df['tag'].isin(['PII|Health|condition_treatment','PII|Health|health_payment','PII|Health|applications_and_claims', 'PII|Employment|performance_review'])]


    df['content'] = ''
    df['content'] = df.apply(get_text_from_row, axis=1)
    df['category_id'] = 0
    df['category_id'] = df.apply(lambda x: label_dictionary[x['tag']], axis=1)
    df['minimum_label'] = df['tag']



    df.to_pickle(os.path.join(in_directory, 'pickled_paragraphs.pkl'))
    print('DONE')






