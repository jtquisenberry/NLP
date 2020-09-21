import pandas as pd



if __name__ == '__main__':
    csv_file = r"E:\Corpora\PII_Jeb_20190507\Paragraph_Reports\health_payment\combined.csv"
    df = pd.read_csv(csv_file)
    df.drop_duplicates(inplace=True)

    df2 = df[['guid', 'paragraph_position', 'tag']].loc[df['tag'] == 'PII|Health|health_payment']
    df2.drop_duplicates(inplace=True)
    print(df2.shape)
    df3 = df[['guid', 'paragraph_position', 'tag']].loc[df['tag'] == 'PII|Health|condition_treatment']
    df3.drop_duplicates(inplace=True)
    print(df3.shape)
    df4 = df[['guid', 'paragraph_position', 'tag']].loc[df['tag'] == 'PII|Health|applications_and_claims']
    df4.drop_duplicates(inplace=True)
    print(df4.shape)
    df5 = df[['guid', 'paragraph_position', 'tag']].loc[df['tag'] == 'PII|Employment|performance_review']
    df5.drop_duplicates(inplace=True)
    print(df5.shape)



    a = 1