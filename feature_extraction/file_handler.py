import pandas as pd
from shutil import copy2  # to copy files
import glob  # to find files matching a pattern
import os
import csv
import pandas as pd
from text_processor import TextProcessor

class FileHandler:


    def __init__(self, in_directory):
        self.in_directory = in_directory
        self.text_directory = os.path.join(in_directory, 'TEXT')
        self.tags_csv = os.path.join(in_directory,'tag_count.csv')
        self.tags_count_dictionary = dict()
        self.dat = os.path.join(in_directory,'loadfile.dat')
        self.df = None

        self.file_dictionary = dict()
        self.tags_dictionary = dict()
        print("In Direcotry: {0}".format(in_directory))


    '''
    def __init__(self, df, in_directory, out_directory):
        self.df = df
        self.file_locations = dict()
        self.in_directory = in_directory
        self.out_directory = out_directory
    '''

    '''
    def store_destinations_no_copy(self):
        # Get distinct files
        guids = self.df['GUID'].unique()
        for guid in guids:
            original_file = self.find_file(guid)
            self.file_locations[guid] = original_file

        a = 1
    '''

    def make_file_dictionary(self):

        for root, subdirs, files in os.walk(self.text_directory):
            for file in files:
                file_without_extension = os.path.splitext(file)[0]
                if file_without_extension not in self.file_dictionary:
                    self.file_dictionary[file_without_extension] = os.path.join(root, file)

    def get_tags(self):

        tags_count_list = []
        with open(self.tags_csv) as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for line in csv_reader:
                if line_count == 0:
                    #print(line)
                    line_count += 1
                else:
                    line_count += 1
                    tags_count_list.append(line)

        tags_count_list.sort(key=lambda x: (int(x[1]), x[0]))
        for i in range(0, len(tags_count_list) ):
            tag_index = i
            tag_name = tags_count_list[i][0]
            self.tags_count_dictionary[tag_name] = tag_index

        a = 1


    def get_minimum_tag(self, tag_string):


        if tag_string == '' or tag_string is None:
            tag_string = 'PII|Biometric|handwritten_signature;PII|Automotive|license_plate;' \
                             'PII|ID|bar_number;PII|Employment|performance_review;' \
                             'PII|Biometric|handwritten_signature;PII|Health|applications_and_claims'

        tag_list = tag_string.split(';')
        tl_orig = [tag for tag in tag_list]
        tag_list = [tag for tag in tag_list if tag in self.tags_count_dictionary]
        if not tag_list:
            return 'NONE'

        try:

            minimum_tag = sorted(tag_list, key=lambda x: self.tags_count_dictionary[x])[0]

        except Exception as e:
            print(e)

        return minimum_tag


    def read_dat(self):

        # The DAT file uses delimiter 20 (0x14) and qualifier 254 (0xFE)
        # Using engine parameter to avoid this:
        # ParserWarning: Falling back to the 'python' engine because ord(quotechar) > 127,
        # meaning the quotechar is larger than one byte, and the 'c' engine does not support
        # such quotechars; you can avoid this warning by specifying engine='python'

        df = pd.read_csv(self.dat, usecols=['GUID','Name','MIME Type','Kind','Tags'],
                         sep=chr(20), quotechar=chr(254), encoding='utf-8', engine='python')
        print(df.head())
        a = 1

        df['content'] = 'a'


        df['minimum_label'] = 'x'
        df['minimum_label'] = df['Tags'].apply(lambda x: self.get_minimum_tag(x))


        # Output labels to CSV for exploration
        output_csv = os.path.join(self.in_directory, 'tags_exploration.csv')
        df.to_csv(output_csv, columns=['GUID', 'minimum_label'], header=True, index=False)


        #df['content'] = df['content'].apply(lambda x: 'b')
        df['content'] = df.apply(self.get_text, axis=1)

        print(df.head())


        # Output DataFrame to pickle for feature extraction
        df.to_pickle(os.path.join(self.in_directory,'df_pickle_001.pkl'))





        print(df.head())


        b=2




    def get_text(self, row):
        guid = row['GUID']
        text_processor = TextProcessor(guid, self.file_dictionary)
        return text_processor.get_text()








    def copy_files(self):

        # Get distinct files
        guids = self.df['GUID'].unique()
        for guid in guids:
            original_file = self.find_file(guid)
            copied_file_directory = self.out_directory  #+ '\\' + guid + '.txt'
            copied_file = copy2(original_file, copied_file_directory)
            self.file_locations[guid] = copied_file

        a = 1

    def update_copied_files_match(self):

        for index, row in self.df.iterrows():
            guid = row['GUID']
            value = row['Value']
            value = value.replace(';', '\n')
            start_index = row['Match Start']

            if start_index > 0:
                start_index -= 1
                value = ' ' + value + ' '
            else:
                value = value + ' '



            if '5ea2da' in str(guid):
                #print(guid)
                #print(value)
                b = 1
            updated_file = self.file_locations[guid];
            with open(updated_file, 'r+', encoding='utf-8') as f:
                f.seek(start_index)
                f.write(value)
                a = 1

            b = 1

            #print(index, row['Value'])
            #print(row['c1'], row['c2'])

            #self.find_file(guid)

    '''
    def update_copied_files_context(self):

        for index, row in self.df.iterrows():
            guid = row['GUID']
            value = row['ValueContext']
            value = value.replace(';', '\n')
            start_index = row['Match Start']

            start_index -= 100
            if start_index < 0:
                start_index = 0


            if '5ea2da' in str(guid):
                print(guid)
                print(value)
                b = 1
            updated_file = self.file_locations[guid];
            with open(updated_file, 'r+', encoding='utf-8') as f:
                f.seek(start_index)
                f.write(value)
                a = 1

            b = 1
    '''



    def find_file(self, guid):

        # Get the subdirectory based on the first two characters of the guid.
        out_file_pattern = self.in_directory + '\\' + str(guid[0:2]) + '\\' + guid + '*' + '.txt'
        out_file = ''

        #print(out_file_pattern)
        for file in glob.glob(out_file_pattern):
            out_file = file

        return out_file