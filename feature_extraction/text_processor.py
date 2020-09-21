import io

class TextProcessor:

    def __init__(self, guid, file_dictionary):

        self.guid = guid
        self.file_dictionary = file_dictionary

    def get_text(self):



        #self.guid = self.guid.replace('-','')

        with open(self.file_dictionary[self.guid], encoding='utf-8') as f:
            self.content = f.read()
            a = 1

        # Analyze only the first 30 Mcharacters
        self.content = self.content[:30*1024*1024]
        self.perform_analytics()
        return self.content


    def perform_analytics(self):
        #print(len(self.content))
        #print(self.content.count(' '))
        b = 1

