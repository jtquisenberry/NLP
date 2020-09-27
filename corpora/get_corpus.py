import os
import sys
from urllib.request import urlopen
from zipfile import ZipFile

corpus_name = 'sttng'

class Corpus():
    def __init__(self, name=''):
        self.name = name



os.path.exists(r'./{0}'.format(corpus_name))
corpus = Corpus('Star Trek: The Next Generation')

zipurl = 'Valid URL to zip file'
    # Download the file from the URL
zipresp = urlopen(zipurl)
    # Create a new file on the hard drive
tempzip = open("/tmp/tempfile.zip", "wb")
    # Write the contents of the downloaded file into the new file
tempzip.write(zipresp.read())
    # Close the newly-created file
tempzip.close()
    # Re-open the newly-created file with ZipFile()
zf = ZipFile("/tmp/tempfile.zip")
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
zf.extractall(path = '<extraction_path>')
    # close the ZipFile instance
zf.close()


