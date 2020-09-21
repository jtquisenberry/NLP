import file_handler


if __name__ == '__main__':

    in_directory = r"E:\Corpora\PII_Jeb_20190507"

    fh = file_handler.FileHandler(in_directory=in_directory)
    fh.make_file_dictionary()
    fh.get_tags()
    #minimum_tag = fh.get_minimum_tag(None)
    #print(minimum_tag)
    fh.read_dat()

    print("1")