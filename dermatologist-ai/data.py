'''
Difines functions to read/write data from/into the files
'''
import os

def load_dataset(path):
    '''
    Returns found image files along with their labels

    :param path: Path where to find the files. Expected format: there are several
                 subdirectories under the path for each type of skin cancer,
                 there are many images of this type in each subdirectory
    :return: Tuple of (file name, skin cancer type)
    '''
    subdirs = os.listdir(path)
    classes = {}
    labels = []
    files = []

    class_to_digit = 0
    for subdir in subdirs:
        classes[subdir] = class_to_digit
        class_to_digit += 1
        for file in os.listdir(path + '/' + subdir):
            labels.append(subdir)
            files.append(path + '/' + subdir + '/' + file)

    return classes, labels, files
