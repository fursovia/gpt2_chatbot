import argparse
import os
import unicodecsv
import random
from six.moves import urllib
import tarfile
import csv

def translate_dialog_to_lists(dialog_filename):
    """
    Translates the dialog to a list of lists of utterances. In the first
    list each item holds subsequent utterances from the same user. The second level
    list holds the individual utterances.
    :param dialog_filename:
    :return:
    """

    dialog_file = open(dialog_filename, 'r')
    csv.register_dialect('dialectread', quoting=csv.QUOTE_NONE,delimiter='\t')
    dialog_reader =csv.reader(dialog_file, dialect='dialectread')

    # go through the dialog
    first_turn = True
    dialog = []
    same_user_utterances = []
    # last_user = None
    dialog.append(same_user_utterances)


    for dialog_line in dialog_reader:

        if first_turn:
            last_user = dialog_line[1]
            first_turn = False

        if last_user != dialog_line[1]:
            # user has changed
            same_user_utterances = []
            dialog.append(same_user_utterances)

        same_user_utterances.append(dialog_line[3])

        last_user = dialog_line[1]

    return dialog

def convert_csv_with_dialog_paths(csv_file):
    """
    Converts CSV file with comma separated paths to filesystem paths.
    :param csv_file:
    :return:
    """

    def convert_line_to_path(line):
        file, dir = map(lambda x: x.strip(), line.split(","))
        return os.path.join(dir, file)

    return map(convert_line_to_path, csv_file)


def prepare_data_maybe_download(directory):
    """
    Download and unpack dialogs if necessary.
    """
    filename = 'ubuntu_dialogs.tgz'
    url = 'http://cs.mcgill.ca/~jpineau/datasets/ubuntu-corpus-1.0/ubuntu_dialogs.tgz'
    dialogs_path = os.path.join(directory, 'dialogs')

    # test it there are some dialogs in the path
    if not os.path.exists(os.path.join(directory, "10", "1.tst")):
        # dialogs are missing
        archive_path = os.path.join(directory, filename)
        if not os.path.exists(archive_path):
            # archive missing, download it
            print("Downloading %s to %s" % (url, archive_path))
            filepath, _ = urllib.request.urlretrieve(url, archive_path)
            print
            "Successfully downloaded " + filepath

        # unpack data
        if not os.path.exists(dialogs_path):
            print("Unpacking dialogs ...")
            with tarfile.open(archive_path) as tar:
                tar.extractall(path=directory)
            print("Archive unpacked.")

        return


#####################################################################################
# Command line script related code
#####################################################################################

if __name__ == '__main__':

    # download and unpack data if necessary
    prepare_data_maybe_download(".")

    f = open(os.path.join("meta", "trainfiles.csv"), 'r')
    dialog_paths = list(map(lambda path: os.path.join(".", "dialogs", path), convert_csv_with_dialog_paths(f)))

    print(len(dialog_paths))
    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    for c, dialog_path in enumerate(dialog_paths):
        if (c != 100):
            dialog = translate_dialog_to_lists(dialog_path)
            for idx in range(len(dialog)):
                dialog[idx] = " ".join(dialog[idx])
            with open('sample.csv', 'a') as csvFile:
                writer = csv.writer(csvFile, dialect='myDialect')
                writer.writerow(dialog)
        else:
            exit(1)
