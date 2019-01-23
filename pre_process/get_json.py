#! /usr/bin/env python
import argparse
import sys
sys.path.append('../')
from pre_process.request_name_from_bsn import get_name_address
from utils.save_and_load import save_obj, load_obj
import json
import glob
from tqdm import tqdm
import pandas as pd
import re


class ParseJSONS:

    def __init__(self, file_path_to_load: str, file_name_to_save: str, size_to_load=None):
        """
        Initialize the class to read in the json files, process them, and saves it in a ../data folder
        :param file_path_to_load: file_path to the unprocessed json files
        :param file_name_to_save: save_name to save pd.DataFrame
        :param size_to_load: int, int to load in part of data, else None to load in whole dataset
        """

        my_list = [file for file in glob.glob(str(file_path_to_load) + "*.json")]
        self.file_path_list = my_list
        self.data = pd.DataFrame
        self.file_name_to_save = file_name_to_save
        self.size_to_load = size_to_load

    def load_and_fix(self):
        """
        Load json files, parse out necessary fields and save a pandas matrix in the .data folder
        :return: saves a pandas matrix in the .data map
        """
        # Read in json
        self.read_json()

        if self.size_to_load:
            self.data = self.data[:self.size_to_load]

        # Add names from database given _bsn:
        self.extend_dataframe_with_personnames()

        # Clean rows in the data_frame where the names column is empty - > thus no response from the database
        self.clean_none_response()

        # Fix path from A09.pdf to A09.json
        self.fix_path()

        # Get the correct names from the database response
        self.parse_names_from_response()

        print(" --- Final Shape Data ---")
        print(self.data.shape)
        print(list(self.data))

        # Save pickled object in ./data map
        self.save_obj(self.data, self.file_name_to_save)

    def fix_path(self):
        """
        Fixes path prefix from .pdf to .json
        :return: Pandas matrix with fixes to the 'path' column
        """
        paths = self.data['path'].tolist()
        prefixes = [re.findall(r'[A-Z\-0-9]+', path) for path in paths]
        prefix_good = [str(prefix[0]) + ".json" for prefix in prefixes]
        self.data['path'] = prefix_good

    def read_json(self):
        """
        Read in JSON files and get the required fields
        :return: updates the pandas dataframe atrribute in the class
        """
        list_filepath_bsn_test = []

        try:
            for json_file in tqdm(self.file_path_list, desc='Loading in json files'):
                with open(json_file) as f:
                    data = json.load(f)

                    # Get out: filepath, _bsn, text, offsets and main text
                    list_filepath_bsn_test.append([data['filepath'], data['fields']['_belanghebbende_bsn'], data['text'],
                                                  data['offsets']['main_text']])
        except:
            print("Faulty json file: ,", json_file)

        # Make it into a data_frame
        self.data = pd.DataFrame(list_filepath_bsn_test)
        headers = ['path', 'bsn', 'text', 'offsets']
        self.data.columns = headers

    def extend_dataframe_with_personnames(self):
        """
        Gets the _BSN from the list: first element and requests a response from the database then extend it
        to the matrix to get to shape: [file_path, _bsn, text, offsets_maintext, names_response]
        :return: extend the list with personNames response
        """

        # Create an empty name column
        self.data['names'] = self.data.apply(lambda _: '', axis=1)

        for idx, bsn_list in enumerate(tqdm(self.data['bsn'], desc='Requesting response from database given bsn')):
            try:
                # Check if its a 9 digit number BSN or a list
                if isinstance(bsn_list, str):
                    self.data['names'][idx] = get_name_address(bsn_list)
                # Take the first element of the list
                else:
                    self.data['names'][idx] = get_name_address(bsn_list[0])
            except:
                print('\n')
                print("This BSN did not get a response: ", bsn_list)

    def clean_none_response(self):
        """
        Clean out the rows where the _bsn got a None response (ghost people)
        :return: Object (the matrix) with the none response rows deleted
        """

        print("# Rows before non response are removed: {} ".format(len(self.data)))
        self.data = self.data[self.data['names'].map(lambda d: len(d) > 0)]
        print("# Rows after non response are removed: {} ".format(len(self.data)))

    def parse_names_from_response(self):
        """
        Given the response column parse out the necessary name permutations to match the text
        :return: updates the pandas names column and updates the attribute in the class
        """
        unnecessary_keys = ['usageType', 'usageValue', 'startDate', 'xpersonNameUsageType', 'xpersonNameUsageValue']

        full_names_to_parse_list = []

        # Get out names, leave out unnecessary_keys
        for dictionary in self.data['names']:
            names_to_parse = []
            for key, value in dictionary['personNames'][0].items():
                if key in unnecessary_keys:
                    pass
                else:
                    names_to_parse.append(value)

            full_names_to_parse_list.append(names_to_parse)

        self.data['names'] = full_names_to_parse_list

    @staticmethod
    def save_obj(save_object: pd.DataFrame, file_name: str):
        """
        method
        :param save_object: object to be saved
        :param file_name: file name of the object to be saved in ./data directory, makes directory if not exists
        :return:
        """
        save_obj(save_object, file_name)

    @staticmethod
    def load_obj(file_name: str) -> pd.DataFrame:
        """
        :param file_name: name of pickle object to load from the ./data directory
        :return:
        """
        return load_obj(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process json files for matching')

    parser.add_argument(metavar='filepath', type=str, dest='filepath',
                        help='filepath to the json files that need to be processed')

    parser.add_argument(metavar='save_file_name', type=str, dest='save_file_name',
                        help='filename for the pd.DataFrame to be pickled')

    results = parser.parse_args()


    get_json = ParseJSONS(file_path_to_load=results.filepath, file_name_to_save=results.save_file_name)

