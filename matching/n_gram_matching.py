#! /usr/bin/env python
import re
from fuzzywuzzy import fuzz, process
from nltk import ngrams
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pandas as pd
desired_width = 500
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)
import ftfy
import string
import time
from typing import List
from utils.save_and_load import save_obj, load_obj
import spacy
nlp = spacy.load('nl_core_news_sm')


class FuzzyGramMatching:

    def __init__(self, file_name_save: str, file_to_load: str,  n_gram=7, spacy=False, ftfy_fix= False, verbose=True, conf_thresh=90):
        """
        Initialize the FuzzyGramMatching class that performs n-gram matching on names
        :param file_name_save: filename of pd.DataFrame to be saved in ../data
        :param file_to_load: filepath to jsonfiles
        :param n_gram: how many n-grams to try matching on
        :param spacy: boolean to include a spacy ner-classifictaion annotation column
        :param ftfy_fix: perform matching with or without ftfy fixing
        :param verbose: boolean to control verbosity of output
        :param conf_thresh: minimal fuzzy match score that a match should have to be included.
        """

        self.start = time.time()
        self.data = load_obj(file_to_load)

        self.ftfy_fix = ftfy_fix
        self.spacy = spacy
        self.file_name_save = file_name_save
        self.n_gram = n_gram
        self.verbose = verbose
        self.conf_thresh = conf_thresh

    @staticmethod
    def save_obj(save_object: pd.DataFrame, file_name: str):
        """
        method
        :param save_object: object to be saved
        :param file_name: file name of the object to be saved in /data directory, makes directory if /data not exists
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

    @staticmethod
    def tokenize_documents(document: str) -> List[str]:
        """
        :param document: input string
        :return: list of words split on whitespace
        """
        tokens = re.split(r'\s+', document)
        return tokens

    @staticmethod
    def make_indices(document: str) -> List[int]:
        """
        :param document: string
        Receives a string and returns a list of integers representing the offsets of each individual word split on whitespace
        e.g: document=  'Hello World' --> [0, 5, 6, 11]   or  '  Hello World' --> [2, 7, 8, 13]
        so word one ranges from 0 to 5 and word two from 6 to 11. Thus 5 to 6 represents whitespace.
        :return: list indices
        """

        tokens = re.split(r'(\s+)', document)
        indices = [len(token) for token in tokens]

        for i in range(1, len(indices)):
            indices[i] = indices[i] + indices[i - 1]

        if document[0].isspace():
            indices.pop(0)
        else:
            indices = [0] + indices

        return indices

    def concat_tokens(self, tokens: List[str]) -> str:
        """
        :param tokens: list of n_grams
        :return: single space separated string
        e.g ['Hello, 'World']  -> 'Hello World'
        """
        return " ".join(tokens)

    @staticmethod
    def n_gram_to_index(indices: List[int], count_n_gram: int, n_gram_token_list: List[str]) -> List:
        """
        Receives a list of strings and returns a list of triples where a begin and end integer is added to each string
        where the begin and end integer is determined as if the list of strings was one string
        e.g  indices=[0, 5, 6, 11], count_n_gram=1, n_gram_token_list=['Hello', 'World'] --> [(0,5,'Hello'), (6,11,'World)]
        :param indices: list of ints
        :param count_n_gram: integer
        :param n_gram_token_list: list of strings
        :return:
        """
        new_list = []
        first_el = 0
        second_el = count_n_gram * 2 - 1
        for idx, n_gram in enumerate(n_gram_token_list):
            new_list.append((indices[first_el], indices[second_el], n_gram))
            first_el += 2
            second_el += 2

        return new_list

    @staticmethod
    def check_potential_of_ngram(n_gram: str) -> bool:
        """
        Inputs a string and checks if there are:
        at least two capital letters present
        at least one lowercase
        at least 3 characters
        at max 25 characters

        :param n_gram: n_gram to check for potential
        :return: True if all conditions are met
        """

        rules = [lambda s: sum(x.isupper() for x in s) >= 2,    # must have at least two uppercase
                 lambda s: len(s) >= 5,                         # must be at least 3 characters
                 lambda s: len(s) <= 32,                        # must be at max 35 characters
                 lambda s: sum(x.isdigit() for x in s) <= 2,    # no more than 2 digits
                 ]

        return all(rule(n_gram) for rule in rules)

    def match_one_n_gram_column(self, number_n_gram, col_name):
        """
        Perform fuzzy matching on one column
        :param number_n_gram: int
        :param col_name: string named n_gram with respective i
        :return: pandas matrix
        """

        fuzzymatch = 'fuzzy_match_{}'.format(col_name)

        self.data[col_name] = self.data['tokenized_text'].progress_apply(lambda document_string: list(ngrams(document_string, n=number_n_gram)))

        # For example with n-gram=2 from:  ['This', 'is', 'an', 'example'] to ['This is', 'an example']
        self.data[col_name] = self.data[col_name].progress_apply(lambda tokens: [self.concat_tokens(triple) for triple in tokens])

        if number_n_gram == 1:
            # From n-grams to a list of indices
            self.data['indices_list'] = self.data['text'].progress_apply(lambda document_string: self.make_indices(document_string))

        # From n grams to n_grams triples with a begin and end index added for each n_gram
        tqdm.pandas(desc='Adding index to n_gram {}'.format(number_n_gram))
        self.data[col_name] = self.data.progress_apply(
            lambda row: self.n_gram_to_index(indices=row['indices_list'], count_n_gram=number_n_gram, n_gram_token_list=row[col_name]), axis=1)


        tqdm.pandas(desc='Check potential of n_gram {} '.format(number_n_gram))
        # Throw away triples that are either too short or long to be a match
        self.data[col_name] = self.data[col_name].progress_apply(
            lambda n_gram_list: [n_gram for n_gram in n_gram_list if self.check_potential_of_ngram(n_gram[2])])

        tqdm.pandas(desc='Fuzzy matching between n_gram: {}, and database names'.format(number_n_gram))
        self.data[fuzzymatch] = self.data.progress_apply(
            lambda row: [[triple, process.extract(triple[2], row['names'], scorer=fuzz.ratio, limit=1)] for triple in  row[col_name]], axis=1)

        best_matches_nth = "best_match_{}".format(col_name)

        # Fuzzymatch column :  [(360, 377, 'J.J. jan Jantjes'), ('J J JAN JANTJES', 97)] ->  So the first tuple
        # we found 'J.J. jan Jantjes' in the text with offsets 360,377. The closest database match is J J VAN JANTJES with a score of 97
        # Keep the matches that score higher than 90 (this cutoff is rather arbitrary, but we do not have a gold standard to test against)
        tqdm.pandas(desc='Threshold the matches ')
        self.data[best_matches_nth] = self.data[fuzzymatch].progress_apply(
            lambda x: [(triple, best_match) for triple, best_match in x if best_match[0][1] >= self.conf_thresh])

        # Drop the n-gram column for efficiency, matching on it has been done so not necessary
        self.data.drop([col_name], axis=1)

    def spacy_matching(self, column_name: str):
        """
        Funtion to execute spacy ner tagging on the text only with regard to the entity label PER.
        :param column_name: column name to write the spacy match annotations to
        :return:
        """

        tqdm.pandas(desc='print("Spacy NER matching :')
        self.data['entities'] = self.data['text'].progress_apply(lambda text:
                                [[ent.text, ent.start_char, ent.end_char, ent.label_] for ent in (nlp(text)).ents if (ent.label_ == 'PER' and ent.text is not '\n' and len(ent.text) >= 4)])

        self.data[column_name] = self.data.progress_apply(lambda row:
                                                               [
                                                                   {
                                                                       "begin": row['entities'][i][1],
                                                                       "end": row['entities'][i][2],
                                                                       "text": row['entities'][i][0],
                                                                       "label": row['entities'][i][3],
                                                                       "names from BSN response": row['names']
                                                                   }
                                                                   for i in range(len(row['entities']))
                                                               ]
                                                               , axis=1
                                                               )

        self.data = self.data.drop(columns=['entities'])

    def concat_columns(self):
        """
        Concatenate the best match columns.
        :return:
        """

        columns = ['best_match_n_gram_{}'.format(i) for  i in range(1,self.n_gram+1)]

        self.data['best_match_all'] = self.data[columns].sum(axis=1)


    def annotate_matching(self):
        """
        Refactors the best_matches into annotation (dictionary) objects in a new column, for the frontend application
        :return:
        """

        tqdm.pandas(desc="Creating annotation column")
        self.data['match_annotation'] = self.data.progress_apply(lambda row:
                                                                 [
                                                                     {
                                                                         "begin":row['best_match_all_cleaned'][i][0],
                                                                         "end":row['best_match_all_cleaned'][i][1],
                                                                         "text":row['best_match_all_cleaned'][i][2],
                                                                         "fuzzy_confidence":row['best_match_all_cleaned'][i][4],
                                                                         "closest_database_name":row['best_match_all_cleaned'][i][3],
                                                                         "names from BSN response": row['names']
                                                                     }
                                                                     for i in
                                                                     range(len(row['best_match_all_cleaned']))
                                                                 ]
                                                                 , axis=1
                                                                 )

    def fix_text(self):
        """
        Fixes the text with the ftfy module.
        :return:
        """
        self.data['original_text'] = self.data['text']
        tqdm.pandas(desc="Fixing text with ftyfy package")
        self.data['text'] = self.data['text'].progress_apply(
            lambda text_string: ftfy.fix_text(text_string, fix_encoding=True))

    def remove_empty_text(self):
        """
        Remove rows where text field is empty
        :return:
        """
        print("# Rows before non text are removed: {} ".format(len(self.data)))
        self.data = self.data[self.data['text'].map(lambda d: len(d) > 0)]
        print("# Rows after non text are removed: {} ".format(len(self.data)))


    def clean_smaller_matches(self):
        """
        Cleans smaller matches:
        e.g. [[360, 364, 'J.J.', 'J J', 100], [360, 377, 'J.J. van Jantjes', 'J J VAN JANTJES', 97]] --> [[360, 377, 'J.J. van Jantjes', 'J J VAN JANTJES', 97]]
        :return: fixes the match column in the pandas matrix
        """
        total_list = []

        for idx, best_match_list in enumerate(self.data['best_match_all_cleaned']):
            best_match_list_cleaned = best_match_list
            pop_ids = []
            if len(best_match_list) > 1:
                for idy, best_match in enumerate(best_match_list):
                    for other_match in best_match_list[idy + 1:]:
                        # Check if best_match fits inside the range of any of the other matches.
                        if best_match[0] in range(other_match[0] - 1, other_match[1] + 1):
                            pop_ids.append(idy)

                pop_ids = set(pop_ids)
                # Are there ids to pop
                if len(pop_ids) >= 1 and len(best_match_list_cleaned) > 0:
                    #
                    for bad_index in sorted(pop_ids, reverse=True):
                        best_match_list_cleaned.pop(bad_index)

            total_list.append(best_match_list_cleaned)

        self.data['best_match_all_cleaned'] = total_list

    def count_length(self):
        """
        Prints out the total length row-wise of the dataframe
        :return:
        """
        print("Length total set:  ", len(self.data))

    def clean_last_character_punctuation(self):
        """
        Clean the last character of a match if its punctuation:
        e.g  J. Jantjes, -> J. Jantjes
        :return:
        """

        clean_matches = []
        for idx, best_match_list in enumerate(self.data['best_match_all_cleaned']):
            cleaned = []
            for best_match in best_match_list:
                # Is the last character of bestmatch[2] a part of string.punctuation, then delete it.
                if best_match[2][-1] in string.punctuation:
                    cleaned.append([best_match[0], best_match[1], best_match[2][:-1], best_match[3], best_match[4]])
                else:
                    cleaned.append(best_match)
            clean_matches.append(cleaned)

        self.data['best_match_all_cleaned'] = clean_matches

    def apply_fuzzy_matching(self):
        """
        Apply fuzzy matching of the n-grams columns to the database
        :param n_gram: int, default: 5
        Number of n grams to perform matching on.
        :return:
        """

        self.remove_empty_text()

        if self.ftfy_fix:
            self.fix_text()

        tqdm.pandas(desc='Tokenizing')
        self.data['tokenized_text'] = self.data['text'].progress_apply(lambda text: self.tokenize_documents(text))

        # Operate on n-gram 1 to n_gram
        for i in range(1, self.n_gram + 1):
            col_name = "n_gram_{}".format(i)

            if self.verbose:
                print(' ----------', col_name, ' ----------  ')
                print(self.data.info(memory_usage='deep'))

            # Perform fuzzy n-gram matching on one column
            self.match_one_n_gram_column(number_n_gram=i, col_name=col_name)

        self.concat_columns()

        if self.verbose:
            print(' ----------  DataFrame Memory  ----------  ')
            print(self.data.info(memory_usage='deep'))
            print(list(self.data))

        # Uncomment so save files without matches for inspection:

        # data_non_match = self.data[self.data['best_match_all'].map(lambda x: len(x) == 0)]
        # self.save_obj(data_non_match, 'n_gram_non_match')

        # Remove rows which did not get an above threshold match ie. empty rows
        self.data = self.data[self.data['best_match_all'].map(lambda d: len(d) > 0)]

        # Clean up structure best matches
        # E.g from: ((352, 357, JAN), [('JAN', 100)]) to [352, 357, JAN, JAN, 100]
        self.data['best_match_all_cleaned'] = self.data['best_match_all'].progress_apply(
            lambda row: [[row[i][0][0], row[i][0][1], row[i][0][2], row[i][1][0][0], row[i][1][0][1]] for i in
                         range(len(row))])

        # Drop the column with unclean structure
        self.data = self.data.drop(columns=['best_match_all'])

        self.clean_smaller_matches()

        # Clean matches punctuation
        self.clean_last_character_punctuation()

        # return a new column with annotation dictionaries
        self.annotate_matching()

        if self.spacy:
            self.spacy_matching(column_name='spacy_match_annotation')

        self.save_obj(self.data, self.file_name_save)
        end = time.time()
        print("Time for the whole matching: ", end - self.start, "seconds.")


def evaluate_file(data_file: str, json_link : str):
    """
    Prints out in a loop for every n_gram: n_grams with their fuzzy match from the database,
    the best_match_n_grams that have a score higher than the threshold. And finally all the
    final matches.
    :param data_file: file_name for pd.DataFrame to load that has been matched
    :param json_link: filename to inspect: e.g C72A-AD04-****-****-****.json
    :return:
    """
    data = FuzzyGramMatching.load_obj(data_file)
    idx = data.index[data['path'] == json_link]

    print(list(data))

    for i in range(1, 8):
        print("---- N-gram {}  ---- ".format(i))
        print("length potentials: ", len(data['fuzzy_match_n_gram_{}'.format(i)][idx[0]]))
        print("N_gram {} ".format(i), data['n_gram_{}'.format(i)][idx[0]])
        print(data['fuzzy_match_n_gram_{}'.format(i)][idx[0]])
        print(data['best_match_n_gram_{}'.format(i)][idx[0]])

    print("All final matches")
    print(data['best_match_all_cleaned'.format(i)][idx[0]])

