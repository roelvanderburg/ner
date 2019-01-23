from tqdm import tqdm
import json
tqdm.pandas()
from utils.save_and_load import  load_obj


def check_names_in_offsets(non_match_data: str, match_data: str):
    """
    Returns statistics about the dataset
    Check how many names in tagged are found within the offsets, the # of files with a fuzzy match, and the total data size
    :param non_match_data: filename of the non match pd.DataFrame in the ./data folder
    :param match_data: filename of the  match pd.DataFrame in the ./data folder
    :return:
    """

    counter= 0
    offset_idxs =[]

    data_non_match = load_obj(non_match_data)
    data = load_obj(match_data)
    for idx, (match, offset_range) in enumerate(zip(data['match_annotation'], data['offsets'])):

        begin_int = match[0]['begin']
        if begin_int in range(offset_range[0], offset_range[1]):
            counter += 1
            offset_idxs.append(idx)

    print("Total number of files: {}, length of files with a higher than 85 >= percent (fuzzy) match: {}, "
            "of which {} documents have a match within the offset. Percentage tagged name inside the offsets: {} percent ".format(len(data_non_match) + len(data),len(data), counter, counter / len(data)))

def update_jsonfile(file_to_load: str, file_path_to_write_to: str, file_path_to_read_from: str):
    """
    Update JSON files by loading in the original json files from the file_path_to_read_from directory

    :param file_to_load: filename to load pd.DatFrame that has been processed by the n_gram matching
    :param file_path_to_write_to: path to folder to write the altered json files to
    :param file_path_to_read_from: path to folder of original json files
    :return:
    """

    data = load_obj(file_name=file_to_load)

    for idx, (json_file_name, text, annotation) in enumerate(tqdm(zip(data['path'].tolist(), data['text'], data['match_annotation']), desc='Uploading annotated JSON files in {}'.format(file_path_to_write_to))):
        try:
            with open(file_path_to_read_from + json_file_name, "r") as f:
                jsonfile = json.load(f)

                # Add to annotations.
                jsonfile['annotations'].update({'ner_tag': annotation})
                jsonfile['text'] = text

                # Save our changes to JSON file
            with open(file_path_to_write_to + json_file_name, 'w') as outfile:
                json.dump(jsonfile, outfile)

        except: FileExistsError


if __name__ == '__main__':
    update_jsonfile(file_to_load='13_13_all_data', file_path_to_write_to='/srv/data/test_name_annotations/', file_path_to_read_from='/srv/data/lwb/processed/')
