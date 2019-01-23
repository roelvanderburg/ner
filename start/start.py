from pre_process import get_json
from matching import n_gram_matching, update_files
from classifying import train


if __name__ == '__main__':

    file_name_to_save = 'FILE_NAME'

    # Loads in JSON files from the file_path_to_load and saves it as a pd.DataFrame in an (existing) ../data  as file_name_to_save
    # Size_to_load = None to process whole dataset, else int to process a part.
    get_json.ParseJSONS(file_path_to_load='/srv/data/lwb/processed/', file_name_to_save=file_name_to_save, size_to_load=None).load_and_fix()

    # Checks ../data to load the pd.DataFrame in file_to_load, performs matching and saves the pd.DataFrame in ../data as file_name_save.
    n_gram_matching.FuzzyGramMatching(file_to_load=file_name_to_save, file_name_save=file_name_to_save, ftfy_fix=True, verbose=True, spacy=False, conf_thresh=90).apply_fuzzy_matching()

    # Checks .../data to load pd.DataFrame, loads original json files from file_path_to_read_from, and writes the new json with annotation
    # for the front end to file_path_to_write_to.
    update_files.update_jsonfile(file_to_load=file_name_to_save, file_path_to_read_from='/srv/data/lwb/processed/', file_path_to_write_to='/srv/data/test_name_annotations/')

    # Check ../data from data_file pd.DataFrame to load and train SVM.
    train.train(data_file=file_name_to_save)

