from collections import namedtuple
from pathlib import Path
import pickle
import sys

from essay_feature_extract import extract_features as get_feats

File = namedtuple('File', 'name text')

# gbr_model means gradient boosting regressor model (without lang_id or age)
# note that this model was trained using 'ls' or least squares regression
with open('elc_clc_combined_LRTC_model_13_march.pkl', 'rb') as f:
    loaded_pickle = pickle.load(f)
    gbr_model = loaded_pickle['model']

def read_input(arg):
    """ is `arg` a path or a file? 
        returns a list of read-in text to be 
        passed to the feature extraction tool. """
    
    text_list = []

    p = Path(arg)
    if p.is_file():
        if p.suffixes[0] == '.txt':
            with open(p, 'r', encoding='utf-8') as f:
                n_tup = File(p.name, f.read())
                text_list.append(n_tup)

        else:
            print('[ERROR]: File was detected, but the '
                  'input file is a `.txt` file.')
    
    elif p.is_dir():
        text_files = [x for x in p.iterdir() 
                      if x.is_file() and x.suffixes[0] == '.txt']
        # print('-'*78)
        # print(f'{len(text_files)} found.')
        # print('-'*78)

        for file in text_files:
            with open(file, 'r', encoding='utf-8') as f:
                n_tup = File(file.name, f.read())
                text_list.append(n_tup)

    else:
        print("[ERROR]: Didn't detect a `.txt` file or a directory")

    return text_list

def iterate(texts):
    """ takes texts from `read_input` 
        runs them through the essay scorer """

    for n_tup in texts:
        feat_set = get_feats(n_tup.text) # returns dict
        text_name = n_tup.name
        prediction = gbr_model.predict(feat_set)
        print(text_name, prediction[0], sep=',')


if __name__ == '__main__':
    texts = read_input(sys.argv[1])
    iterate(texts)
