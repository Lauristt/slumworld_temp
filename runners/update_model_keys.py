'''Utility function for quickly updating the keys (layer names) of a saved model to enable compatibility with previous versions of the code
    Usage:
        >>> python3 update_model_keys.py -p /path/to/saved/checkpoint.ckpt -r text2remove -s text2substitue
        e.g. the following line:
        >>> python3 update_model_keys.py -p ./checkpoints/last.ckpt -r encoding_parts -s encoder
        will replace the text 'encoding_parts' with 'encoder' in all model layer names
        
'''

import os
import argparse
import warnings
from collections import OrderedDict
import torch


def remove_pattern(state_dict, remove_pattern='', new_pattern=''):
    odict = OrderedDict()
    updated = False
    for k, v in state_dict.items():
        new_key = k.replace(remove_pattern, new_pattern)
        if remove_pattern in k:
            print(f'changing layer name: {k}\tto: {new_key}')
            updated = True
        odict[new_key] = v
    state_dict = odict
    return state_dict, updated



def main(args):

    try:
        model = torch.load(args['saved_model_file'], map_location='cpu')
        model['state_dict'], updated = remove_pattern(model['state_dict'], args['remove_pattern'], args['new_pattern'])
        if updated:
            torch.save(model, args['saved_model_file'].replace('.ckpt', '_updated.ckpt'))
        else:
            warnings.warn("Suuplied patern not found in any layer name! Nothing to update.")
    except Exception as Error:
        print(Error)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--saved_model_file', type=str, help='Path to a saved model/checkpoint file.')
    parser.add_argument('-r', '--remove_pattern', type=str, help='String pattern to remove (replace) from layer name.')
    parser.add_argument('-s', '--new_pattern', type=str, help='String pattern with which to substitute the removed part of a layers\' name.')

    args = vars(parser.parse_args())
    
    main(args)