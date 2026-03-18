#!/bin/bash

# Collect the path to a (tab separated) list of k hyperparameter combinations from the command line and train k models on a single gpu
# each. MAKE SURE THAT all config files POINT to the CORRECT dataset.csv 
PARAM_FILE=${1:-"~/slumworldML/config/hyperparameter_search.lst"}

if [[ $PARAM_FILE == "h" ]] || [[ $PARAM_FILE == "help" ]]; then
    echo ""
    echo "Script to use for running k single model training runs locally (sequentially) based on k-hyper-parameter combinations defined in "
    echo "tab separated list file: hyperparameter_search.lst"
    echo ""
    echo "ARGS:"
    echo "   PARAM_FILE     str, the parameter_combination file to use - place it within \"\" [default:\"~/slumworldML/config/hyperparameter_search.lst\"]"
    echo "USAGE:"
    echo "   To train a model:"
    echo "   . run-hyperparameter_search.sh \"~/slumworld/config/hyperparameter_search.lst\""
    echo ""
    echo "   To display usage help:"
    echo "   . run-hyperparameter_search.sh  help"
    echo ""
    return 0
fi

chmod u+x $PARAM_FILE
echo $PARAM_FILE

while IFS="" read -r p || [ -n "$p" ]; do 

    echo "$p"
    python3 train.py $p 
    
    done < $PARAM_FILE
