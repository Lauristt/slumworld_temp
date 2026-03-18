#!/home/qigong/venvs/torchenv/bin/python3
# import libraries
import os
import sys
import numpy as np
import json
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, jaccard_score
import click 
import yaml
from tqdm import tqdm
from multiprocessing import Pool
import pdb

def process_files(args):
    true_file, pred_file, threshold, true_path, pred_path = args
    true_file_path = os.path.join(true_path, true_file)
    pred_file_path = os.path.join(pred_path, pred_file)
    return true_file, evaluate_model(true_file_path, pred_file_path, threshold)


# create a function that will receive the paths to two image-mask (probability between 0-1) 
# and a threshold value and will  return a dictionary with the following metrics: 
# accuracy, recall, f1-score, precision, specificity, jaccard, dice, roc-auc, pr-auc, and confusion matrix
def evaluate_model(y_true_filename, y_pred_filename, threshold=0.5):
    y_true = np.array(Image.open(y_true_filename))
    y_pred = np.array(Image.open(y_pred_filename))
    if len(y_pred.shape) > 2:
        y_pred = y_pred[:,:,0]
    y_pred = np.where(y_pred > threshold, 1, 0)
    y_true = y_true.reshape(y_pred.shape)
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(),labels=[0,1])
    acc = accuracy_score(y_true.flatten(), y_pred.flatten())
    recall = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=1)
    precision = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=1)
    try:
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) 
    except Exception as Error:
        print(Error)
        print(y_true_filename)
        print(y_true[0])
        print(y_pred[0])
        print(cm)
        pdb.set_trace()
        specificity = 0
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=1)
    jaccard = jaccard_score(y_true.flatten(), y_pred.flatten(), zero_division=1)

    results = {'accuracy': acc,
               'recall': recall,
               'precision': precision,
               'specificity': specificity,
               'f1-score': f1,
               'jaccard': jaccard,
               'confusion_matrix': cm.tolist()}
    return results
# write a function that will receive a dictionary of result dictionaries and will calculate the total of each metric by summing the elements of each confusion matrix and then calculating the metrics
def calculate_total_metrics(results):
    ''''''
    # create a dictionary to store the total metrics
    total_metrics = {}
    # loop through the results dictionary and sum the confusion matrices
    total_cm = np.zeros((2,2))
    for k, v in results.items():
        total_cm += v['confusion_matrix']
    # compute accuracy  
    acc = np.sum(np.diag(total_cm))/np.sum(total_cm)
    # compute recall
    recall = total_cm[1,1]/(total_cm[1,0]+total_cm[1,1])
    # compute precision
    precision = total_cm[1,1]/(total_cm[0,1]+total_cm[1,1])
    # compute specificity
    specificity = total_cm[0,0]/(total_cm[0,0]+total_cm[0,1])
    # compute f1-score
    f1 = 2*precision*recall/(precision+recall)
    # compute jaccard
    jaccard = total_cm[1,1]/(total_cm[0,1]+total_cm[1,0]+total_cm[1,1])
    # create results dictionary
    total_metrics = {'accuracy': acc,
               'recall': recall,
               'precision': precision,
               'specificity': specificity,
               'f1-score': f1,
               'iou': jaccard,
               'confusion_matrix': total_cm[::-1,::-1].tolist()}
    return total_metrics

# lets convert this script into a command line tool
@click.command()
@click.option('-c', '--config_file', default=None, help='Path to the configuration file')
@click.option('--processes', default=10, help='Number of processes to use in parallel (for accelerated execution).')
@click.option('--true_path', default=None, help='Path to the ground truth data.')
@click.option('--pred_path', default=None, help= 'Path to the predicted data.')
@click.option('--threshold', default=0.5, help='Threshold value for computing metrics.')
@click.option('--results_file', default=None, help='Path to the output results file (file where results are stored). If not supplied, a file will be created.')
@click.help_option('-h', '--help')
def main(config_file, processes, true_path, pred_path, threshold, results_file):
    ''' 
    '''
    if config_file:
        if os.path.exists(config_file):
            with open(config_file, 'r') as ymlfile:
                cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            true_path = cfg['true_path']
            pred_path = cfg['pred_path']
            threshold = cfg['threshold']
            results_file = cfg['results_file']
        else:
            print("Configuration file not found")
            return
    if results_file is None:
        results_file = f"evaluation_results_for_{true_path.replace('/','-').replace(' ','_')}_threshold_{threshold}.json"
    true_label_files = sorted(os.listdir(true_path))
    pred_label_files = sorted(os.listdir(pred_path))
    # loop through all images in the true_path and pred_path, compute the metrics for each image and store the results in json file
    results = {}
    print("Starting evaluation")
    with Pool(processes) as p:
        results = dict(p.map(process_files, zip(true_label_files, pred_label_files, 
                                                [threshold]*len(true_label_files),
                                                [true_path]*len(true_label_files),
                                                [pred_path]*len(true_label_files))))
    # calculate total metrics
    total_metrics = calculate_total_metrics(results)
    # pretify and print the results in a nice format in the terminal using tabulate
    print('Total metrics:')
    print('--------------')
    for k, v in total_metrics.items():
        print(f'{k}: {v}')
    print('--------------')
    # if the rdirectory of the esults_path does not exist, create it
    if not os.path.exists(os.path.dirname(results_file)):
        os.makedirs(os.path.dirname(results_file))
    # save results to json files
    with open(results_file.replace('.json','_total.json'), 'w') as f:
        json.dump(total_metrics, f, indent=4)
    with open(results_file, 'w') as f:
        json.dump(results, f)
    sys.exit(0)

if __name__ == '__main__':
    main()