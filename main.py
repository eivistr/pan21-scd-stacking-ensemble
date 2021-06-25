import json
import pickle
import argparse
import os
import sys
import numpy as np
import time

from generate_embeddings import generate_embeddings
from generate_text_features import generate_features
from utilities import load_documents, task2_parchange_predictions, task3_binary_predictions, task3_authorship_predictions


TASK1_MODEL = os.path.join(sys.path[0], "saved_models/task1_ensemble_78.pickle")
TASK2_MODEL = os.path.join(sys.path[0], "saved_models/task2_ensemble_71.pickle")
TASK3_MODEL = os.path.join(sys.path[0], "saved_models/task3_ensemble_72.pickle")


def typeconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


def main(data_folder, output_folder):

    start_time = time.time()

    # Load documents
    docs, doc_ids = load_documents(data_folder)
    print(f"Loaded {len(docs)} documents ...")

    # Generate document and paragraph features
    doc_emb, par_emb = generate_embeddings(docs)
    doc_textf, par_textf = generate_features(docs)

    # Task 1
    print("Task 1 predictions ...")
    task1_ensemble = pickle.load(open(TASK1_MODEL, "rb"))
    task1_preds_proba = task1_ensemble.predict_proba([doc_emb, doc_textf])
    task1_preds = np.round(task1_preds_proba)
    del task1_ensemble, doc_emb, doc_textf

    # Task 2
    print("Task 2 predictions ...")
    task2_ensemble = pickle.load(open(TASK2_MODEL, "rb"))
    task2_preds = task2_parchange_predictions(task2_ensemble, par_emb, par_textf)
    del task2_ensemble

    # Task 3
    print("Task 3 predictions ...")
    task3_ensemble = pickle.load(open(TASK3_MODEL, "rb"))
    task3_binary_preds = task3_binary_predictions(task1_preds_proba, task3_ensemble, par_emb, par_textf)
    task3_preds = task3_authorship_predictions(task1_preds_proba, task3_binary_preds, par_emb, par_textf)
    del task1_preds_proba, task3_binary_preds, task3_ensemble

    # Save solutions
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(task1_preds)):
        solution = {
            'multi-author': task1_preds[i],
            'changes': task2_preds[i],
            'paragraph-authors': task3_preds[i]
        }

        file_name = r'solution-problem-' + str(i + 1) + '.json'
        with open(os.path.join(output_folder, file_name), 'w') as file_handle:
            json.dump(solution, file_handle, default=typeconverter)

    print(f"Run finished after {(time.time() - start_time) / 60:0.2f} minutes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAN21 Style Change Detection software submission')
    parser.add_argument("-i", "--input_dir", help="path to the dir holding the data", required=True)
    parser.add_argument("-o", "--output_dir", help="path to the dir to write the results to", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)

