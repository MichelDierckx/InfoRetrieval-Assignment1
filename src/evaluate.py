import os
import pandas as pd
import numpy as np

def evaluate(result_file: str, expected_result_file: str, k: int):
    result = pd.read_csv(result_file)
    expected = pd.read_csv(expected_result_file)
    result_dict = dict()
    expected_dict = dict()
    for i, (_, row) in enumerate(result.iterrows()):
        if row["Query_number"] not in result_dict:
            result_dict[row["Query_number"]] = []
        result_dict[row["Query_number"]].append(row["doc_number"])
    for i, (_, row) in enumerate(expected.iterrows()):
        if row["Query_number"] not in expected_dict:
            expected_dict[row["Query_number"]] = []
        expected_dict[row["Query_number"]].append(row["doc_number"])
    evaluate_precision(expected_dict, result_dict, k)
    evaluate_recall(expected_dict, result_dict, k)

def evaluate_precision(result_dict: dict, expected_dict: dict, k: int):
    precision_list = []
    for key in result_dict.keys():
        relevant = []
        if key in expected_dict.keys():
            for i in range(1, min(k, len(expected_dict[key])) + 1):
                relevant_docs = get_amount_of_relevant_docs(result_dict, expected_dict, key, i)
                relevant.append(relevant_docs/i)
            precision_list.append(np.mean(relevant))
        else:
            precision_list.append(0)
    if len(precision_list) != 0:
        result = np.mean(precision_list)
        print(f'Mean average precision at {k}: {result}')
    else:
        print(f'Mean average precision at {k}: 0')
def evaluate_recall(result_dict: dict, expected_dict: dict, k: int):
    recall_list = []
    for key in result_dict.keys():
        if key in expected_dict.keys():
            relevant_docs = get_amount_of_relevant_docs(result_dict, expected_dict, key, k)
            recall_list.append(relevant_docs/len(expected_dict[key]))
        else:
            recall_list.append(0)
    if len(recall_list) != 0:
        result = np.mean(recall_list)
        print(f'Mean average recall at {k}: {result}')
    else:
        print(f'Mean average recall at {k}: 0')
def get_amount_of_relevant_docs(result_dict: dict, expected_dict: dict, key: str, k:int):
    expected_list = expected_dict[key]
    relevant_docs = 0
    for i in range(min(k, len(result_dict[key]))):
        result = result_dict[key][i]
        if result in expected_list:
            relevant_docs += 1
    return relevant_docs


if __name__ == "__main__":
    path = "results/rankings/"
    evaluate(path + "dev_queries_ranking.csv", path + "dev_query_results.csv", 5)

