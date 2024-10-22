import math


def calculate_tf_weight(tf_doc: int):
    tf_weight = 1 + math.log(tf_doc, 10)
    return tf_weight


def calculate_idf_weight(total_docs: int, df: int):
    idf_weight = math.log((total_docs / df), 10)
    return idf_weight


def calculate_tf_idf_weight(tf_doc: int, total_docs: int, df: int):
    return calculate_tf_weight(tf_doc) * calculate_idf_weight(total_docs, df)
