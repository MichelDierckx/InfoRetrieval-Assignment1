import os
import pickle
from collections import Counter
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from natsort import natsorted

from tokenizer import Tokenizer


def extract_id_from_filename(filename):
    id_str = filename.split('_')[1]
    id_str = id_str.split('.')[0]
    return int(id_str)


class InvertedIndex:
    def __init__(self, doc_count: int):
        # Dictionary structure:
        #   - Key: str -> term
        #   - Value: Tuple[int, np.ndarray] -> (document frequency, structured array of postings)
        # The structured array contains:
        #   - 'document_id': int -> ID of the document
        #   - 'term_frequency': int -> Frequency of the term in the document

        self.index: Dict[str, Tuple[int, np.ndarray]] = {}  # Inverted index
        self.doc_count: int = doc_count  # Total number of documents
        self.doc_lengths: np.ndarray = np.zeros(doc_count,
                                                dtype=np.float64)  # Array to save calculated document lengths

    def add_document(self, doc_id: int, tokens: List[str]):
        for term in tokens:
            if term not in self.index:
                structured_array = np.zeros(1, dtype=[('document_id', 'i4'), ('term_frequency', 'i4')])
                structured_array[0] = (doc_id, 1)
                self.index[term] = (1, structured_array)
            else:
                doc_freq, structured_array = self.index[term]
                last_entry = structured_array[-1]
                if last_entry['document_id'] == doc_id:
                    last_entry['term_frequency'] += 1
                else:
                    new_entry = np.array([(doc_id, 1)], dtype=structured_array.dtype)
                    structured_array = np.append(structured_array, new_entry)
                    self.index[term] = (doc_freq + 1, structured_array)

    def calculate_idf_weight(self, term: str) -> float:
        """Calculate idf_weight given a term"""
        doc_freq, _ = self.index.get(term, (0, None))
        if doc_freq == 0:
            return 0.0
        return np.log(self.doc_count / doc_freq)

    def calculate_document_lengths(self) -> None:
        """
        Calculate document lengths.
        """
        # reset doc_lengths array to 0 array
        self.doc_lengths.fill(0)

        # loop through index
        for term, (doc_freq, postings) in self.index.items():
            # calculate idf_weight
            idf_weight = np.log(self.doc_count / doc_freq)

            # for every document in the posting list
            for entry in postings:
                doc_id = entry['document_id']
                tf = entry['term_frequency']

                # calculate tf_weight
                tf_weight = 1 + np.log(tf)

                # calculate tf_idf_weight
                tf_idf_weight = tf_weight * idf_weight

                # sum the squares of tf-idf weights
                self.doc_lengths[doc_id - 1] += tf_idf_weight ** 2  # Use doc_id - 1 as the array index

        # take the square root, result is vector length
        np.sqrt(self.doc_lengths, out=self.doc_lengths)

    def save(self, index_filename: str, lengths_filename: str):
        """save the inverted index to index_filename and save the document vector lengths to lengths_filename."""
        # Save the document lengths (using numpy)
        np.save(lengths_filename, self.doc_lengths)
        print(f"Document lengths saved to {lengths_filename}")

        # Save the index using pickle
        with open(index_filename, 'wb') as f:
            pickle.dump({'index': self.index, 'doc_count': self.doc_count}, f)
        print(f"Inverted index saved to {index_filename}")

    @classmethod
    def load(cls, index_filename: str, lengths_filename: str):
        """load the index from index_filename and load the document vector lengths from lengths_filename."""
        index_instance = cls(0)  # get InvertedIndex class instance with document count 0

        # Load the document lengths
        index_instance.doc_lengths = np.load(lengths_filename)

        # document count is number of entries in the doc_lengths array
        index_instance.doc_count = index_instance.doc_lengths.shape[0]

        # load the index using pickle
        with open(index_filename, 'rb') as f:
            data = pickle.load(f)
            index_instance.index = data['index']
            index_instance.doc_count = data['doc_count']  # Ensure the document count is set

        print(f"Inverted index loaded from {index_filename}")
        print(f"Document lengths loaded from {lengths_filename}")
        return index_instance

    def print_index(self):
        """print inverted index."""
        for term, (df, arr) in self.index.items():
            print(f"Term: '{term}', Document Frequency: {df}, Entries: {arr.tolist()}")

    def print_posting_list(self, term: str):
        """print posting list for a given term"""
        if term in self.index:
            df, arr = self.index[term]
            print(f"Posting list for term: '{term}'")
            print(f"Document Frequency: {df}")
            print("Postings (Document ID, Term Frequency):")
            for entry in arr:
                print(f"  Document ID: {entry['document_id']}, Term Frequency: {entry['term_frequency']}")
        else:
            print(f"Term: '{term}' not found in the index.")


class Indexer:
    def __init__(self, tokenizer: Tokenizer, save_tokenization: bool = False, load_tokenization: bool = False,
                 token_cache_directory: Optional[str] = None):
        self.tokenizer = tokenizer
        self.save_tokenization = save_tokenization
        self.load_tokenization = load_tokenization
        self.token_cache_directory = token_cache_directory or 'data/tokenized_documents/full_docs_small'

        # Create the token cache directory if tokenization is saved
        if self.save_tokenization or self.load_tokenization:
            os.makedirs(self.token_cache_directory, exist_ok=True)

    def _save_tokenized_document(self, document_id: int, tokens: List[str]) -> None:
        """Saves tokenized output to cache as bytes if enabled."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"  # Changed to .pkl
        with open(cache_file, 'wb') as f:  # Open in binary write mode
            pickle.dump(tokens, f)  # Use pickle to serialize tokens

    def _load_tokenized_document(self, document_id: int) -> Optional[List[str]]:
        """Loads tokenized output from cache if available."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"  # Changed to .pkl
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:  # Open in binary read mode
                return pickle.load(f)  # Use pickle to deserialize tokens
        return None

    def create_index_from_directory(self, directory: str) -> InvertedIndex:
        print(f'Creating positional index for directory: {directory}')

        # List all text files in the directory (sorted alphabetically)
        files = natsorted(os.listdir(directory))
        documents = [file for file in files if file.endswith(".txt")]
        inverted_index = InvertedIndex(len(documents))
        for document in documents:
            document_id = extract_id_from_filename(document)

            # tokenize (load previous tokenization if flag is set to true and available)
            if self.load_tokenization:
                tokens = self._load_tokenized_document(document_id) or self.tokenizer.tokenize(
                    open(f'{directory}/{document}', 'r').read())
            else:
                tokens = self.tokenizer.tokenize(open(f'{directory}/{document}', 'r').read())

            # save results of tokenization if flag is set to true
            if self.save_tokenization:
                self._save_tokenized_document(document_id, tokens)

            inverted_index.add_document(document_id, tokens)
        inverted_index.calculate_document_lengths()
        return inverted_index


class DocumentRanker:
    def __init__(self, tokenizer: Tokenizer, inverted_index: InvertedIndex):
        self.tokenizer = tokenizer
        self.inverted_index = inverted_index

    def rank_documents(self, query: str) -> List[Tuple[int, float]]:
        """
        Rank documents based on the query. Makes use of smart components ltc.ltc.
        :param query: The search query
        :return: A list of tuples (document_id, score) sorted by score in descending order.
        """

        query_vector = self.get_query_vector(query)  # normalized query vector
        accumulators = Counter()  # hold accumulated score for every relevant document

        # loop over terms in query
        for term, tf_idf_query in query_vector.items():
            # calculate idf_weight
            doc_freq, postings_list = self.inverted_index.index[term]
            idf_weight_doc = np.log(self.inverted_index.doc_count / doc_freq)
            # loop over posting
            for posting in postings_list:
                doc_id = posting['document_id']
                tf_doc = posting['term_frequency']
                doc_length = self.inverted_index.doc_lengths[doc_id - 1]

                # calculate tf_weight
                tf_weight_doc = 1 + np.log(tf_doc)

                # calculate tf_idf_weight
                tf_idf_doc = (tf_weight_doc * idf_weight_doc) / doc_length

                # update score for doc id
                accumulators[doc_id] += tf_idf_doc * tf_idf_query

        # sort the documents by score (descending)
        sorted_doc_scores = accumulators.most_common()  # sorted [(doc_id, score), ...]
        return sorted_doc_scores

    def get_query_vector(self, query) -> Dict[str, float]:
        query_tokens = self.tokenizer.tokenize(query)
        term_frequencies = Counter(query_tokens)
        sum_of_tf_idf_squared = 0
        query_vector = {}

        for query_token in term_frequencies.keys():
            if query_token in self.inverted_index.index.keys():
                tf_weight = 1 + np.log(term_frequencies[query_token])
                idf_weight = np.log(self.inverted_index.doc_count / self.inverted_index.index[query_token][0])

                # calculate tf_idf_weight
                tf_idf_weight = tf_weight * idf_weight
                query_vector[query_token] = tf_idf_weight

                # sum the squares of tf-idf weights
                sum_of_tf_idf_squared += tf_idf_weight ** 2
        query_vector_length = np.sqrt(sum_of_tf_idf_squared)
        # normalize query vector
        if query_vector_length > 0:
            for term in query_vector:
                query_vector[term] /= query_vector_length
        return query_vector

    def rank_queries_from_file(self, input_file: str, output_file: str, delimiter: str = ',',
                               top_k: Optional[int] = None) -> None:
        """
        Reads queries from a csv file and generates a ranking for them.

        :param input_file: Path to the input CSV or TSV file with queries.
        :param output_file: Path to the output file where rankings will be saved.
        :param delimiter: The character used to separate values in the input file (default is ',').
        :param top_k: How many top ranked documents to save in the output file; if None, saves all.
        """
        queries_df = pd.read_csv(input_file, delimiter=delimiter)
        with open(output_file, 'w') as f:
            # loop over queries
            for _, row in queries_df.iterrows():
                query_number = row['Query number']
                query_text = row['Query']

                # get the ranked documents for the current query
                ranked_documents = self.rank_documents(query_text)

                # only the top k should be returned
                if top_k is not None:
                    ranked_documents = ranked_documents[:top_k]

                # write to file
                for doc_id, _ in ranked_documents:
                    f.write(f"{query_number},{doc_id}\n")

        print(f"Rankings saved to {output_file}")
