import os
import pickle
from collections import Counter
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from natsort import natsorted
from scipy.sparse import save_npz, load_npz, lil_matrix

from tokenizer import Tokenizer


def extract_id_from_filename(filename):
    id_str = filename.split('_')[1]
    id_str = id_str.split('.')[0]
    return int(id_str)


class InvertedIndex:
    def __init__(self, term_count, doc_count: int, term_to_id: Dict[str, int]):
        # Total number of documents
        self.doc_count: int = doc_count
        self.term_count: int = term_count  # To keep track of the number of unique terms
        self.term_to_id: Dict[str, int] = term_to_id  # Mapping from term to its index in the sparse matrix

        # Initialize the term frequency matrix (term_count, doc_count)
        self.term_frequency_matrix = lil_matrix((term_count, doc_count), dtype=np.int16)
        self.doc_lengths: np.ndarray = np.zeros(doc_count, dtype=np.float32)  # Document lengths

    def add_document(self, doc_id: int, tokens: List[str]):
        for term in tokens:
            term_index = self.term_to_id[term]
            self.term_frequency_matrix[term_index, doc_id - 1] += 1  # Update frequency

    def finalize_index(self):
        # Convert term frequency matrix to csr_matrix for efficient calculations
        self.term_frequency_matrix = self.term_frequency_matrix.tocsr()

    def calculate_document_lengths(self) -> None:
        """
        Calculate document lengths.
        """
        self.doc_lengths = np.zeros(self.doc_count, dtype=np.float64)  # Reset lengths

        # iterate over rows
        for term_index in range(self.term_frequency_matrix.shape[0]):
            # pointers to start and end of row
            start, end = self.term_frequency_matrix.indptr[term_index], self.term_frequency_matrix.indptr[
                term_index + 1]
            doc_indices = self.term_frequency_matrix.indices[start:end]  # indices for nonzero entries
            tf_values = self.term_frequency_matrix.data[start:end]  # term frequency values

            # Skip terms with zero document frequency
            doc_freq = len(doc_indices)

            # Calculate IDF weight for the term
            idf_weight = np.log(self.doc_count / doc_freq)

            # Calculate TF weights (1 + log(tf)) and squared TF-IDF weights
            tf_weights = 1 + np.log(tf_values)
            tf_idf_weights = (tf_weights * idf_weight) ** 2

            # Accumulate squared tf-idf weights into document lengths
            np.add.at(self.doc_lengths, doc_indices, tf_idf_weights)

        # Take the square root to get the final vector lengths
        np.sqrt(self.doc_lengths, out=self.doc_lengths)

    def save(self, index_filename: str, lengths_filename: str, term_frequency_filename: str):
        """Save the inverted index and other data."""
        # Save document lengths using numpy
        np.save(lengths_filename, self.doc_lengths)
        print(f"Document lengths saved to {lengths_filename}")

        # Save term frequency matrix using scipy
        save_npz(term_frequency_filename, self.term_frequency_matrix)
        print(f"Term frequency matrix saved to {term_frequency_filename}")

        self.save_index(term_frequency_filename)
        # Save metadata using pickle
        with open(index_filename, 'wb') as f:
            pickle.dump({
                'index': self.term_to_id,
                'doc_count': self.doc_count,
                'term_count': self.term_count
            }, f)
        print(f"Inverted index saved to {index_filename}")

    def save_index(self, term_frequency_filename):
        # Save term frequency matrix using scipy
        save_npz(term_frequency_filename, self.term_frequency_matrix)

    @classmethod
    def load(cls, index_filename: str, lengths_filename: str, term_frequency_filename: str):
        """Load the index and other data."""
        # Create an instance with dummy values
        index_instance = cls(0, 0, {})

        # Load document lengths using numpy
        index_instance.doc_lengths = np.load(lengths_filename)
        index_instance.doc_count = index_instance.doc_lengths.shape[0]

        # Load term frequency matrix using scipy
        index_instance.term_frequency_matrix = load_npz(term_frequency_filename)
        index_instance.term_count = index_instance.term_frequency_matrix.shape[0]  # Number of unique terms

        # Load metadata using pickle
        with open(index_filename, 'rb') as f:
            data = pickle.load(f)
            index_instance.term_to_id = data['index']
            index_instance.doc_count = data['doc_count']
            index_instance.term_count = data['term_count']

        print(f"Inverted index loaded from {index_filename}")
        print(f"Document lengths loaded from {lengths_filename}")
        print(f"Term frequency matrix loaded from {term_frequency_filename}")

        return index_instance

    @classmethod
    def load_from_partial(cls, partial_index_directory: str, doc_count: int, term_count: int,
                          term_to_id: Dict[int, str]):
        # List all .npz files in the specified directory
        index_instance = cls(0, 0, {})
        index_instance.doc_count = doc_count
        index_instance.term_count = term_count
        index_instance.term_to_id = term_to_id
        # index_instance.term_frequency_matrix = csr_matrix((1, 1), dtype=np.int16)
        # index_instance.term_frequency_matrix.tocsr()

        csr_files = [f for f in os.listdir(partial_index_directory) if f.endswith('.npz')]
        nr_csr_files = len(csr_files)

        index_instance.term_frequency_matrix = load_npz(os.path.join(partial_index_directory, csr_files[0]))

        for i, csr_file in enumerate(csr_files[1:]):
            crs_matrix = load_npz(os.path.join(partial_index_directory, csr_files[i + 1]))
            index_instance.term_frequency_matrix = index_instance.term_frequency_matrix + crs_matrix
            if (i + 1) % 500 == 0:
                print(f'Merged {i + 1}/{nr_csr_files} partial indexes...')
        return index_instance


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

    def create_index_from_directory(self, directory: str, partial_index_directory: str,
                                    batch_size: int = 1000) -> InvertedIndex:
        print(f'Creating positional index for directory: {directory}')

        # List all text files in the directory (sorted alphabetically)
        files = natsorted(os.listdir(directory))
        documents = [file for file in files if file.endswith(".txt")]
        nr_documents = len(documents)

        term_to_id = {}
        term_counter = 0

        # Process each document and build the inverted index
        for i, document in enumerate(documents):
            document_id = extract_id_from_filename(document)

            # Tokenize (load previous tokenization if the flag is set to true and available)
            if self.load_tokenization:
                tokens = self._load_tokenized_document(document_id) or self.tokenizer.tokenize(
                    open(f'{directory}/{document}', 'r').read())
            else:
                tokens = self.tokenizer.tokenize(open(f'{directory}/{document}', 'r').read())

            # Save results of tokenization if the flag is set to true
            if self.save_tokenization:
                self._save_tokenized_document(document_id, tokens)

            for token in tokens:
                if token not in term_to_id:
                    term_to_id[token] = term_counter
                    term_counter += 1

            # status update
            if (i + 1) % 10000 == 0:
                print(f'Collected terms from {i + 1}/{nr_documents} documents...')

        print(f'Creating partial indexes...')
        # Process each document and build the inverted index in batches
        for i in range(0, nr_documents, batch_size):
            partial_inverted_index = InvertedIndex(term_counter, nr_documents, term_to_id)
            for j in range(i, min(i + batch_size, nr_documents)):
                document = documents[j]
                document_id = extract_id_from_filename(document)
                tokens = self._load_tokenized_document(document_id)
                partial_inverted_index.add_document(document_id, tokens)
                # status update
                if (j + 1) % 10000 == 0:
                    print(f'Processed {j + 1}/{nr_documents} documents...')
            partial_inverted_index.finalize_index()
            partial_inverted_index.save_index(f'{partial_index_directory}/{i}.npz')

        inverted_index = InvertedIndex.load_from_partial(partial_index_directory, nr_documents, term_counter,
                                                         term_to_id)
        print(f'Calculating document vector lengths...')
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
        for term_id, tf_idf_query in query_vector.items():
            # calculate idf_weight
            doc_freq = self.inverted_index.term_frequency_matrix[term_id, :].count_nonzero()
            idf_weight_doc = np.log(self.inverted_index.doc_count / doc_freq)

            # Get the term frequencies for the term across all documents
            tf_doc = self.inverted_index.term_frequency_matrix[term_id, :].toarray().flatten()  # Convert to 1D array
            doc_length = self.inverted_index.doc_lengths  # Document lengths array

            # Loop over all documents and calculate scores
            for doc_id in range(len(tf_doc)):
                tf_doc_value = tf_doc[doc_id]
                if tf_doc_value > 0:  # Only consider documents that contain the term
                    # Calculate tf_weight
                    tf_weight_doc = 1 + np.log(tf_doc_value)

                    # Calculate tf_idf_weight
                    tf_idf_doc = (tf_weight_doc * idf_weight_doc) / doc_length[doc_id]

                    # Update score for doc id
                    accumulators[doc_id + 1] += tf_idf_doc * tf_idf_query

        # Sort the documents by score (descending)
        sorted_doc_scores = accumulators.most_common()  # sorted [(doc_id, score), ...]
        return sorted_doc_scores

    def get_query_vector(self, query) -> Dict[int, float]:
        query_tokens = self.tokenizer.tokenize(query)
        term_frequencies = Counter(query_tokens)
        sum_of_tf_idf_squared = 0
        query_vector = {}

        for query_token in term_frequencies.keys():
            if query_token in self.inverted_index.term_to_id.keys():
                tf_weight = 1 + np.log(term_frequencies[query_token])

                term_id = self.inverted_index.term_to_id[query_token]
                doc_freq = self.inverted_index.term_frequency_matrix[term_id, :].count_nonzero()
                idf_weight = np.log(self.inverted_index.doc_count / doc_freq)

                # calculate tf_idf_weight
                tf_idf_weight = tf_weight * idf_weight
                query_vector[term_id] = tf_idf_weight

                # sum the squares of tf-idf weights
                sum_of_tf_idf_squared += tf_idf_weight ** 2
        query_vector_length = np.sqrt(sum_of_tf_idf_squared)
        # normalize query vector
        if query_vector_length > 0:
            for term_id in query_vector:
                query_vector[term_id] /= query_vector_length
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
            f.write("Query_number,doc_number\n")
            # loop over queries
            for i, (_, row) in enumerate(queries_df.iterrows()):
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

                # status update
                if (i + 1) % 500 == 0:
                    print(f"Processed {i + 1} queries...")
        print(f"Processed all queries...")
        print(f"Rankings saved to {output_file}")
