import os
import pickle
from typing import List, Optional, Tuple, Dict

import numpy as np
from natsort import natsorted

from tokenizer import Tokenizer


# save document vector lengths in a (sorted numpy) array


# Query processing:

# tokenize query

# see slide 28


def extract_id_from_filename(filename):
    id_str = filename.split('_')[1]
    id_str = id_str.split('.')[0]
    return int(id_str)


class InvertedIndex:
    def __init__(self):
        # Dictionary structure:
        #   - Key: str -> term
        #   - Value: Tuple[int, np.ndarray] -> (document frequency, structured array of postings)
        # The structured array contains:
        #   - 'document_id': int -> ID of the document
        #   - 'term_frequency': int -> Frequency of the term in the document

        self.index: Dict[str, Tuple[int, np.ndarray]] = {}  # Inverted index
        self.doc_count: int = 0  # Total number of documents

    def add_document(self, doc_id: int, tokens: List[str]):
        self.doc_count += 1  # Increment document count
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
        """Calculate IDF for a term."""
        doc_freq, _ = self.index.get(term, (0, None))
        if doc_freq == 0:
            return 0.0
        return np.log(self.doc_count / doc_freq)

    def calculate_document_lengths(self) -> np.ndarray:
        """
        Calculates the vector length for each document based on the tf-idf weight of each term
        and returns a NumPy array of document lengths.
        """
        # Preallocate a NumPy array for document lengths, using document_count as size
        doc_lengths = np.zeros(self.doc_count, dtype=np.float64)

        # Loop through each term and its posting list in the index
        for term, (doc_freq, postings) in self.index.items():
            # Calculate the IDF for the term
            idf_weight = np.log(self.doc_count / doc_freq)

            # Process each document in the posting list
            for entry in postings:
                doc_id = entry['document_id']
                tf = entry['term_frequency']

                # Calculate the term frequency weight (tf_weight)
                tf_weight = 1 + np.log(tf)

                # Calculate the tf-idf weight
                tf_idf_weight = tf_weight * idf_weight

                # Sum the squares of tf-idf weights for each document
                doc_lengths[doc_id - 1] += tf_idf_weight ** 2  # Use doc_id - 1 as the array index

        # Take the square root of the sum of squares to get the vector length
        doc_lengths = np.sqrt(doc_lengths)

        return doc_lengths

    def save(self, filename: str):
        """Save the inverted index to a file."""
        with open(filename, 'wb') as f:
            # Save both the index and document count
            pickle.dump({'index': self.index, 'doc_count': self.doc_count}, f)
        print(f"Inverted index saved to {filename}")

    @classmethod
    def load(cls, filename: str):
        """Load the inverted index from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            index_instance = cls()
            index_instance.index = data['index']
            index_instance.doc_count = data['doc_count']
        print(f"Inverted index loaded from {filename}")
        return index_instance

    def print_index(self):
        """Prints the inverted index for visualization."""
        for term, (df, arr) in self.index.items():
            print(f"Term: '{term}', Document Frequency: {df}, Entries: {arr.tolist()}")

    def print_posting_list(self, term: str):
        """Prints the entries for a specific term."""
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
    def __init__(self, save_tokenization: bool = False, token_cache_directory: Optional[str] = None):
        self.tokenizer = Tokenizer()
        self.save_tokenization = save_tokenization
        self.token_cache_directory = token_cache_directory or 'data/tokenized_documents/full_docs_small'

        # Create the token cache directory if tokenization is saved
        if self.save_tokenization:
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
        inverted_index = InvertedIndex()

        # List all text files in the directory (sorted alphabetically)
        files = natsorted(os.listdir(directory))
        documents = [file for file in files if file.endswith(".txt")]
        for document in documents:
            document_id = extract_id_from_filename(document)
            tokens = self._load_tokenized_document(document_id) or self.tokenizer.tokenize(
                open(f'{directory}/{document}', 'r').read())

            inverted_index.add_document(document_id, tokens)
        return inverted_index
