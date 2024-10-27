import os
import pickle
import sys
from typing import List, Optional

import numpy as np
from natsort import natsorted

from tokenizer import Tokenizer


def extract_id_from_filename(filename):
    id_str = filename.split('_')[1]
    id_str = id_str.split('.')[0]
    return int(id_str)


class InvertedIndex:
    def __init__(self):
        # Dictionary to hold the inverted index
        # Key: term, Value: (document frequency, structured numpy array)
        self.index = {}

    def add_document(self, doc_id: int, tokens: List[str]):
        """Tokenizes the text and updates the inverted index."""
        # Update the inverted index
        for term in tokens:
            if term not in self.index:
                # Create a structured array for the term
                structured_array = np.zeros(1, dtype=[('document_id', 'i4'), ('term_frequency', 'i4')])
                structured_array[0] = (doc_id, 1)  # Initialize term frequency to 1
                self.index[term] = (1, structured_array)  # (document frequency, structured array)
            else:
                doc_freq, structured_array = self.index[term]

                # Check only the last entry in the structured array
                last_entry = structured_array[-1]
                if last_entry['document_id'] == doc_id:
                    last_entry['term_frequency'] += 1  # Increment term frequency
                else:
                    # If the document ID does not exist, append a new entry
                    new_entry = np.array([(doc_id, 1)], dtype=structured_array.dtype)
                    structured_array = np.append(structured_array, new_entry)
                    self.index[term] = (doc_freq + 1, structured_array)  # Increment document frequency

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
        print("Size of dict: " + str(sys.getsizeof(inverted_index.index)) + "bytes")
        return inverted_index
