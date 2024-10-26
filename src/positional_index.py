import json
import math
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional

from custom_types import Term, DocID
from document_id_mapper import DocumentIDMapper
from tf_idf import calculate_tf_idf_weight
from tokenizer import Tokenizer


class Posting:
    def __init__(self):
        self.tf = 0  # Term frequency
        self.tfidf = 0.0  # normalized TF-IDF
        self.positions: List[int] = []  # List of positions in the document

    def update(self, term_position: int) -> None:
        """
        Update the Posting by adding a new position and incrementing the term frequency.
        """
        self.positions.append(term_position)
        self.tf += 1

    def calculate_tfidf(self, total_docs: int, df: int) -> None:
        self.tfidf = calculate_tf_idf_weight(self.tf, total_docs, df)

    def to_list(self) -> List:
        """
        Convert Posting object to a List[tf, tfidf, List[position]]
        """
        return [self.tf, self.tfidf, self.positions]

    @staticmethod
    def from_list(data: List):
        """
        Create a Posting object from a List[tf, tfidf, List[position]].
        """
        posting = Posting()
        posting.tf = data[0]
        posting.tfidf = data[1]
        posting.positions = data[2]
        return posting

    def pretty_print(self) -> None:
        """
        Pretty print the Posting
        """
        print(self.to_list())


class PostingsList:
    def __init__(self):
        self.df = 0
        self.postings: Dict[DocID, Posting] = {}

    def update(self, document_id: DocID, term_position: int):
        if document_id in self.postings:
            self.postings[document_id].update(term_position)
        else:
            self.postings[document_id] = Posting()
            self.postings[document_id].update(term_position)
            self.df += 1

    def calculate_tfidf(self, total_docs: int) -> None:
        for posting in self.postings.values():
            posting.calculate_tfidf(total_docs, self.df)

    def to_dict(self) -> Dict:
        """
        Convert PostingsList object to a dictionary.
        """
        return {
            'df': self.df,
            'postings': {doc_id: posting.to_list() for doc_id, posting in self.postings.items()}
        }

    @staticmethod
    def from_dict(data: Dict):
        """
        Create a PostingsList object from a dictionary.
        """
        postings_list = PostingsList()
        postings_list.df = data['df']
        postings_list.postings = {
            int(doc_id): Posting.from_list(posting) for doc_id, posting in data['postings'].items()
        }
        return postings_list

    def to_list(self) -> List:
        return [
            self.df,  # first element is df
            {doc_id: posting.to_list() for doc_id, posting in self.postings.items()}
        ]

    @staticmethod
    def from_list(data: List):
        postings_list = PostingsList()
        postings_list.df = data[0]  # first element is df
        postings_list.postings = {
            int(doc_id): Posting.from_list(posting) for doc_id, posting in data[1].items()
        }
        return postings_list

    def pretty_print(self) -> None:
        """
        Pretty print the PostingsList
        """
        print(json.dumps(self.to_dict(), indent=4))


class PositionalIndex:
    def __init__(self, document_id_mapper: Optional[DocumentIDMapper]):
        self.document_id_mapper = document_id_mapper
        self.positional_index: Dict[Term, PostingsList] = defaultdict(PostingsList)

    def add_posting(self, term: Term, document_id: DocID, term_position: int):
        self.positional_index[term].update(document_id, term_position)

    def calculate_tfidf(self):
        if self.document_id_mapper is None:
            return
        total_docs = self.document_id_mapper.total_docs
        print('Calculating tf-idf weights...')
        for postings_list in self.positional_index.values():
            postings_list.calculate_tfidf(total_docs)
        print('Normalizing tf-idf weights...')
        self.normalize_tfidf()

    def get_doc_lengths(self) -> Dict[DocID, float]:
        doc_lengths: Dict[DocID, float] = {}

        for postings_list in self.positional_index.values():
            for document_id, posting in postings_list.postings.items():
                doc_lengths.setdefault(document_id, 0)  # if not present, first init to zero
                doc_lengths[document_id] += posting.tfidf ** 2

        for document_id, doc_length in doc_lengths.items():
            doc_lengths[document_id] = math.sqrt(doc_length)

        return doc_lengths

    def normalize_tfidf(self):
        doc_lengths = self.get_doc_lengths()
        for postings_list in self.positional_index.values():
            for document_id, posting in postings_list.postings.items():
                posting.tfidf = posting.tfidf / doc_lengths[document_id]

    def to_dict(self) -> Dict:
        if self.document_id_mapper is None:
            return {
                'index': {term: postings.to_dict() for term, postings in self.positional_index.items()}
            }
        return {
            'document_id_mapper': self.document_id_mapper.to_dict(),
            'index': {term: postings.to_dict() for term, postings in self.positional_index.items()}
        }

    @staticmethod
    def from_dict(data: Dict):
        if 'document_id_mapper' in data.keys():
            document_id_mapper = DocumentIDMapper.from_dict(
                data['document_id_mapper'])
        else:
            document_id_mapper = None
        index = PositionalIndex(document_id_mapper)
        index.positional_index = {
            term: PostingsList.from_dict(postings_data) for term, postings_data in data['index'].items()
        }
        return index

    @staticmethod
    def load_from_file(filename: str):
        """
        Load the positional index from a file (binary using pickle).
        """
        if not os.path.exists(filename):
            print(f"Error: The file '{filename}' does not exist.")
            return None

        try:
            print(f'Loading positional index from {filename}')
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f'Successfully loaded positional index from {filename}')
            return PositionalIndex.from_dict(data)
        except FileNotFoundError:
            print(f"Error: The file '{filename}' was not found.")
        except pickle.UnpicklingError:
            print(f"Error: The file '{filename}' contains invalid pickle data.")
        except Exception as e:
            print(f"An error occurred while loading: {e}")


class SPIMIIndexer:
    def __init__(self, index_directory: str = 'saved_index_full_docs_small',
                 save_tokenization: bool = False, token_cache_directory: Optional[str] = None):
        """
        Initializes the SPIMIIndexer object with directories for index files and token caches.
        """
        self.document_id_mapper = DocumentIDMapper()
        self.tokenizer = Tokenizer()
        self.partial_index_count = 0
        self.index_directory = index_directory
        self.save_tokenization = save_tokenization
        self.token_cache_directory = token_cache_directory or f"{index_directory}/token_cache"

        # Create necessary directories
        os.makedirs(self.index_directory, exist_ok=True)
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

    def create_partial_index(self, documents: List[List[str]], document_ids: List[int]) -> PositionalIndex:
        """Creates a partial index from given documents and their IDs."""
        partial_index = PositionalIndex(document_id_mapper=None)
        for document, document_id in zip(documents, document_ids):
            for term_position, term in enumerate(document, start=1):
                partial_index.add_posting(term, document_id, term_position)
        return partial_index

    def save_partial_index(self, partial_index: PositionalIndex) -> str:
        """Saves a partial index to disk and returns the filename."""
        filename = f"{self.index_directory}/partial_index_{self.partial_index_count}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(partial_index.to_dict(), f)
        self.partial_index_count += 1
        return filename

    def merge_partial_indexes(self, partial_index_files: List[str]) -> PositionalIndex:
        """Merges all partial indexes into a final index."""
        final_index = PositionalIndex(document_id_mapper=self.document_id_mapper)
        start_time = time.time()

        for idx, filename in enumerate(partial_index_files, start=1):
            with open(filename, 'rb') as f:
                partial_index_data = pickle.load(f)
                partial_index = PositionalIndex.from_dict(partial_index_data)

            for term, postings_list in partial_index.positional_index.items():
                for doc_id, posting in postings_list.postings.items():
                    for position in posting.positions:
                        final_index.add_posting(term, doc_id, position)

            if idx % 5000 == 0:
                print(f"Merged {idx} partial indexes... Elapsed time: {time.time() - start_time:.2f} seconds")

        return final_index

    def save_final_index(self, final_index: PositionalIndex, filename: str = 'final_index.pickle') -> None:
        """Saves the final merged index to disk."""
        filepath = f"{self.index_directory}/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(final_index.to_dict(), f)
        print(f'Final index saved to {filepath}')

    def create_index_from_directory(self, directory: str, memory_limit: int = 2000) -> PositionalIndex:
        """
        Creates a positional index using the SPIMI approach for documents in a directory.
        """
        print(f'Creating positional index for directory: {directory}')
        self.document_id_mapper.create_from_directory(directory)
        total_documents = self.document_id_mapper.total_docs
        print(f'Total documents: {total_documents}')

        partial_index_files = []
        document_batch, document_id_batch = [], []
        batch_size, total_docs_processed = 0, 0
        start_time = time.time()

        for count, (document_name, document_id) in enumerate(self.document_id_mapper.document_to_id.items(), start=1):
            tokens = self._load_tokenized_document(document_id) or self.tokenizer.tokenize(
                open(f'{self.document_id_mapper.directory}/{document_name}', 'r').read())
            if self.save_tokenization:
                self._save_tokenized_document(document_id, tokens)

            document_batch.append(tokens)
            document_id_batch.append(document_id)
            batch_size += sum(len(token) for token in tokens)

            if batch_size >= memory_limit or count == total_documents:
                partial_index = self.create_partial_index(document_batch, document_id_batch)
                partial_index_files.append(self.save_partial_index(partial_index))

                total_docs_processed += len(document_batch)
                if total_docs_processed % 10000 < len(document_batch):
                    print(
                        f"Processed {total_docs_processed}/{total_documents} documents... Elapsed time: {time.time() - start_time:.2f} seconds")

                document_batch, document_id_batch, batch_size = [], [], 0

        print(f'Merging {len(partial_index_files)} partial indexes...')
        final_index = self.merge_partial_indexes(partial_index_files)
        final_index.calculate_tfidf()
        self.save_final_index(final_index)

        for filename in partial_index_files:
            os.remove(filename)

        return final_index
