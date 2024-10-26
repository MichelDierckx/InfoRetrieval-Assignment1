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
    def __init__(self, index_directory: str = 'saved_index_full_docs_small'):
        """
        Creates an SPIMIIndexer object. The parameter index_directory specifies the directory the SPIMIIndexer will load indexes from and save indexes to.
        """
        self.document_id_mapper = DocumentIDMapper()
        self.tokenizer = Tokenizer()
        self.partial_index_count = 0  # To count the number of partial indexes
        self.index_directory = index_directory

        # Create index directory if it doesn't exist
        if not os.path.exists(self.index_directory):
            os.makedirs(self.index_directory)

    def create_partial_index(self, documents: List[str], document_ids: List[int]) -> PositionalIndex:
        """
        Given a list of documents and their corresponding document ids, create a partial index.
        """
        partial_index = PositionalIndex(document_id_mapper=None)  # Partial index doesn't need a DocumentIDMapper

        for document, document_id in zip(documents, document_ids):
            tokens = self.tokenizer.tokenize(document)
            for term_position, term in enumerate(tokens, start=1):
                partial_index.add_posting(term, document_id, term_position)

        return partial_index

    def save_partial_index(self, partial_index: PositionalIndex) -> str:
        """
        Save a partial index to a file named partial_index_x.pickle in the index directory.
        """
        filename = f"{self.index_directory}/partial_index_{self.partial_index_count}.pickle"
        self.partial_index_count += 1
        with open(filename, 'wb') as f:
            pickle.dump(partial_index.to_dict(), f)
        return filename

    def merge_partial_indexes(self, partial_index_files: List[str]) -> PositionalIndex:
        """
        Merge multiple partial indexes into a final index.
        """
        final_index = PositionalIndex(
            document_id_mapper=self.document_id_mapper)  # Create final index with the DocumentIDMapper

        # Start the timer for the merge process
        start_time = time.time()
        total_partial_indexes_merged = 0  # Counter for the number of partial indexes merged

        for filename in partial_index_files:
            with open(filename, 'rb') as f:
                partial_index_data = pickle.load(f)
                partial_index = PositionalIndex.from_dict(partial_index_data)

                for term, postings_list in partial_index.positional_index.items():
                    for doc_id, posting in postings_list.postings.items():
                        # Update posting in final_index with all term positions from the posting in the partial index
                        for position in posting.positions:
                            final_index.add_posting(term, doc_id, position)

                total_partial_indexes_merged += 1  # Increment the partial index counter

                # Print status update every 5000 partial indexes merged
                if total_partial_indexes_merged % 5000 == 0:
                    elapsed_time = time.time() - start_time  # Calculate elapsed time
                    print(f"Merged {total_partial_indexes_merged} partial indexes... "
                          f"Total elapsed time: {elapsed_time:.2f} seconds")
        return final_index

    def save_final_index(self, final_index: PositionalIndex, filename: str = 'final_index.pickle') -> None:
        """
        Save the final index to a file named final_index.pickle in the index directory.
        """
        filepath = f"{self.index_directory}/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(final_index.to_dict(), f)
        print(f'Final index saved to {filepath}')

    def create_index_from_directory(self, directory: str, memory_limit: int = 2000) -> PositionalIndex:
        """
        Create a positional index using the SPIMI approach for documents in a directory.
        :param memory_limit: The combined length of documents (in characters) in a batch will be limited by memory_limit.
        """
        print(f'Creating positional index for directory: {directory}')
        self.document_id_mapper.create_from_directory(directory)
        total_documents = self.document_id_mapper.total_docs
        print(f'Total documents: {total_documents}')

        partial_index_files = []
        document_batch = []
        document_id_batch = []
        batch_size = 0
        total_docs_processed = 0

        # Start the timer
        start_time = time.time()

        # Process documents in chunks based on memory limit
        for count, (document_name, document_id) in enumerate(self.document_id_mapper.document_to_id.items(), start=1):
            with open(f'{self.document_id_mapper.directory}/{document_name}', 'r') as f:
                document = f.read()
                document_batch.append(document)
                document_id_batch.append(document_id)
                batch_size += len(document)  # Approximate memory usage by document text length

            # Batch gets processed when memory limit is reached or the last document is processed
            if batch_size >= memory_limit or count == total_documents:
                partial_index = self.create_partial_index(document_batch, document_id_batch)
                partial_index_file = self.save_partial_index(partial_index)
                partial_index_files.append(partial_index_file)

                # Increment total documents processed
                total_docs_processed += len(document_batch)

                # Progress report every 10,000 documents processed
                if total_docs_processed % 10000 < len(document_batch):
                    elapsed_time = time.time() - start_time  # Calculate total elapsed time
                    print(f"Processed {total_docs_processed} / {total_documents} documents... "
                          f"Total elapsed time: {elapsed_time:.2f} seconds")

                # Clear the batch
                document_batch = []
                document_id_batch = []
                batch_size = 0

        print(f'Merging {len(partial_index_files)} partial indexes...')
        final_index = self.merge_partial_indexes(partial_index_files)
        print('Index creation complete.')

        # Calculate TF-IDF scores
        final_index.calculate_tfidf()

        # Save final index
        self.save_final_index(final_index, 'final_index.pickle')

        # Remove partial indexes
        for filename in partial_index_files:
            os.remove(filename)

        return final_index
