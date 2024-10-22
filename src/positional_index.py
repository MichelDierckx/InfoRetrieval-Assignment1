"""
Contains the InvertedIndex class
"""
import json
import os
import pickle
from typing import Dict, List, Optional

from custom_types import Term
from document_id_mapper import DocumentIDMapper
from tokenizer import Tokenizer


class Posting:
    def __init__(self):
        self.tf = 0  # Term frequency
        self.positions: List[int] = []  # List of positions in the document

    def update(self, term_position: int) -> None:
        """
        Update the Posting by adding a new position and incrementing the term frequency.
        """
        self.positions.append(term_position)
        self.tf += 1

    def to_list(self) -> List:
        """
        Convert Posting object to a List[tf, List[position]]
        """
        return [self.tf, self.positions]

    @staticmethod
    def from_list(data: List):
        """
        Create a Posting object from a List[tf, List[position]].
        """
        posting = Posting()
        posting.tf = data[0]
        posting.positions = data[1]
        return posting

    def pretty_print(self) -> None:
        """
        Pretty print the Posting
        """
        print(self.to_list())


class PostingsList:
    def __init__(self):
        self.df = 0
        self.postings: Dict[int, Posting] = {}

    def update(self, document_id: int, term_position: int):
        if document_id in self.postings:
            self.postings[document_id].update(term_position)
        else:
            self.postings[document_id] = Posting()
            self.postings[document_id].update(term_position)
            self.df += 1

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

    def pretty_print(self) -> None:
        """
        Pretty print the PostingsList
        """
        print(json.dumps(self.to_dict(), indent=4))

    def get_posting(self, document_id: int) -> Optional[Posting]:
        return self.postings.get(document_id)


class PositionalIndex:
    def __init__(self):
        self.positional_index: Dict[Term, PostingsList] = {}
        self.document_id_mapper = DocumentIDMapper()

    def create_from_directory(self, directory: str):
        print(f'Creating a positional index for the specified directory: {directory}')
        tokenizer = Tokenizer()
        self.document_id_mapper.create_from_directory(directory)
        for document_name, document_id in self.document_id_mapper.document_to_id.items():
            with open(f'{self.document_id_mapper.directory}/{document_name}', 'r') as f:
                document = f.read()
            tokens = tokenizer.tokenize(document)
            for term_position, term in enumerate(tokens, start=1):
                if term not in self.positional_index:
                    self.positional_index[term] = PostingsList()
                self.positional_index[term].update(document_id, term_position)
        print('Successfully created positional index.')

    def to_dict(self) -> Dict:
        """
        Convert the PositionalIndex to a dictionary.
        """
        return {
            'positional_index': {term: postings.to_dict() for term, postings in self.positional_index.items()},
            'document_id_mapper': self.document_id_mapper.to_dict()
        }

    @staticmethod
    def from_dict(data: Dict):
        """
        Create a PositionalIndex from a dictionary.
        """
        positional_index = PositionalIndex()
        positional_index.positional_index = {term: PostingsList.from_dict(postings_data)
                                             for term, postings_data in data['positional_index'].items()}
        positional_index.document_id_mapper = DocumentIDMapper.from_dict(data['document_id_mapper'])
        return positional_index

    def save_to_file(self, filename: str) -> None:
        """
        Save the positional index to a file (binary using pickle).
        """
        try:
            print(f'Saving positional index to {filename}')
            with open(filename, 'wb') as outfile:
                pickle.dump(self.to_dict(), outfile)
            print(f'Successfully saved positional index to {filename}')
        except FileNotFoundError:
            print(f"Error: The directory for '{filename}' does not exist.")
        except Exception as e:
            print(f"An error occurred while saving: {e}")

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

    def pretty_print(self) -> None:
        """
        Pretty print the positional index.
        """
        print(json.dumps(self.to_dict(), indent=4))

    def get_terms(self) -> List[Term]:
        return list(self.positional_index.keys())

    def get_postings_list(self, term: Term) -> Optional[PostingsList]:
        return self.positional_index.get(term)
