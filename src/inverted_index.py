"""
Contains the InvertedIndex class
"""
import json
from typing import Dict, List

from src.custom_types import Term
from src.document_id_mapper import DocumentIDMapper
from src.tokenizer import Tokenizer

FULL_DOCS_SMALL = "data/documents/full_docs_small"


class Postings:
    def __init__(self):
        self.df = 0
        self.postings_list: Dict[int, List[int]] = {}

    def update(self, document_id: int, term_position: int):
        if document_id in self.postings_list:
            self.postings_list[document_id].append(term_position)
        else:
            self.postings_list[document_id] = [term_position]
            self.df += 1

    def to_dict(self) -> Dict:
        """
        Convert Postings object to a dictionary.
        """
        return {
            'df': self.df,
            'postings_list': self.postings_list
        }

    @staticmethod
    def from_dict(data: Dict):
        """
        Create a Postings object from a dictionary.
        """
        postings = Postings()
        postings.df = data['df']
        postings.postings_list = data['postings_list']
        return postings


class PositionalIndex:
    def __init__(self):
        self.positional_index: Dict[Term, Postings] = {}
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
                    self.positional_index[term] = Postings()
                self.positional_index[term].update(document_id, term_position)
        print('Successfully created positional index.')

    def to_dict(self) -> Dict:
        """
        Convert the PositionalIndex to a dictionary for JSON serialization.
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
        positional_index.positional_index = {term: Postings.from_dict(postings_data)
                                             for term, postings_data in data['positional_index'].items()}
        positional_index.document_id_mapper = DocumentIDMapper.from_dict(data['document_id_mapper'])
        return positional_index

    def save_to_file(self, filename: str) -> None:
        """
        Save the positional index to a file (JSON).
        """
        with open(filename, 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    @staticmethod
    def load_from_file(filename: str):
        """
        Load the positional index from a file (JSON).
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return PositionalIndex.from_dict(data)

    def pretty_print(self) -> None:
        """
        Pretty print the positional index.
        """
        print(json.dumps(self.to_dict(), indent=4))
