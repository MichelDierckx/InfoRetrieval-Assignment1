"""
Contains the InvertedIndex class
"""
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
