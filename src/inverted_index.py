"""
Contains the InvertedIndex class
"""

FULL_DOCS_SMALL = "data/documents/full_docs_small"

from src.document_id_mapper import DocumentIDMapper


class InvertedIndex:
    def __init__(self, directory):
        self.inverted_index = {}
        self.document_id_mapper = DocumentIDMapper(directory)

    def update(self):
        pass
