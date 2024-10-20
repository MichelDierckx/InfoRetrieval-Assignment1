"""
Contains the DocumentIDMapper class. This class is used associate identifiers with filenames and vice versa.
"""

import os
from typing import Dict

from natsort import natsorted


class DocumentIDMapper:
    def __init__(self):
        """
        Initializes the DocumentIDMapper
        """
        self.document_to_id: Dict[str, int] = {}  # (filename, id)
        self.id_to_document: Dict[int, str] = {}  # (id, filename)
        self.directory = ""

    def create_from_directory(self, directory: str) -> None:
        """
        Constructs two dictionaries:
        1. Mapping from filenames to unique IDs.
        2. Mapping from unique IDs to filenames.
        Also saves the associated directory path.
        """
        print(f'Create mapping from document names in {directory} to unique IDs.')
        try:
            # list all textfiles in directory (sorted alphabetically)
            files = natsorted(os.listdir(self.directory))
            documents = []
            for file in files:
                if file.endswith(".txt"):
                    documents.append(file)

            # assign identifier to each document
            for index, text_file in enumerate(documents, start=1):
                self.document_to_id[text_file] = index  # Map filename to ID
                self.id_to_document[index] = text_file  # Map ID to filename
            self.directory = directory
        except FileNotFoundError:
            print(f"Could not find directory '{self.directory}'.")
        except Exception as e:
            print(f"Error: {e}")

    def get_id(self, document_name: str) -> int:
        """
        Retrieves the ID for a given filename.
        """
        return self.document_to_id.get(document_name)

    def get_filename(self, file_id: int) -> str:
        """
        Retrieves the filename for a given ID.
        """
        return self.id_to_document.get(file_id)

    def to_dict(self) -> Dict:
        """
        Convert DocumentIDMapper to a dictionary.
        """
        return {
            'document_to_id': self.document_to_id,
            'id_to_document': self.id_to_document,
            'directory': self.directory
        }

    @staticmethod
    def from_dict(data: Dict):
        """
        Create a DocumentIDMapper from a dictionary.
        """
        document_id_mapper = DocumentIDMapper()
        document_id_mapper.document_to_id = data['document_to_id']
        document_id_mapper.id_to_document = data['id_to_document']
        document_id_mapper.directory = data['directory']
        return document_id_mapper
