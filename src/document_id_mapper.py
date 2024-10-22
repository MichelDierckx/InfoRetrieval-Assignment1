"""
Contains the DocumentIDMapper class. This class is used associate identifiers with filenames and vice versa.
"""
import json
import os
from typing import Dict

from natsort import natsorted

from custom_types import DocID


class DocumentIDMapper:
    def __init__(self):
        """
        Initializes the DocumentIDMapper
        """
        self.directory = ""
        self.total_docs = 0
        self.document_to_id: Dict[str, DocID] = {}  # (filename, id)
        self.id_to_document: Dict[DocID, str] = {}  # (id, filename)

    def create_from_directory(self, directory: str) -> None:
        """
        Constructs two dictionaries:
        1. Mapping from filenames to unique IDs.
        2. Mapping from unique IDs to filenames.
        Also saves the associated directory path.
        """
        print(f'Creating a mapping from document names in {directory} to unique IDs.')

        # Check if the provided directory exists
        if not os.path.isdir(directory):
            print(f"Error: The directory '{directory}' does not exist.")
            return  # Stop execution if the directory is invalid

        try:
            # List all text files in the directory (sorted alphabetically)
            files = natsorted(os.listdir(directory))
            documents = [file for file in files if file.endswith(".txt")]

            # Assign an identifier to each document
            for index, text_file in enumerate(documents, start=1):
                self.document_to_id[text_file] = index  # Map filename to ID
                self.id_to_document[index] = text_file  # Map ID to filename
                self.total_docs += 1

            # Save the directory path
            self.directory = directory
            print(f"Successfully mapped {len(self.document_to_id)} documents.")

        except FileNotFoundError:
            print(f"Error: The directory '{directory}' could not be found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def get_id(self, document_name: str) -> int:
        """
        Retrieves the ID for a given filename.
        """
        return self.document_to_id.get(document_name)

    def get_filename(self, document_id: DocID) -> str:
        """
        Retrieves the filename for a given ID.
        """
        return self.id_to_document.get(document_id)

    def get_total_docs(self):
        return self.total_docs

    def to_dict(self) -> Dict:
        """
        Convert DocumentIDMapper to a dictionary.
        """
        return {
            'directory': self.directory,
            'total_docs': self.total_docs,
            'document_to_id': self.document_to_id,
            'id_to_document': self.id_to_document
        }

    @staticmethod
    def from_dict(data: Dict):
        """
        Create a DocumentIDMapper from a dictionary.
        """
        document_id_mapper = DocumentIDMapper()
        document_id_mapper.directory = data['directory']
        document_id_mapper.directory = data['total_docs']
        document_id_mapper.document_to_id = data['document_to_id']
        document_id_mapper.id_to_document = data['id_to_document']
        return document_id_mapper

    def pretty_print(self) -> None:
        """
        Pretty print the document_id_mapper
        """
        print(json.dumps(self.to_dict(), indent=4))
