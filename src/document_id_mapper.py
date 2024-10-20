"""
Contains the DocumentIDMapper class. This class is used associate identifiers with filenames and vice versa.
"""
import json
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

    def pretty_print(self) -> None:
        """
        Pretty print the document_id_mapper
        """
        print(json.dumps(self.to_dict(), indent=4))
