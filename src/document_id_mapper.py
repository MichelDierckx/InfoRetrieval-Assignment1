import os
from typing import Dict

from natsort import natsorted


class DocumentIDMapper:
    def __init__(self, directory: str):
        """
        Initializes the DocumentIDMapper
        :param directory: A directory containing documents.
        """
        self.directory = directory
        self.document_to_id: Dict[str, int] = {}  # (filename, id)
        self.id_to_document: Dict[int, str] = {}  # (id, filename)
        self.setup()

    def setup(self):
        """
        Constructs two dictionaries:
        1. Mapping from filenames to unique IDs.
        2. Mapping from unique IDs to filenames.
        """
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
