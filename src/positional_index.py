import os
import sqlite3
import time
import pickle
from typing import Optional, List

from custom_types import DocID
from document_id_mapper import DocumentIDMapper
from tf_idf import calculate_tf_idf_weight
from tokenizer import Tokenizer



class PositionalIndex:
    def __init__(self, database_path: str, document_id_mapper: DocumentIDMapper):
        self.conn = sqlite3.connect(":memory:")  # Create an in-memory database
        self.cursor = self.conn.cursor()
        self.database_path = database_path
        self.document_id_mapper = document_id_mapper

        # Initialize term ID counter
        self.term_id_counter = 1  # Start term IDs from 1

        # Initialize tables for terms and postings in memory
        self._initialize_db()

    def _initialize_db(self):
        """Sets up SQLite tables without storing positions or tf-idf."""
        self.cursor.execute('''
            CREATE TABLE terms (
                term TEXT PRIMARY KEY,
                term_id INTEGER UNIQUE,
                df INTEGER DEFAULT 0
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE postings (
                term_id INTEGER,
                doc_id INTEGER,
                tf INTEGER DEFAULT 1,
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms (term_id)
            )
        ''')

    def add_posting(self, term: str, document_id: int):
        """
        Add or update a posting for a term, storing only tf.
        """
        # Check if the term already exists
        self.cursor.execute("SELECT term_id, df FROM terms WHERE term = ?", (term,))
        row = self.cursor.fetchone()

        if row is None:
            # Assign a new term_id
            term_id = self.term_id_counter
            self.term_id_counter += 1  # Increment counter for the next term
            self.cursor.execute("INSERT INTO terms (term, term_id, df) VALUES (?, ?, 1)", (term, term_id))
        else:
            term_id, df = row
            self.cursor.execute("UPDATE terms SET df = df + 1 WHERE term_id = ?", (term_id,))

        # Check if posting exists
        self.cursor.execute("SELECT tf FROM postings WHERE term_id = ? AND doc_id = ?", (term_id, document_id))
        posting_row = self.cursor.fetchone()

        if posting_row:
            # Update existing posting: increment tf
            tf = posting_row[0] + 1
            self.cursor.execute(
                "UPDATE postings SET tf = ? WHERE term_id = ? AND doc_id = ?",
                (tf, term_id, document_id)
            )
        else:
            # Insert new posting with tf = 1
            self.cursor.execute(
                "INSERT INTO postings (term_id, doc_id, tf) VALUES (?, ?, 1)",
                (term_id, document_id)
            )

    def save_to_disk(self, database_path: str):
        """Flush the in-memory database to disk using the backup method."""
        backup_db = sqlite3.connect(database_path)
        self.conn.backup(backup_db)
        backup_db.close()  # Close the backup connection manually

    def load_from_disk(self, database_path: str):
        """Load the database from an existing SQLite file into memory using the backup method."""
        disk_db = sqlite3.connect(database_path)
        disk_db.backup(self.conn)
        disk_db.close()  # Close the disk connection manually

    def close(self):
        """Close in-memory SQLite connection."""
        self.conn.close()

    def calculate_tfidf(self, term: str, doc_id: int) -> Optional[float]:
        """
        Dynamically calculate the tf-idf for a given term and document ID.
        """
        # Fetch term_id, document frequency (df) for the term, and total documents
        self.cursor.execute("SELECT term_id, df FROM terms WHERE term = ?", (term,))
        result = self.cursor.fetchone()

        if not result:
            return None  # Term does not exist in index

        term_id, df = result
        total_docs = self.document_id_mapper.total_docs

        # Fetch term frequency (tf) for the given document
        self.cursor.execute("SELECT tf FROM postings WHERE term_id = ? AND doc_id = ?", (term_id, doc_id))
        tf_row = self.cursor.fetchone()

        if tf_row:
            tf = tf_row[0]
            return calculate_tf_idf_weight(tf, total_docs, df)
        return None  # Document does not contain the term


class SPIMIIndexer:
    def __init__(self, database_path: str = 'data/saved_indexes/full_docs_small/full_docs_small_index.sqlite',
                 save_tokenization: bool = False, token_cache_directory: Optional[str] = None):
        self.tokenizer = Tokenizer()
        self.database_path = database_path
        self.save_tokenization = save_tokenization
        self.token_cache_directory = token_cache_directory or 'data/tokenized_documents/full_docs_small'

        # Create the token cache directory if tokenization is saved
        if self.save_tokenization:
            os.makedirs(self.token_cache_directory, exist_ok=True)

    def _save_tokenized_document(self, document_id: int, tokens: List[str]) -> None:
        """Saves tokenized output to cache as bytes if enabled."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(tokens, f)

    def _load_tokenized_document(self, document_id: int) -> Optional[List[str]]:
        """Loads tokenized output from cache if available."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def create_index_from_directory(self, directory: str) -> PositionalIndex:
        print(f'Creating positional index for directory: {directory}')
        document_id_mapper = DocumentIDMapper()
        positional_index = PositionalIndex(self.database_path, document_id_mapper)
        positional_index.document_id_mapper.create_from_directory(directory)
        total_documents = positional_index.document_id_mapper.total_docs
        print(f'Total documents: {total_documents}')

        total_docs_processed = 0
        start_time = time.time()

        for count, (document_name, document_id) in enumerate(positional_index.document_id_mapper.document_to_id.items(),
                                                             start=1):
            tokens = self._load_tokenized_document(document_id) or self.tokenizer.tokenize(
                open(f'{positional_index.document_id_mapper.directory}/{document_name}', 'r').read())
            if self.save_tokenization:
                self._save_tokenized_document(document_id, tokens)

            for term in tokens:
                positional_index.add_posting(term, document_id)
            total_docs_processed += 1

            if total_docs_processed % 1000 == 0:
                print(
                    f"Processed {total_docs_processed}/{total_documents} documents... Elapsed time: {time.time() - start_time:.2f} seconds")

        positional_index.conn.commit()
        positional_index.save_to_disk(self.database_path)
        print(f"Index created at {self.database_path}")

        return positional_index
