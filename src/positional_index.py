import json
import os
import pickle
import sqlite3
import time
from typing import Dict, List, Optional

from custom_types import DocID
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
        """Sets up SQLite tables with BLOB storage for positions."""
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
                tfidf REAL,  -- Add tfidf column
                positions BLOB,
                PRIMARY KEY (term_id, doc_id),
                FOREIGN KEY (term_id) REFERENCES terms (term_id)
            )
        ''')

    def add_posting(self, term: str, document_id: int, term_position: int):
        """
        Add or update a posting for a term, storing positions as a BLOB.
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
        self.cursor.execute("SELECT tf, positions FROM postings WHERE term_id = ? AND doc_id = ?",
                            (term_id, document_id))
        posting_row = self.cursor.fetchone()

        if posting_row:
            # Update existing posting: increment tf and update positions
            tf, positions_blob = posting_row
            tf += 1
            positions = pickle.loads(positions_blob)  # Deserialize positions
            positions.append(term_position)
            positions_blob = pickle.dumps(positions)  # Re-serialize positions
            self.cursor.execute(
                "UPDATE postings SET tf = ?, positions = ? WHERE term_id = ? AND doc_id = ?",
                (tf, positions_blob, term_id, document_id)
            )
        else:
            # Insert new posting with tf = 1 and the initial position
            positions_blob = pickle.dumps([term_position])
            self.cursor.execute(
                "INSERT INTO postings (term_id, doc_id, tf, tfidf, positions) VALUES (?, ?, 1, NULL, ?)",
                (term_id, document_id, positions_blob)  # Initially set tfidf to NULL
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

    def get_postings_list(self, term: str) -> Optional[PostingsList]:
        """
        Retrieve the Posting list for a specific term.
        Returns a PostingsList object or None if the term does not exist.
        """
        # Fetch term_id and document frequency (df) for the term
        self.cursor.execute("SELECT term_id, df FROM terms WHERE term = ?", (term,))
        result = self.cursor.fetchone()

        if result:
            term_id, df = result
            postings_list = PostingsList()
            postings_list.df = df

            # Fetch all postings for this term
            self.cursor.execute("SELECT doc_id, tf, tfidf, positions FROM postings WHERE term_id = ?", (term_id,))
            rows = self.cursor.fetchall()

            for doc_id, tf, tfidf, positions_blob in rows:
                positions = pickle.loads(positions_blob)  # Deserialize positions
                posting = Posting()
                posting.tf = tf
                posting.tfidf = tfidf  # Ensure tfidf is included
                posting.positions = positions
                postings_list.postings[doc_id] = posting

            return postings_list
        return None

    def get_positions(self, term: str, doc_id: int) -> Optional[List[int]]:
        """
        Retrieve positions of a term in a specific document.
        """
        self.cursor.execute("SELECT term_id FROM terms WHERE term = ?", (term,))
        term_id_row = self.cursor.fetchone()

        if term_id_row:
            term_id = term_id_row[0]
            self.cursor.execute("SELECT positions FROM postings WHERE term_id = ? AND doc_id = ?", (term_id, doc_id))
            row = self.cursor.fetchone()
            if row:
                return pickle.loads(row[0])  # Deserialize positions
        return None

    def calculate_tfidf(self):
        """
        Calculate tf-idf for each posting.
        """
        total_docs = self.document_id_mapper.total_docs
        print('Calculating tf-idf weights...')

        # Calculate tf-idf and update postings
        self.cursor.execute("SELECT term_id, df FROM terms")
        for term_id, df in self.cursor.fetchall():
            self.cursor.execute("SELECT doc_id, tf FROM postings WHERE term_id = ?", (term_id,))
            postings = self.cursor.fetchall()
            for doc_id, tf in postings:
                tfidf = calculate_tf_idf_weight(tf, total_docs, df)
                self.cursor.execute(
                    "UPDATE postings SET tfidf = ? WHERE term_id = ? AND doc_id = ?", (tfidf, term_id, doc_id)
                )
        self.conn.commit()

    def normalize_tfidf(self):
        """
        Normalize tf-idf values by document length using a single SQL statement.
        """
        self.cursor.execute('''
            WITH doc_lengths AS (
                SELECT doc_id, SQRT(SUM(tfidf * tfidf)) AS length
                FROM postings
                GROUP BY doc_id
            )
            UPDATE postings
            SET tfidf = tfidf / dl.length
            FROM doc_lengths dl
            WHERE postings.doc_id = dl.doc_id AND dl.length > 0
        ''')

        self.conn.commit()


class SPIMIIndexer:
    def __init__(self, database_path: str = 'data/saved_indexes/full_docs_small/full_docs_small_index.sqlite',
                 save_tokenization: bool = False, token_cache_directory: Optional[str] = None):
        self.tokenizer = Tokenizer()
        self.database_path = database_path  # Changed from index_directory to database_path
        self.save_tokenization = save_tokenization
        self.token_cache_directory = token_cache_directory or 'data/tokenized_documents/full_docs_small'

        # Create the token cache directory if tokenization is saved
        if self.save_tokenization:
            os.makedirs(self.token_cache_directory, exist_ok=True)

    def _save_tokenized_document(self, document_id: int, tokens: List[str]) -> None:
        """Saves tokenized output to cache as bytes if enabled."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"  # Changed to .pkl
        with open(cache_file, 'wb') as f:  # Open in binary write mode
            pickle.dump(tokens, f)  # Use pickle to serialize tokens

    def _load_tokenized_document(self, document_id: int) -> Optional[List[str]]:
        """Loads tokenized output from cache if available."""
        cache_file = f"{self.token_cache_directory}/doc_{document_id}.pkl"  # Changed to .pkl
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:  # Open in binary read mode
                return pickle.load(f)  # Use pickle to deserialize tokens
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

            for term_position, term in enumerate(tokens, start=1):
                positional_index.add_posting(term, document_id, term_position)
            total_docs_processed += 1

            if total_docs_processed % 1000 == 0:
                print(
                    f"Processed {total_docs_processed}/{total_documents} documents... Elapsed time: {time.time() - start_time:.2f} seconds")

        positional_index.calculate_tfidf()
        positional_index.normalize_tfidf()
        positional_index.save_to_disk(self.database_path)
        # positional_index.close()
        print(f"Index created at {self.database_path}")

        return positional_index
