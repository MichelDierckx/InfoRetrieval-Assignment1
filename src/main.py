import cProfile
import io
import pstats

from inverted_index import Indexer, DocumentRanker
from src.inverted_index import InvertedIndex
from tokenizer import Tokenizer

# Document collections
FULL_DOCS_SMALL = "full_docs_small"
FULL_DOCS = "full_docs"

# Document directory
DIRECTORY_DOCUMENTS = "data/documents"

# Query directory
DIRECTORY_QUERIES = "data/queries"

# Tokenized document directory
DIRECTORY_TOKENIZED_DOCUMENTS = "results/tokenized_documents"

# Index directory
DIRECTORY_INDEX = "results/indexes"

# Rankings directory
DIRECTORY_RANKINGS = "results/rankings"

# Default file names
DEFAULT_INDEX_FILE_NAME = "index.pkl"
DEFAULT_DOC_LENGTHS_FILE_NAME = "doc_lengths.npy"
DEFAULT_TERM_FREQUENCY_MATRIX_FILE_NAME = "term_frequency_matrix.npz"  # New file for term frequency matrix

# Query filenames and delimiter
DEV_QUERIES = ("dev_queries.tsv", '\t')
DEV_QUERIES_SMALL = ("dev_small_queries.csv", ',')
QUERIES = ("queries.csv", '\t')

# Calculated rankings filenames
DEV_QUERIES_RANKING = "dev_queries_ranking.csv"
DEV_QUERIES_SMALL_RANKING = "dev_queries_small_ranking.csv"
QUERIES_RANKING = "queries_rankings.csv"

# Flags
BATCH_SIZE = 2000
SAVE_TOKENIZED_DOCUMENTS = False
LOAD_TOKENIZED_DOCUMENTS = True
LOAD_INDEX = True

DEV = True
SIZE = "FULL"  # Either FULL or SMALL


def main():
    """ Main program """
    if DEV:
        if SIZE == "SMALL":
            documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS_SMALL}'
            index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/{DEFAULT_INDEX_FILE_NAME}'
            partial_index_directory = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/partial'
            doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
            term_frequency_file = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/{DEFAULT_TERM_FREQUENCY_MATRIX_FILE_NAME}'  # New line
            token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS_SMALL}'
            queries_file = f'{DIRECTORY_QUERIES}/{DEV_QUERIES_SMALL[0]}'
            queries_file_delimiter = DEV_QUERIES_SMALL[1]
            queries_ranking = f'{DIRECTORY_RANKINGS}/{DEV_QUERIES_SMALL_RANKING}'
        elif SIZE == "FULL":
            documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS}'
            index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_INDEX_FILE_NAME}'
            partial_index_directory = f'{DIRECTORY_INDEX}/{FULL_DOCS}/partial'
            doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
            term_frequency_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_TERM_FREQUENCY_MATRIX_FILE_NAME}'  # New line
            token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS}'
            queries_file = f'{DIRECTORY_QUERIES}/{DEV_QUERIES[0]}'
            queries_file_delimiter = DEV_QUERIES[1]
            queries_ranking = f'{DIRECTORY_RANKINGS}/{DEV_QUERIES_RANKING}'
        else:
            print(f'Size {SIZE} should be SMALL or FULL')
            return
    else:
        documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS}'
        index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_INDEX_FILE_NAME}'
        partial_index_directory = f'{DIRECTORY_INDEX}/{FULL_DOCS}/partial'
        doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
        term_frequency_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_TERM_FREQUENCY_MATRIX_FILE_NAME}'  # New line
        token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS}'
        queries_file = f'{DIRECTORY_QUERIES}/{QUERIES[0]}'
        queries_file_delimiter = QUERIES[1]
        queries_ranking = f'{DIRECTORY_RANKINGS}/{QUERIES_RANKING}'

    # Create a StringIO buffer to capture the profiling results
    profile_buffer = io.StringIO()
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Init tokenizer object
    tokenizer = Tokenizer()

    if not LOAD_INDEX:

        # Init indexer, used to create inverted index
        indexer = Indexer(tokenizer=tokenizer, save_tokenization=SAVE_TOKENIZED_DOCUMENTS,
                          load_tokenization=LOAD_TOKENIZED_DOCUMENTS, token_cache_directory=token_cache_dir)

        # Let indexer create inverted index from the specified documents directory
        inverted_index = indexer.create_index_from_directory(directory=documents_dir,
                                                             partial_index_directory=partial_index_directory,
                                                             batch_size=BATCH_SIZE)

        # Save the inverted index and document lengths to files
        inverted_index.save(index_filename=index_file,
                            lengths_filename=doc_lengths_file,
                            term_frequency_filename=term_frequency_file)  # Updated to include term frequency file

    else:
        inverted_index = InvertedIndex.load(index_filename=index_file, lengths_filename=doc_lengths_file,
                                            term_frequency_filename=term_frequency_file)

    document_ranker = DocumentRanker(tokenizer=tokenizer, inverted_index=inverted_index)

    document_ranker.rank_queries_from_file(input_file=queries_file, output_file=queries_ranking,
                                           delimiter=queries_file_delimiter, top_k=k)

    # Stop profiling
    profiler.disable()

    # profiling results
    pstats.Stats(profiler, stream=profile_buffer).sort_stats('cumulative').print_stats("src")
    print(profile_buffer.getvalue())

    return 0


if __name__ == "__main__":
    main()
