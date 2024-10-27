import cProfile
import io
import pstats

from positional_index import Indexer

FULL_DOCS_SMALL_DIRECTORY = "data/documents/full_docs_small"
FULL_DOCS_DIRECTORY = "data/documents/full_docs"
FULL_DOCS_SMALL_INDEX_FILE = "data/saved_indexes/full_docs_small/full_docs_small_index.sqlite"
FULL_DOCS_INDEX_FILE = "data/saved_indexes/full_docs/full_docs_index.sqlite"
FULL_DOCS_SMALL_TOKENIZED_DOCUMENTS_DIRECTORY = "data/tokenized_documents/full_docs_small"
FULL_DOCS_TOKENIZED_DOCUMENTS_DIRECTORY = "data/tokenized_documents/full_docs"

SAVE_TOKENIZED_DOCUMENTS = True
DEV = True


def main():
    """ Main program """
    if DEV:
        documents_dir = FULL_DOCS_SMALL_DIRECTORY
        index_file = FULL_DOCS_SMALL_INDEX_FILE
        token_cache_dir = FULL_DOCS_SMALL_TOKENIZED_DOCUMENTS_DIRECTORY
    else:
        documents_dir = FULL_DOCS_DIRECTORY
        index_file = FULL_DOCS_INDEX_FILE
        token_cache_dir = FULL_DOCS_TOKENIZED_DOCUMENTS_DIRECTORY

    # Create a StringIO buffer to capture the profiling results
    profile_buffer = io.StringIO()
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    indexer = Indexer(SAVE_TOKENIZED_DOCUMENTS, token_cache_dir)

    final_index = indexer.create_index_from_directory(documents_dir)

    final_index.save_to_file(index_file)

    final_index.load_from_file(index_file)

    final_index.print_posting_list("tolerate")

    profiler.disable()

    # Print profiling results (only from module src)
    pstats.Stats(profiler, stream=profile_buffer).sort_stats('cumulative').print_stats("src")
    print(profile_buffer.getvalue())

    return 0


if __name__ == "__main__":
    main()
