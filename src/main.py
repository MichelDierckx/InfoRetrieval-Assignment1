import cProfile
import io
import pstats

from inverted_index import Indexer, InvertedIndex, DocumentRanker
from tokenizer import Tokenizer

# Directories and file paths
FULL_DOCS_SMALL_DIRECTORY = "data/documents/full_docs_small"
FULL_DOCS_DIRECTORY = "data/documents/full_docs"
FULL_DOCS_SMALL_INDEX_FILE = "data/saved_indexes/full_docs_small/full_docs_small_inverted_index.pkl"
FULL_DOCS_INDEX_FILE = "data/saved_indexes/full_docs/full_docs_inverted_index.pkl"
FULL_DOCS_SMALL_DOCUMENT_LENGTHS_FILE = "data/saved_indexes/full_docs_small/full_docs_small_lengths.npy"
FULL_DOCS_DOCUMENT_LENGTHS_FILE = "data/saved_indexes/full_docs/full_docs_lengths.npy"
FULL_DOCS_SMALL_TOKENIZED_DOCUMENTS_DIRECTORY = "data/tokenized_documents/full_docs_small"
FULL_DOCS_TOKENIZED_DOCUMENTS_DIRECTORY = "data/tokenized_documents/full_docs"

SAVE_TOKENIZED_DOCUMENTS = True
DEV = True


def main():
    """ Main program """
    if DEV:
        documents_dir = FULL_DOCS_SMALL_DIRECTORY
        index_file = FULL_DOCS_SMALL_INDEX_FILE
        doc_lengths_file = FULL_DOCS_SMALL_DOCUMENT_LENGTHS_FILE
        token_cache_dir = FULL_DOCS_SMALL_TOKENIZED_DOCUMENTS_DIRECTORY
    else:
        documents_dir = FULL_DOCS_DIRECTORY
        index_file = FULL_DOCS_INDEX_FILE
        doc_lengths_file = FULL_DOCS_DOCUMENT_LENGTHS_FILE
        token_cache_dir = FULL_DOCS_TOKENIZED_DOCUMENTS_DIRECTORY

    # Create a StringIO buffer to capture the profiling results
    profile_buffer = io.StringIO()
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # init tokenizer object
    tokenizer = Tokenizer()

    # init indexer, used to create inverted index
    indexer = Indexer(tokenizer, SAVE_TOKENIZED_DOCUMENTS, token_cache_dir)

    # let indexer create inverted index from the specified documents directory
    final_index = indexer.create_index_from_directory(documents_dir)

    # save the inverted index and document lengths to files
    final_index.save(index_file, doc_lengths_file)  # Saving lengths as a separate file

    # load the inverted index and document lengths from files
    loaded_index = InvertedIndex.load(index_file, doc_lengths_file)

    # print posting list for tolerate
    loaded_index.print_posting_list("tolerate")

    # object to rank documents given a query
    document_ranker = DocumentRanker(tokenizer, loaded_index)

    ranked_documents = document_ranker.rank_documents("types of road hugger tires")
    # Print the rankings
    print('Rankings:')
    for doc_id, score in ranked_documents:
        print(f'Document ID: {doc_id}, Score: {score:.4f}')  # Format score to 4 decimal places

    # stop profiling
    profiler.disable()

    # profiling results
    pstats.Stats(profiler, stream=profile_buffer).sort_stats('cumulative').print_stats("src")
    print(profile_buffer.getvalue())

    return 0


if __name__ == "__main__":
    main()
