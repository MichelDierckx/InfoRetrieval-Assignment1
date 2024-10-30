import cProfile
import io
import pstats

from inverted_index import Indexer, DocumentRanker
from tokenizer import Tokenizer

# document collections
FULL_DOCS_SMALL = "full_docs_small"
FULL_DOCS = "full_docs"

# document directory
DIRECTORY_DOCUMENTS = "data/documents"

# query directory
DIRECTORY_QUERIES = "data/queries"

# tokenized document directory
DIRECTORY_TOKENIZED_DOCUMENTS = "results/tokenized_documents"

# index directory
DIRECTORY_INDEX = "results/indexes"

# rankings directory
DIRECTORY_RANKINGS = "results/rankings"

# default file names
DEFAULT_INDEX_FILE_NAME = "index.pkl"
DEFAULT_DOC_LENGTHS_FILE_NAME = "doc_lengths.npy"

# query filenames and delimiter
DEV_QUERIES = ("dev_queries.tsv", '\t')
DEV_QUERIES_SMALL = ("dev_small_queries.csv", ',')
QUERIES = ("queries.csv", '\t')

# calculated rankings filenames
DEV_QUERIES_RANKING = "dev_queries_ranking.csv"
DEV_QUERIES_SMALL_RANKING = "dev_queries_small_ranking.csv"
QUERIES_RANKING = "queries_rankings.csv"

# flags
SAVE_TOKENIZED_DOCUMENTS = False
LOAD_TOKENIZED_DOCUMENTS = True
DEV = True
SIZE = "SMALL"  # either FULL or SMALL


def main():
    """ Main program """
    if DEV:
        if SIZE == "SMALL":
            documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS_SMALL}'
            index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/{DEFAULT_INDEX_FILE_NAME}'
            doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS_SMALL}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
            token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS_SMALL}'
            queries_file = f'{DIRECTORY_QUERIES}/{DEV_QUERIES_SMALL[0]}'
            queries_file_delimiter = DEV_QUERIES_SMALL[1]
            queries_ranking = f'{DIRECTORY_RANKINGS}/{DEV_QUERIES_SMALL_RANKING}'
            k = 1  # all relevant ranked documents are returned
        elif SIZE == "FULL":
            documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS}'
            index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_INDEX_FILE_NAME}'
            doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
            token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS}'
            queries_file = f'{DIRECTORY_QUERIES}/{DEV_QUERIES[0]}'
            queries_file_delimiter = DEV_QUERIES[1]
            queries_ranking = f'{DIRECTORY_RANKINGS}/{DEV_QUERIES_RANKING}'
            k = None
        else:
            print(f'Size {SIZE} should be SMALL or FULL')
            return
    else:
        documents_dir = f'{DIRECTORY_DOCUMENTS}/{FULL_DOCS}'
        index_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_INDEX_FILE_NAME}'
        doc_lengths_file = f'{DIRECTORY_INDEX}/{FULL_DOCS}/{DEFAULT_DOC_LENGTHS_FILE_NAME}'
        token_cache_dir = f'{DIRECTORY_TOKENIZED_DOCUMENTS}/{FULL_DOCS}'
        queries_file = f'{DIRECTORY_QUERIES}/{QUERIES[0]}'
        queries_file_delimiter = QUERIES[1]
        queries_ranking = f'{DIRECTORY_RANKINGS}/{QUERIES_RANKING}'
        k = 10  # top k ranked documents are returned

    # Create a StringIO buffer to capture the profiling results
    profile_buffer = io.StringIO()
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # init tokenizer object
    tokenizer = Tokenizer()

    # init indexer, used to create inverted index
    indexer = Indexer(tokenizer=tokenizer, save_tokenization=SAVE_TOKENIZED_DOCUMENTS,
                      load_tokenization=LOAD_TOKENIZED_DOCUMENTS, token_cache_directory=token_cache_dir)

    # let indexer create inverted index from the specified documents directory
    inverted_index = indexer.create_index_from_directory(directory=documents_dir)

    inverted_index.get_index_size_in_bytes()

    # save the inverted index and document lengths to files
    inverted_index.save(index_filename=index_file,
                        lengths_filename=doc_lengths_file)  # Saving lengths as a separate file

    # load the inverted index and document lengths from files
    # loaded_index = InvertedIndex.load(index_file, doc_lengths_file)

    # object to rank documents given a query
    document_ranker = DocumentRanker(tokenizer=tokenizer, inverted_index=inverted_index)

    document_ranker.rank_queries_from_file(input_file=queries_file, output_file=queries_ranking,
                                           delimiter=queries_file_delimiter, top_k=k)

    # stop profiling
    profiler.disable()

    # profiling results
    pstats.Stats(profiler, stream=profile_buffer).sort_stats('cumulative').print_stats("src")
    print(profile_buffer.getvalue())

    return 0


if __name__ == "__main__":
    main()
