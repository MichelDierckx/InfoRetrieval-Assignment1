import cProfile
import io
import pstats

from positional_index import SPIMIIndexer

FULL_DOCS_SMALL_DIRECTORY = "data/documents/full_docs_small"
FULL_DOCS_DIRECTORY = "data/documents/full_docs"
FULL_DOCS_SMALL_INDEX_DIRECTORY = "data/saved_indexes/full_docs_small"
FULL_DOCS_INDEX_DIRECTORY = "data/saved_indexes/full_docs"
MEMORY_LIMIT = 10000

DEV = True


def main():
    """ Main program """
    if DEV:
        documents_dir = FULL_DOCS_SMALL_DIRECTORY
        index_dir = FULL_DOCS_SMALL_INDEX_DIRECTORY
    else:
        documents_dir = FULL_DOCS_DIRECTORY
        index_dir = FULL_DOCS_INDEX_DIRECTORY

    # Create a StringIO buffer to capture the profiling results
    profile_buffer = io.StringIO()
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    spimi_indexer = SPIMIIndexer(index_dir)

    final_index = spimi_indexer.create_index_from_directory(documents_dir, MEMORY_LIMIT)

    tolerate = final_index.positional_index.get("tolerate")  # Use .get() to avoid KeyError if not found
    if tolerate:
        tolerate.pretty_print()

    # Stop profiling
    profiler.disable()

    # Print profiling results (only from module src)
    pstats.Stats(profiler, stream=profile_buffer).sort_stats('cumulative').print_stats("src")
    print(profile_buffer.getvalue())

    return 0


if __name__ == "__main__":
    main()
