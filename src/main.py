from positional_index import PositionalIndex

FULL_DOCS_SMALL_DIRECTORY = "data/documents/full_docs_small"
FULL_DOCS_DIRECTORY = "data/documents/full_docs"
FULL_DOCS_SMALL_INDEX = "data/saved_indexes/full_docs_small.json"
FULL_DOCS_INDEX = "data/saved_indexes/full_docs.json"

DEV = True


def main():
    """ Main program """
    if DEV:
        documents_dir = FULL_DOCS_SMALL_DIRECTORY
        index_file = FULL_DOCS_SMALL_INDEX
    else:
        documents_dir = FULL_DOCS_DIRECTORY
        index_file = FULL_DOCS_INDEX

    # Code goes over here.
    positional_index = PositionalIndex()
    positional_index.create_from_directory(documents_dir)
    positional_index.save_to_file(index_file)
    positional_index.load_from_file(index_file)

    positional_index.get_postings_list("tolerate").pretty_print()
    # print(positional_index.get_terms())
    return 0


if __name__ == "__main__":
    main()
