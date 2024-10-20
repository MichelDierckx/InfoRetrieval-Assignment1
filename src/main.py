from positional_index import PositionalIndex

FULL_DOCS_SMALL_DIRECTORY = "data/documents/full_docs_small"
FULL_DOCS_SMALL_INDEX = "data/saved_indexes/full_docs_small.json"


def main():
    """ Main program """
    # Code goes over here.
    positional_index = PositionalIndex()
    positional_index.create_from_directory(FULL_DOCS_SMALL_DIRECTORY)
    positional_index.save_to_file(FULL_DOCS_SMALL_INDEX)
    positional_index.load_from_file(FULL_DOCS_SMALL_INDEX)

    # print(positional_index.get_terms())
    return 0


if __name__ == "__main__":
    main()
