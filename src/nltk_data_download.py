"""
Run this python program to download NLTK datasets
"""

import os
import ssl

import nltk


def setup_nltk_data():
    """
    Change directory where nltk looks for data to src/nltk_data. Creates this directory if it doesn't exist
    """
    # Construct the path to the src/nltk_data
    nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

    # Create src/nltk_data if not exists
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)
        print(f"Created directory: {nltk_data_dir}")

    # change directories where nltk will look to src/nltk_data
    nltk.data.path = [nltk_data_dir]


def download_nltk_resources():
    """
    Download the necessary NLTK resources to src/nltk_data
    """
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    print(f"Downloading to: {nltk.data.path[0]}")
    try:
        nltk.download('punkt_tab', download_dir=nltk.data.path[0])
        nltk.download('stopwords', download_dir=nltk.data.path[0])
        nltk.download('wordnet', download_dir=nltk.data.path[0])
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading NLTK resources: {e}")


def main():
    setup_nltk_data()
    download_nltk_resources()


if __name__ == "__main__":
    main()
