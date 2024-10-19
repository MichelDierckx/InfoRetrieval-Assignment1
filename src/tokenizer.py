"""
Contains the Tokenizer class, which can be used to convert a text into a list of tokens.
"""

import os
import string
from typing import List

import nltk

from src.types import Token


class Tokenizer:
    def __init__(self):
        self.set_nltk_data_dir()  # Change directory where nltk looks for data to src/nltk_data

    def set_nltk_data_dir(self):
        """
        Change directory where nltk looks for data to src/nltk_data
        """
        # Construct the path to the src/nltk_data
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

        # change directories where nltk will look to src/nltk_data
        nltk.data.path = [nltk_data_dir]

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize the input text.
        :param text: input text
        :return: List of tokens
        """
        # Remove leading and trailing white space
        text = text.strip()

        # Replace multiple consecutive white space characters with a single space
        text = " ".join(text.split())

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # remove punctuation
        tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]

        # Lowercase the tokens
        lowercased_tokens = [token.lower() for token in tokens_without_punctuation]

        # Get list of stopwords in English
        stopwords = nltk.corpus.stopwords.words("english")

        # Remove stopwords
        tokens_without_stopwords = [token for token in lowercased_tokens if token not in stopwords]

        # Create lemmatizer object
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # Lemmatize each token
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_stopwords]

        return lemmatized_tokens
