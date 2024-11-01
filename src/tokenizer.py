"""
Contains the Tokenizer class, which can be used to convert a text into a list of tokens.
"""

import os
from typing import List

import nltk


class Tokenizer:
    def __init__(self):
        self.set_nltk_data_dir()  # Change directory where nltk looks for data to src/nltk_data
        self.lemmatizer = nltk.stem.WordNetLemmatizer()  # lemmatizer object
        self.stopwords = set(nltk.corpus.stopwords.words("english"))  # set of english stopwords
        self.tokenizer = nltk.RegexpTokenizer(r'\w+')  # regular expression to only match sequences of word characters

    def set_nltk_data_dir(self):
        """
        Change directory where nltk looks for data to src/nltk_data
        """
        # Construct the path to the src/nltk_data
        nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')

        # change directories where nltk will look to src/nltk_data
        nltk.data.path = [nltk_data_dir]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: input text
        :return: List of tokens
        """
        tokens = self.tokenizer.tokenize(text)  # tokenize
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token.lower())  # convert to lowercase + lematize
            for token in tokens
            if token.lower() not in self.stopwords  # remove stopwords
        ]
        return lemmatized_tokens
