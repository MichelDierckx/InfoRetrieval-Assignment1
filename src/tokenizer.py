from typing import List
from types import Token
import nltk


class Tokenizer:

    def __init__(self):
        pass

    @staticmethod
    def tokenize(text: str) -> List[Token]:
        """
        Inspired by https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline/#14_Normalization
        :param text:
        :return:
        """
        # remove leading and trailing white space
        text = text.strip()

        # replace multiple consecutive white space characters with a single space
        text = " ".join(text.split())

        # tokenize the text
        tokens = nltk.word_tokenize(text)

        # lowercase the tokens
        lowercased_tokens = [token.lower() for token in tokens]

        # get list of stopwords in English
        stopwords = nltk.corpus.stopwords.words("english")

        # remove stopwords
        filtered_tokens = [token for token in lowercased_tokens if token not in stopwords]

        # create lemmatizer object
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # lemmatize each token
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        return lemmatized_tokens
