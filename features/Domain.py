# Imports
from collections import Counter

import nltk
import numpy as np
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from openai import OpenAI
from scipy.stats import entropy
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)

# Definitions
nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")


class Domain:
    """
    This class represents a dataset of text documents in a single domain.
    @field name - a string describing the domain.
    @field documents - a list of lists of words corresponding to each text in the domain.
    @field vocab - a set of all words (containing no special characters, no stop words, and lemmatized) in this domain
    @field sentence_embeddings - list of sentence embeddings in each text
    @field word_count - total word count
    @field token_frequencies - dictionary of tokens to their relative frequency in domain
    @field vocab_size - size of unique vocabulary
    @field dataset_size - number of texts in this domain corpus
    @field tfidf_topwords - TFIDD weighted top words
    """

    def __init__(self, documents: list, file_names: list):
        """
        Create a Domain.
        @param file_name - a string describing the domain.
        @param file_paths - a list of string paths of the json files to be used to construct the domains.
        """
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        self.name = file_names
        self.documents = documents
        documents_tokenized, self.sentence_embeddings = self.download(documents)
        self.domain_words = [
            word for document in documents_tokenized for word in document
        ]
        self.word_count = len(self.domain_words)
        self.token_frequencies = Counter(self.domain_words)
        self.dataset_size = len(self.documents)
        self.tfidf_top_words = self.compute_tfidf()
        self.prob_dist = self.get_prob_dist()
        self.shannon_entropy = self.compute_shannon_entropy()

    def __str__(self):
        return f"""
                    ------- Domain Summary ------- \n
                    Dataset: {self.name}\n
                    Number of Documents: {self.dataset_size}\n
                    
                    Total Word Count: {self.word_count}\n"""

    def download(self, data) -> tuple:
        """ "
        Processes source and target json files to obtain processed source domain text lists and target domain text datasets.
        @param self - the current Domain datasets to constructs.
        @param file_paths - a list of string paths of the json files used to construct the domains.
        @returns - a list of lists of words corresponding to each text in the domain corpus and a list of the corresponding sentence embeddings.
        """

        domain_words = []
        sentence_embeddings = []
        # print(f"Processing {self.name} data.")
        for i in range(len(data)):
            document = self.get_article_vocab(data[i])
            domain_words.append(document)
            article = data[i]
            if len(article.split()) > 4000:
                article = "".join(article[:4000])
            sentence_embeddings.append(self.compute_sentence_embeddings(article))
        return domain_words, sentence_embeddings

    def get_article_vocab(self, text: str) -> list:
        tokens = word_tokenize(str(text).lower())

        # Remove special characters from each token
        # tokens = set([re.sub('[^a-zA-Z0-9]+', '', _) for _ in tokens])
        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        special_characters = [
            ",",
            ".",
            "(",
            ")",
            "'",
            "’",
            ";",
            ":",
            "!",
            "@",
            "#",
            "$",
            "%",
            "^",
            "&",
            "*",
            "-",
            "_",
            "+",
            "=",
            "[",
            "]",
            "}",
            "{",
            "<",
            ">",
            "/",
            "~",
            "``",
            "``",
            "\\",
        ]
        tokens = [
            token
            for token in tokens
            if (
                token not in stop_words
                and token not in special_characters
                and token is not None
            )
        ]

        # Remove all numbers
        # tokens = set(val for val in tokens if not val.isdigit())

        return tokens

    def compute_tfidf(self):
        """
        Computes TFIDF for domain corpus.
        @param self - this Domain class.
        @returns vectorizer, array - corresponding TFIDF vectorizor and vectorized array of documents
        """
        tfidf_vectorizer = TfidfVectorizer(
            smooth_idf=True,
            use_idf=True,
            stop_words="english",
        )  # max_df=0.10, min_df=0.01,)
        tfidf = tfidf_vectorizer.fit_transform(self.documents)
        n_top = 10000
        importance = np.argsort(np.asarray(tfidf.sum(axis=0)).ravel())[::-1]
        tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
        tfidf_top_words = tfidf_feature_names[importance[:n_top]]
        return tfidf_top_words

    def compute_sentence_embeddings(self, text):
        """
        Convert sentences to OpenAI embeddings vectors.
        @param self - this Domain class.
        @returns array - vector embedding
        """
        client = OpenAI()
        response = (
            client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1024,
            )
            .data[0]
            .embedding
        )
        return response

    def get_prob_dist(self):
        prob_dist = dict()
        for word in self.token_frequencies:
            prob_dist[word] = self.token_frequencies[word] / self.word_count
        prob_dist = {
            k: v for k, v in sorted(prob_dist.items(), key=lambda item: item[1])
        }
        return prob_dist

    def compute_shannon_entropy(self):
        prob_dist = list(self.prob_dist.values())
        shannon = entropy(prob_dist)
        return shannon

    def compute_learning_difficulty(self, alpha=0.7, beta=0.3, max_length=100000):
        # print("Documents:", self.documents)

        def chunk_text(text, max_length):
            """Divide a long text into smaller chunks based on max_length."""
            for i in range(0, len(text), max_length):
                yield text[i : i + max_length]

        # Combine all text into one string
        combined_text = " ".join(
            text.strip() for text in self.documents if text.strip()
        )

        # Process text in chunks
        total_dependency_length = 0
        total_dependencies = 0
        max_tree_depth = 0

        for chunk in chunk_text(combined_text, max_length):
            doc = nlp(chunk)

            for sentence in doc.sents:
                # Calculate dependency length
                for token in sentence:
                    dependency_length = abs(token.i - token.head.i)
                    total_dependency_length += dependency_length
                    total_dependencies += 1

                # Calculate tree depth
                def calculate_depth(token):
                    if not list(token.children):
                        return 1
                    return 1 + max(calculate_depth(child) for child in token.children)

                tree_depth = max(calculate_depth(token) for token in sentence)
                max_tree_depth = max(max_tree_depth, tree_depth)

        # Compute average dependency length
        avg_dependency_length = (
            total_dependency_length / total_dependencies
            if total_dependencies > 0
            else 0
        )

        # Calculate the combined syntactic learning difficulty score
        syntactic_difficulty = alpha * avg_dependency_length + beta * max_tree_depth

        # Return results
        return syntactic_difficulty
