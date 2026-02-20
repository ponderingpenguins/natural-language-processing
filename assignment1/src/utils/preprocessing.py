"""
Text preprocessing functions for the AG News dataset.
"""

import string

import nltk  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)  # for stopword removal
nltk.download("wordnet", quiet=True)  # for lemmatization

# Initialize the lemmatizer and stemmer
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


# Preprocessing functions
def remove_whitespace(text: str) -> str:
    """Remove extra whitespace from the text."""

    return " ".join(text.split())


def remove_punctuation(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text: str, language: str = "english") -> str:
    """Remove stopwords from the text."""
    stop_words = set(stopwords.words(language))
    return " ".join(word for word in text.split() if word not in stop_words)


def lemmatize_text(text: str) -> str:
    """Lemmatize the text."""
    return " ".join(LEMMATIZER.lemmatize(word) for word in text.split())

def stem_text(text: str) -> str:
    """Stem the text."""
    return " ".join(STEMMER.stem(word) for word in text.split())


def apply_preprocessing_pipeline(text: str, pipeline: dict) -> str:
    """Apply a preprocessing pipeline to the text."""
    for _, func in pipeline.items():
        text = func(text)
    return text


def preprocess_dataset(dataset, pipeline):
    """Preprocess the dataset using the provided pipeline.

    Creates two fields:
        - raw_text: title + description combined, no transformations.
        - text: title + description after the preprocessing pipeline.
    """
    return dataset.map(
        lambda x: {
            "raw_text": x["title"] + " " + x["description"],
            "text": apply_preprocessing_pipeline(x["title"], pipeline)
            + " "
            + apply_preprocessing_pipeline(x["description"], pipeline),
        }
    )


<<<<<<< HEAD
def build_preprocessing_pipeline(cfg) -> dict:
    """Build a preprocessing pipeline based on the training config."""
    pipeline = {
        "lowercase": lambda x: x.lower(),  # Convert text to lowercase
        "remove_whitespace": remove_whitespace,  # Remove extra whitespace
        "remove_punctuation": remove_punctuation,  # Remove punctuation
    }
    if cfg.remove_stopwords:
        pipeline["remove_stopwords"] = lambda x: remove_stopwords(
            x, language=cfg.stopword_language
        )
    pipeline["lemmatization"] = lemmatize_text  # Lemmatize the text
    return pipeline
=======
text_preprocessing_pipeline = {
    "lowercase": lambda x: x.lower(),  # Convert text to lowercase
    "remove_whitespace": remove_whitespace,  # Remove extra whitespace
    "remove_punctuation": remove_punctuation,  # Remove punctuation
    "remove_stopwords": remove_stopwords,  # Remove stopwords
    "lemmatization": lemmatize_text,  # Lemmatize the text
    "stemming": stem_text,  # Stem the text
}
>>>>>>> a8487ba4d28503544bc7e3d68c17b0c680fc04dc
