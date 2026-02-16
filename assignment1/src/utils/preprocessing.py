import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords", quiet=True)  # for stopword removal
nltk.download("wordnet", quiet=True)  # for lemmatization

# Initialize the lemmatizer
LEMMATIZER = WordNetLemmatizer()


# Preprocessing functions
def remove_whitespace(text: str) -> str:
    """Remove extra whitespace from the text."""

    return " ".join(text.split())


def remove_punctuation(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(text: str) -> str:
    """Remove stopwords from the text."""
    stop_words = set(stopwords.words("english"))
    return " ".join(word for word in text.split() if word not in stop_words)


def lemmatize_text(text: str) -> str:
    """Lemmatize the text."""
    return " ".join(LEMMATIZER.lemmatize(word) for word in text.split())


def apply_preprocessing_pipeline(text: str, pipeline: dict) -> str:
    """Apply a preprocessing pipeline to the text."""
    for _, func in pipeline.items():
        text = func(text)
    return text


def preprocess_dataset(dataset, pipeline):
    """Preprocess the dataset using the provided pipeline."""
    return dataset.map(
        lambda x: {
            "text": apply_preprocessing_pipeline(x["title"], pipeline)
            + " "
            + apply_preprocessing_pipeline(x["description"], pipeline)
        }
    )


text_preprocessing_pipeline = {
    "lowercase": lambda x: x.lower(),  # Convert text to lowercase
    "remove_whitespace": remove_whitespace,  # Remove extra whitespace
    "remove_punctuation": remove_punctuation,  # Remove punctuation
    "remove_stopwords": remove_stopwords,  # Remove stopwords
    "lemmatization": lemmatize_text,  # Lemmatize the text
}
