"""
NLP Pipeline
=============
Core NLP processing module for the Resume Analyzer.
Implements tokenization, stop-word removal, lemmatization,
N-gram extraction, POS tagging, chunking, and NER.
"""

import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Set

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

import spacy

# ---------------------------------------------------------------------------
# NLTK data download (safe to call multiple times)
# ---------------------------------------------------------------------------
NLTK_PACKAGES = [
    "punkt", "punkt_tab", "stopwords", "wordnet",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
    "omw-1.4",
]

def ensure_nltk_data():
    """Download required NLTK data packages if not already present."""
    for pkg in NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else pkg)
        except LookupError:
            nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# spaCy model loader
# ---------------------------------------------------------------------------
_nlp_spacy = None

def get_spacy_nlp():
    """Load the spaCy model (cached)."""
    global _nlp_spacy
    if _nlp_spacy is None:
        try:
            _nlp_spacy = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            _nlp_spacy = spacy.load("en_core_web_sm")
    return _nlp_spacy


# ===================================================================
# TEXT CLEANING
# ===================================================================

def clean_text(text: str) -> str:
    """Basic text cleanup: collapse whitespace, strip stray chars."""
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = text.strip()
    return text


# ===================================================================
# EXPERIMENT 2 & 3 — Tokenization, Stop-word Removal, Lemmatization
# ===================================================================

def tokenize(text: str) -> List[str]:
    """Word-level tokenization using NLTK."""
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stop words and punctuation."""
    stop_words = set(stopwords.words("english"))
    return [
        t for t in tokens
        if t.lower() not in stop_words and t not in string.punctuation
        and len(t) > 1
    ]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens using WordNet lemmatizer."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]


def preprocess(text: str) -> List[str]:
    """Full preprocessing pipeline: clean → tokenize → remove stop words → lemmatize."""
    text = clean_text(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return tokens


# ===================================================================
# EXPERIMENT 5 — N-gram Extraction
# ===================================================================

def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Extract n-grams as joined strings."""
    return [" ".join(gram) for gram in ngrams(tokens, n)]


def extract_all_ngrams(text: str, max_n: int = 3) -> Dict[int, List[str]]:
    """Extract unigrams through max_n-grams from raw text."""
    tokens = preprocess(text)
    result = {1: tokens}
    for n in range(2, max_n + 1):
        result[n] = extract_ngrams(tokens, n)
    return result


# ===================================================================
# EXPERIMENT 6 — POS Tagging
# ===================================================================

def pos_tagging(text: str) -> List[Tuple[str, str]]:
    """POS-tag the cleaned text using NLTK."""
    tokens = word_tokenize(clean_text(text))
    return pos_tag(tokens)


# ===================================================================
# EXPERIMENT 7 — Chunking (Noun Phrase extraction)
# ===================================================================

def extract_noun_phrases_nltk(text: str) -> List[str]:
    """Extract noun phrases via regex chunking (NLTK)."""
    tagged = pos_tagging(text)
    # Grammar: optional determiner, any number of adjectives, one or more nouns
    grammar = r"NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    tree = cp.parse(tagged)

    noun_phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
        phrase = " ".join(word for word, tag in subtree.leaves())
        noun_phrases.append(phrase)
    return noun_phrases


def extract_noun_phrases_spacy(text: str) -> List[str]:
    """Extract noun phrases using spaCy."""
    nlp = get_spacy_nlp()
    doc = nlp(clean_text(text))
    return [chunk.text for chunk in doc.noun_chunks]


# ===================================================================
# EXPERIMENT 8 — Named Entity Recognition
# ===================================================================

def extract_entities_nltk(text: str) -> Dict[str, List[str]]:
    """NER using NLTK's ne_chunk."""
    tagged = pos_tagging(text)
    tree = ne_chunk(tagged)
    entities: Dict[str, List[str]] = {}
    for subtree in tree:
        if isinstance(subtree, Tree):
            label = subtree.label()
            entity = " ".join(word for word, tag in subtree.leaves())
            entities.setdefault(label, []).append(entity)
    return entities


def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """NER using spaCy — returns {label: [entity, …]}."""
    nlp = get_spacy_nlp()
    doc = nlp(clean_text(text))
    entities: Dict[str, List[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


# ===================================================================
# SKILL EXTRACTION (uses n-grams + skills DB)
# ===================================================================

def extract_skills(text: str, skills_set: Set[str]) -> Set[str]:
    """
    Match skills from *skills_set* against the text.
    Uses unigrams, bigrams, and trigrams to catch multi-word skills
    like 'machine learning' or 'natural language processing'.
    """
    text_lower = clean_text(text).lower()
    found: Set[str] = set()

    # Direct substring matching for multi-word skills
    for skill in skills_set:
        # Use word-boundary–aware regex to avoid partial matches
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    return found


# ===================================================================
# KEYWORD FREQUENCY
# ===================================================================

def keyword_frequency(text: str, top_n: int = 15) -> List[Tuple[str, int]]:
    """Return the most common preprocessed tokens."""
    tokens = preprocess(text)
    return Counter(tokens).most_common(top_n)
