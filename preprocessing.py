import string


# Common English stopwords used for basic NLP preprocessing.
# This local list keeps the project simple and avoids extra downloads.
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "very",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


# Small rule-based word lists for Version 1 sentiment detection.
POSITIVE_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "brilliant",
    "clean",
    "comfortable",
    "easy",
    "enjoy",
    "excellent",
    "fantastic",
    "fast",
    "friendly",
    "good",
    "great",
    "happy",
    "helpful",
    "impressive",
    "like",
    "love",
    "nice",
    "perfect",
    "positive",
    "quality",
    "recommend",
    "satisfied",
    "smooth",
    "useful",
    "wonderful",
}


NEGATIVE_WORDS = {
    "awful",
    "bad",
    "boring",
    "broken",
    "confusing",
    "delay",
    "disappointed",
    "disappointing",
    "expensive",
    "fail",
    "hate",
    "horrible",
    "issue",
    "negative",
    "poor",
    "problem",
    "sad",
    "slow",
    "terrible",
    "unhappy",
    "useless",
    "waste",
    "weak",
    "worst",
}


def preprocess_text(text):
    """
    Apply basic NLP preprocessing and return every step for display.

    Steps:
    1. Keep original text
    2. Convert to lowercase
    3. Remove punctuation
    4. Tokenize by splitting on spaces
    5. Remove stopwords
    """
    original_text = text
    lowercased_text = text.lower()

    # str.translate removes punctuation efficiently using Python's built-in tools.
    cleaned_text = lowercased_text.translate(str.maketrans("", "", string.punctuation))

    raw_tokens = cleaned_text.split()
    tokens = [token for token in raw_tokens if token not in STOPWORDS]
    final_text = " ".join(tokens)

    return {
        "original_text": original_text,
        "lowercased_text": lowercased_text,
        "cleaned_text": cleaned_text,
        "raw_tokens": raw_tokens,
        "tokens": tokens,
        "stopwords_removed": tokens,
        "final_text": final_text,
    }


def analyze_sentiment(tokens):
    """
    Predict sentiment using simple word matching.

    The function compares how many positive and negative words appear in the
    processed tokens. Confidence is intentionally simple for Version 1.
    """
    positive_matches = [word for word in tokens if word in POSITIVE_WORDS]
    negative_matches = [word for word in tokens if word in NEGATIVE_WORDS]

    positive_count = len(positive_matches)
    negative_count = len(negative_matches)
    total_matches = positive_count + negative_count

    if total_matches == 0:
        return {
            "sentiment": "Neutral",
            "confidence": 60,
            "positive_matches": positive_matches,
            "negative_matches": negative_matches,
            "explanation": "No strong positive or negative words were detected.",
        }

    if positive_count > negative_count:
        sentiment = "Positive"
        confidence = calculate_confidence(positive_count, total_matches)
        explanation = "More positive words were detected than negative words."
    elif negative_count > positive_count:
        sentiment = "Negative"
        confidence = calculate_confidence(negative_count, total_matches)
        explanation = "More negative words were detected than positive words."
    else:
        sentiment = "Neutral"
        confidence = 55
        explanation = "Positive and negative signals are balanced."

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_matches": positive_matches,
        "negative_matches": negative_matches,
        "explanation": explanation,
    }


def calculate_confidence(winning_count, total_matches):
    """Calculate a beginner-friendly confidence score between 60 and 95."""
    ratio = winning_count / total_matches
    confidence = int(60 + (ratio * 35))
    return min(confidence, 95)
