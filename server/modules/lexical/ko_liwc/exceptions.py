"""Custom exceptions for ko-liwc package."""


class KoLIWCError(Exception):
    """Base exception for ko-liwc package."""
    pass


class TokenizationError(KoLIWCError):
    """Error during text tokenization."""
    pass


class FeatureExtractionError(KoLIWCError):
    """Error during feature extraction."""
    pass


class ScoringError(KoLIWCError):
    """Error during score calculation."""
    pass


class InvalidTranscriptError(KoLIWCError):
    """Error for invalid transcript data."""
    pass


class DictionaryNotFoundError(KoLIWCError):
    """Error when required dictionary is not found."""
    pass
