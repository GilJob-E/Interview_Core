"""Korean morphological analyzer wrapper using Kiwi."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from kiwipiepy import Kiwi


@dataclass
class Token:
    """A single morphological token.

    Attributes:
        form: The surface form of the token.
        tag: The POS tag (Kiwi format).
        start: Start position in original text.
        end: End position in original text.
        score: Kiwi's confidence score.
    """
    form: str
    tag: str
    start: int
    end: int
    score: float = 0.0

    def __repr__(self) -> str:
        return f"Token({self.form}/{self.tag})"


class KiwiTokenizer:
    """Wrapper for Kiwi Korean morphological analyzer.

    This class provides a simplified interface for tokenizing Korean text
    and extracting morphological information needed for LIWC feature extraction.

    Example:
        >>> tokenizer = KiwiTokenizer()
        >>> tokens = tokenizer.tokenize("안녕하세요, 저는 김철수입니다.")
        >>> for token in tokens:
        ...     print(f"{token.form}/{token.tag}")
    """

    def __init__(self, num_workers: int = 0, integrate_allomorph: bool = True) -> None:
        """Initialize Kiwi tokenizer.

        Args:
            num_workers: Number of worker threads (0 for auto).
            integrate_allomorph: Whether to integrate allomorphs (이/가, 을/를 etc).
        """
        self._kiwi = Kiwi(num_workers=num_workers, integrate_allomorph=integrate_allomorph)

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize Korean text into morphological tokens.

        Args:
            text: Korean text to tokenize.

        Returns:
            List of Token objects with POS tags.
        """
        if not text or not text.strip():
            return []

        result = self._kiwi.tokenize(text)
        tokens = []
        for token in result:
            tokens.append(Token(
                form=token.form,
                tag=token.tag,
                start=token.start,
                end=token.end,
                score=token.score
            ))
        return tokens

    def tokenize_batch(self, texts: List[str]) -> List[List[Token]]:
        """Tokenize multiple texts in batch.

        Args:
            texts: List of Korean texts to tokenize.

        Returns:
            List of token lists, one per input text.
        """
        results = []
        for text in texts:
            results.append(self.tokenize(text))
        return results

    def get_morphemes(self, text: str) -> List[str]:
        """Get list of morpheme forms only.

        Args:
            text: Korean text to tokenize.

        Returns:
            List of morpheme surface forms.
        """
        return [token.form for token in self.tokenize(text)]

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get list of (morpheme, POS tag) pairs.

        Args:
            text: Korean text to tokenize.

        Returns:
            List of (form, tag) tuples.
        """
        return [(token.form, token.tag) for token in self.tokenize(text)]

    def count_morphemes(self, text: str) -> int:
        """Count total number of morphemes in text.

        Args:
            text: Korean text to tokenize.

        Returns:
            Number of morphemes.
        """
        return len(self.tokenize(text))

    def count_unique_morphemes(self, text: str) -> int:
        """Count unique morphemes in text.

        Args:
            text: Korean text to tokenize.

        Returns:
            Number of unique morphemes.
        """
        tokens = self.tokenize(text)
        return len(set(token.form for token in tokens))

    def filter_by_pos(self, text: str, pos_tags: List[str]) -> List[Token]:
        """Filter tokens by POS tags.

        Args:
            text: Korean text to tokenize.
            pos_tags: List of POS tags to include.

        Returns:
            Filtered list of tokens.
        """
        return [token for token in self.tokenize(text) if token.tag in pos_tags]
