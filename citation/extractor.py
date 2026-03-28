import re


class CitationExtractor:
    """Extract legal citations from LLM-generated text."""

    PATTERNS = [
        (r'\(\d{4}\)\s+\d+\s+SCC\s+\d+', "SCC"),
        (r'AIR\s+\d{4}\s+SC\s+\d+', "AIR"),
        (r'\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+', "SCC_Online"),
        (r'\[\d{4}\]\s+\d+\s+SCR\s+\d+', "SCR"),
        (r'ILR\s+\d{4}\s+\w+\s+\d+', "ILR"),
    ]

    def extract(self, text: str) -> list[dict]:
        """Extract all citations from text with their positions."""
        citations = []
        for pattern, source_type in self.PATTERNS:
            for match in re.finditer(pattern, text):
                citations.append({
                    "citation": match.group(),
                    "source_type": source_type,
                    "start": match.start(),
                    "end": match.end(),
                })
        return citations
