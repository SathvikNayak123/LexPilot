import re


class CitationExtractor:
    """Extract legal citations from LLM-generated text."""

    PATTERNS = [
        # SCC modern canonical: (YYYY) N SCC PAGE
        (r'\(\d{4}\)\s+\d+\s+SCC\s+\d+', "SCC"),
        # SCC SUPP with parens year: (YYYY) Supp (N) SCC PAGE
        (r'\(\d{4}\)\s+Supp\s*\(\d+\)\s+SCC\s+\d+', "SCC"),
        # SCC SUPP: YYYY SUPP (N) SCC PAGE
        (r'\d{4}\s+SUPP\s*\(\d+\)\s+SCC\s+\d+', "SCC"),
        # SCC reporter-before-volume: YYYY SCC (N) PAGE
        (r'\d{4}\s+SCC\s+\(\d+\)\s+\d+', "SCC"),
        # SCC old IndianKanoon: YYYY (N) SCC PAGE
        (r'\d{4}\s+\(\d+\)\s+SCC\s+\d+', "SCC"),
        # SCC no-parens: YYYY N SCC PAGE (must come after more specific patterns)
        (r'\d{4}\s+\d+\s+SCC\s+\d+', "SCC"),
        # AIR
        (r'AIR\s+\d{4}\s+SC\s+\d+', "AIR"),
        # SCC Online
        (r'\d{4}\s+SCC\s+OnLine\s+\w+\s+\d+', "SCC_Online"),
        # SCR
        (r'\[\d{4}\]\s+\d+\s+SCR\s+\d+', "SCR"),
        # ILR
        (r'ILR\s+\d{4}\s+\w+\s+\d+', "ILR"),
    ]

    def extract(self, text: str) -> list[dict]:
        """Extract all citations from text with their positions.

        Patterns are tried in order (most specific first). A match is
        skipped if its span overlaps with an already-extracted citation
        to prevent duplicates from less-specific patterns.
        """
        citations = []
        taken_spans: list[tuple[int, int]] = []

        for pattern, source_type in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start, end = match.start(), match.end()
                # Skip if this span overlaps with an already-matched citation
                if any(s < end and start < e for s, e in taken_spans):
                    continue
                citations.append({
                    "citation": match.group(),
                    "source_type": source_type,
                    "start": start,
                    "end": end,
                })
                taken_spans.append((start, end))

        return citations
