from citation.extractor import CitationExtractor


def test_extracts_modern_scc():
    extractor = CitationExtractor()
    text = "See Vishaka v. State of Rajasthan (1997) 6 SCC 241 for guidance."
    citations = extractor.extract(text)

    assert len(citations) == 1
    assert citations[0]["citation"] == "(1997) 6 SCC 241"
    assert citations[0]["source_type"] == "SCC"


def test_extracts_air():
    extractor = CitationExtractor()
    text = "AIR 1973 SC 1461 established the basic structure doctrine."
    citations = extractor.extract(text)

    assert any(c["source_type"] == "AIR" for c in citations)
    assert any(c["citation"] == "AIR 1973 SC 1461" for c in citations)


def test_extracts_scr_with_brackets():
    extractor = CitationExtractor()
    text = "Reported as [2017] 10 SCR 569."
    citations = extractor.extract(text)

    assert len(citations) == 1
    assert citations[0]["source_type"] == "SCR"


def test_no_overlap_between_patterns():
    extractor = CitationExtractor()
    # The same citation should not match twice via the canonical and no-parens patterns.
    text = "See (2018) 1 SCC 809 for the holding."
    citations = extractor.extract(text)

    assert len(citations) == 1


def test_returns_empty_for_no_citations():
    extractor = CitationExtractor()
    citations = extractor.extract("This text contains no citations whatsoever.")

    assert citations == []


def test_extracts_multiple_distinct_citations():
    extractor = CitationExtractor()
    text = (
        "The court relied on (1997) 6 SCC 241 and AIR 1973 SC 1461, "
        "while distinguishing 2019 SCC OnLine SC 1438."
    )
    citations = extractor.extract(text)

    source_types = {c["source_type"] for c in citations}
    assert {"SCC", "AIR", "SCC_Online"}.issubset(source_types)


def test_citation_positions_are_correct():
    extractor = CitationExtractor()
    text = "Prefix (2018) 1 SCC 809 suffix"
    citations = extractor.extract(text)

    assert len(citations) == 1
    assert text[citations[0]["start"]:citations[0]["end"]] == citations[0]["citation"]
