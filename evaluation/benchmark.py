import json
from pathlib import Path


class LegalBenchmark:
    """Manages benchmark datasets for LexPilot evaluation."""

    def __init__(self, benchmark_dir: str = "evaluation/benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)

    def load_retrieval_benchmark(self) -> list[dict]:
        """50 legal questions with known relevant judgments.
        Tests retrieval accuracy independent of generation."""
        path = self.benchmark_dir / "retrieval_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return self._generate_retrieval_benchmark()

    def load_compliance_benchmark(self) -> list[dict]:
        """20 annotated Indian privacy policies with known DPDP gaps.
        Tests compliance scanner recall."""
        path = self.benchmark_dir / "compliance_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return []

    def load_qa_benchmark(self) -> list[dict]:
        """100 legal Q&A pairs for end-to-end evaluation."""
        path = self.benchmark_dir / "qa_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return self._generate_qa_benchmark()

    def _generate_retrieval_benchmark(self) -> list[dict]:
        """Generate starter retrieval benchmark cases."""
        return [
            {"query": "Right to privacy as fundamental right", "relevant_citations": ["(2017) 10 SCC 1"]},
            {"query": "Defamation under Indian Penal Code", "relevant_citations": ["AIR 1962 SC 305"]},
            {"query": "DPDP consent requirements", "relevant_doc_types": ["statute"]},
            {"query": "Doctrine of basic structure", "relevant_citations": ["(1973) 4 SCC 225"]},
            {"query": "Right to information and privacy balance", "relevant_citations": ["(2017) 10 SCC 1"]},
            {"query": "Arbitration clause enforceability", "relevant_citations": ["(2012) 9 SCC 552"]},
            {"query": "Specific performance of contract", "relevant_citations": ["AIR 2016 SC 521"]},
            {"query": "Employer liability for workplace harassment", "relevant_citations": ["(1997) 6 SCC 241"]},
            {"query": "Environmental clearance requirements", "relevant_citations": ["AIR 1987 SC 1086"]},
            {"query": "Fundamental right to education", "relevant_citations": ["(1993) 1 SCC 645"]},
            {"query": "Freedom of speech and reasonable restrictions", "relevant_citations": ["AIR 1950 SC 124"]},
            {"query": "Triple talaq constitutional validity", "relevant_citations": ["(2017) 9 SCC 1"]},
            {"query": "Right to clean environment", "relevant_citations": ["AIR 1987 SC 1086"]},
            {"query": "Dowry death presumption under Section 304B", "relevant_citations": ["(2012) 6 SCC 1"]},
            {"query": "SEBI insider trading regulations", "relevant_citations": ["(2014) 8 SCC 883"]},
            {"query": "Natural justice principles in administrative law", "relevant_citations": ["AIR 1978 SC 597"]},
            {"query": "Bail conditions for economic offences", "relevant_citations": ["(2018) 11 SCC 1"]},
            {"query": "Marital rape exception constitutionality", "relevant_citations": []},
            {"query": "Consumer protection for deficiency in service", "relevant_citations": ["(2009) 9 SCC 221"]},
            {"query": "Land acquisition compensation principles", "relevant_citations": ["(2014) 3 SCC 183"]},
            {"query": "Trademark infringement test", "relevant_citations": ["(2004) 6 SCC 145"]},
            {"query": "Constitutionality of Section 66A IT Act and online free speech", "relevant_citations": ["(2015) 5 SCC 1"]},
            {"query": "Cheque bounce liability under NI Act", "relevant_citations": ["(2014) 12 SCC 539"]},
            {"query": "Anticipatory bail principles", "relevant_citations": ["(1980) 2 SCC 565"]},
            {"query": "Decriminalization of Section 377 and LGBTQ rights", "relevant_citations": ["(2018) 10 SCC 1"]},
            {"query": "Tenancy rights in rent control", "relevant_citations": ["AIR 1990 SC 396"]},
            {"query": "Succession rights of daughters Hindu law", "relevant_citations": ["(2020) 9 SCC 1"]},
            {"query": "Motor vehicle accident compensation", "relevant_citations": ["(2017) 16 SCC 680"]},
            {"query": "Writ jurisdiction of High Court", "relevant_citations": ["AIR 1967 SC 1"]},
            {"query": "Cyber crime jurisdiction issues", "relevant_citations": []},
            {"query": "GST classification disputes", "relevant_citations": []},
            {"query": "RERA compliance for builders", "relevant_citations": []},
            {"query": "Insolvency resolution process timeline", "relevant_citations": ["(2019) 2 SCC 1"]},
            {"query": "Employee provident fund withdrawal rules", "relevant_citations": []},
            {"query": "Women entry in Sabarimala temple and gender discrimination", "relevant_citations": ["(2018) 8 SCC 1"]},
            {"query": "Domestic violence protection orders", "relevant_citations": ["(2005) 15 SCC 733"]},
            {"query": "Section 498A misuse safeguards", "relevant_citations": ["(2014) 4 SCC 1"]},
            {"query": "Right to die with dignity and passive euthanasia", "relevant_citations": ["(2018) 5 SCC 1"]},
            {"query": "Power of attorney validity requirements", "relevant_citations": ["(2012) 5 SCC 370"]},
            {"query": "Partnership firm dissolution", "relevant_citations": ["AIR 1966 SC 1"]},
            {"query": "Specific relief for breach of contract", "relevant_citations": ["AIR 2016 SC 521"]},
            {"query": "Public interest litigation standing", "relevant_citations": ["(1982) 1 SCC 618"]},
            {"query": "Death penalty rarest of rare doctrine", "relevant_citations": ["(1980) 2 SCC 684"]},
            {"query": "Juvenile justice age determination", "relevant_citations": ["(2017) 1 SCC 416"]},
            {"query": "Transfer of property gift requirements", "relevant_citations": ["AIR 1970 SC 999"]},
            {"query": "Banking fraud investigation powers", "relevant_citations": []},
            {"query": "Data localization requirements India", "relevant_citations": []},
            {"query": "Personal data breach notification DPDP", "relevant_citations": []},
            {"query": "Cross-border data transfer conditions", "relevant_citations": []},
            {"query": "Consent withdrawal rights under DPDP", "relevant_citations": []},
        ]

    def _generate_qa_benchmark(self) -> list[dict]:
        """Generate starter QA benchmark cases."""
        return [
            {
                "question": "What are the rights of a Data Principal under DPDP Act 2023?",
                "ground_truth": "Under Section 9, Data Principals have rights to access, correction, erasure, grievance redressal, and nomination.",
                "category": "factual",
            },
            {
                "question": "When can personal data be processed without consent under DPDP?",
                "ground_truth": "Under Section 6, consent is not needed for voluntary provision, state functions, medical emergencies, and employment purposes.",
                "category": "factual",
            },
            {
                "question": "What is the doctrine of basic structure?",
                "ground_truth": "Established in Kesavananda Bharati v. State of Kerala (1973), it holds that Parliament cannot amend the Constitution to destroy its basic structure.",
                "category": "factual",
            },
            {
                "question": "What are the obligations of a Significant Data Fiduciary under DPDP?",
                "ground_truth": "Under Section 8, they must appoint a Data Protection Officer, conduct periodic data audits, and undertake data protection impact assessments.",
                "category": "factual",
            },
            {
                "question": "What penalties does the DPDP Act prescribe for data breaches?",
                "ground_truth": "The Act prescribes penalties up to Rs 250 crore for failure to take reasonable security safeguards to prevent personal data breach.",
                "category": "factual",
            },
            {
                "question": "What is the right to privacy in Indian constitutional law?",
                "ground_truth": "In K.S. Puttaswamy v. Union of India (2017), the Supreme Court declared right to privacy a fundamental right under Article 21.",
                "category": "factual",
            },
            {
                "question": "What constitutes sexual harassment at workplace under Indian law?",
                "ground_truth": "Defined in Vishaka v. State of Rajasthan (1997) and codified in the Sexual Harassment of Women at Workplace Act, 2013.",
                "category": "factual",
            },
            {
                "question": "What is the test for granting anticipatory bail?",
                "ground_truth": "Per Gurbaksh Singh Sibbia v. State of Punjab (1980), courts consider nature of accusation, applicant's role, likelihood of fleeing, and possibility of tampering.",
                "category": "factual",
            },
            {
                "question": "When is specific performance of contract granted?",
                "ground_truth": "Under the Specific Relief Act 1963 (as amended 2018), specific performance is granted as a general rule unless damages would be adequate relief.",
                "category": "factual",
            },
            {
                "question": "What are the conditions for transfer of personal data outside India under DPDP?",
                "ground_truth": "Under Section 16, the Central Government may restrict transfer to notified countries. Transfer is permitted unless restricted.",
                "category": "factual",
            },
        ]
