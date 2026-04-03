import json
from pathlib import Path


class LegalBenchmark:
    """Manages benchmark datasets for LexPilot evaluation.

    All data is generated inline — no external JSON files required.
    Cached JSON files in benchmark_dir/ take precedence if present.
    """

    def __init__(self, benchmark_dir: str = "evaluation/benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)

    def load_retrieval_benchmark(self) -> list[dict]:
        """Legal queries with known relevant judgments.
        Tests the hybrid GraphRAG retrieval pipeline independent of generation.
        Each entry: {query, relevant_citations}.
        relevant_citations=[] are precision tests — document is NOT in corpus."""
        path = self.benchmark_dir / "retrieval_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return self._generate_retrieval_benchmark()

    def load_compliance_benchmark(self) -> list[dict]:
        """Annotated Indian data policies with ground-truth DPDP 2023 gaps.
        Tests DPDPScanner recall and precision.
        Each entry: {id, name, clauses, known_gaps}."""
        path = self.benchmark_dir / "compliance_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return self._generate_compliance_benchmark()

    def load_qa_benchmark(self) -> list[dict]:
        """Legal Q&A pairs spanning all four LexPilot agent capabilities.
        Each entry: {question, ground_truth, category}.
        Categories: compliance | precedent | contract | risk"""
        path = self.benchmark_dir / "qa_benchmark.json"
        if path.exists():
            return json.loads(path.read_text())
        return self._generate_qa_benchmark()

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _generate_retrieval_benchmark(self) -> list[dict]:
        """Retrieval benchmark: queries that discriminate on specific case facts.

        Design principles:
        - Queries name unique parties, facts, and legal tests — not generic topics
        - Multiple relevant_citations per entry are reporter aliases for the same
          physical document (SCC + AIR, or normalised format variants)
        - relevant_citations=[] are precision tests: the queried document is NOT
          indexed; the system must NOT confidently surface a wrong result
        - Citation format variants are tested so normalisation is exercised
          (modern canonical, old IK format, no-parens, reporter-before-volume, AIR)

        All positive citations are verified against the ingested corpus.
        """
        return [
            # ── Puttaswamy — Right to privacy, 9-judge bench ──────────────
            {
                "query": "K.S. Puttaswamy retired judge nine bench unanimous privacy fundamental right overruling ADM Jabalpur informational autonomy Article 21",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },
            {
                "query": "Privacy proportionality test legality legitimate aim necessity balancing state surveillance Puttaswamy nine judge 2017",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },
            {
                "query": "Right to privacy intrinsic life liberty not penumbral Article 21 nine judges 2017 unanimous",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },

            # ── Kesavananda Bharati — Basic structure, 13-judge bench ──────
            # Corpus stores citation as "1973 4 SCC 225" (no parens); normaliser maps both
            {
                "query": "Kesavananda Bharati basic structure doctrine Parliament cannot destroy essential features Constitution amendment power limits Kerala land reform",
                "relevant_citations": ["(1973) 4 SCC 225"],
            },
            {
                "query": "Twenty-fourth amendment Golaknath overruled thirteen judges essential features unamendable basic structure 1973",
                "relevant_citations": ["(1973) 4 SCC 225"],
            },

            # ── Maneka Gandhi — Golden triangle Art 14/19/21 ──────────────
            # Stored as "1978 SCC (1) 248"; AIR alias also accepted
            {
                "query": "Maneka Gandhi passport impounded procedure established by law Article 21 just fair reasonable golden triangle",
                "relevant_citations": ["AIR 1978 SC 597", "1978 SCC (1) 248"],
            },
            {
                "query": "Personal liberty Articles 14 19 21 interconnected audi alteram partem natural justice passport Maneka 1978",
                "relevant_citations": ["AIR 1978 SC 597", "1978 SCC (1) 248"],
            },

            # ── Vishaka — Sexual harassment at workplace ───────────────────
            {
                "query": "Vishaka Bhanwari Devi gang rape Rajasthan sexual harassment workplace binding guidelines internal complaints committee CEDAW",
                "relevant_citations": ["(1997) 6 SCC 241"],
            },
            {
                "query": "Employer duty prevent redress sexual harassment absence legislation Vishaka writ guidelines 1997",
                "relevant_citations": ["(1997) 6 SCC 241"],
            },

            # ── MC Mehta — Absolute liability, Shriram oleum gas leak ──────
            # Stored as "(1987) 1 SCC 395"; AIR alias also accepted
            {
                "query": "MC Mehta Shriram Food Fertilisers oleum gas leak absolute liability hazardous enterprise no exceptions Rylands Fletcher distinguished",
                "relevant_citations": ["AIR 1987 SC 1086", "(1987) 1 SCC 395"],
            },
            {
                "query": "Absolute liability rule enterprise engaged hazardous activity owes no defence compensation enterprise capacity Bhagwati CJ 1987",
                "relevant_citations": ["AIR 1987 SC 1086", "(1987) 1 SCC 395"],
            },

            # ── Unni Krishnan — Right to education ────────────────────────
            {
                "query": "Unni Krishnan right to education fundamental right implied Article 21 free compulsory primary unaided private schools",
                "relevant_citations": ["(1993) 1 SCC 645"],
            },

            # ── Romesh Thappar — Freedom of press, pre-censorship ─────────
            # Stored as "1950 SCR 594"; AIR alias also accepted
            {
                "query": "Romesh Thappar Cross Roads magazine Madras government pre-censorship ban freedom of press Article 19 public order 1950",
                "relevant_citations": ["AIR 1950 SC 124", "1950 SCR 594"],
            },

            # ── Shayara Bano — Triple talaq unconstitutional ───────────────
            {
                "query": "Shayara Bano instant triple talaq talaq-e-biddat manifestly arbitrary unconstitutional Muslim personal law three pronouncements simultaneously",
                "relevant_citations": ["(2017) 9 SCC 1"],
            },
            {
                "query": "Triple talaq one sitting Article 14 equality manifestly arbitrary void unconstitutional 3:2 majority bench 2017",
                "relevant_citations": ["(2017) 9 SCC 1"],
            },

            # ── Vineeta Sharma — Hindu daughters as coparceners ───────────
            {
                "query": "Vineeta Sharma daughter coparcener Hindu Succession Amendment 2005 equal birth right HUF retrospective",
                "relevant_citations": ["(2020) 9 SCC 1"],
            },
            {
                "query": "Hindu daughter coparcenary right by birth Prakash v Phulavati overruled retrospective father alive 2005 amendment",
                "relevant_citations": ["(2020) 9 SCC 1"],
            },

            # ── Shreya Singhal — Section 66A IT Act struck down ───────────
            {
                "query": "Shreya Singhal Section 66A Information Technology Act struck down unconstitutional grossly offensive annoyance chilling effect free speech",
                "relevant_citations": ["(2015) 5 SCC 1"],
            },
            {
                "query": "Section 66A vagueness overbreadth internet speech Article 19 Shreya Singhal three judge bench 2015",
                "relevant_citations": ["(2015) 5 SCC 1"],
            },

            # ── Navtej Singh Johar — Section 377 decriminalised ───────────
            {
                "query": "Navtej Singh Johar Section 377 IPC consensual same-sex adults struck down constitutional morality dignity autonomy five judge bench",
                "relevant_citations": ["(2018) 10 SCC 1"],
            },
            {
                "query": "Section 377 partially struck down LGBT sexual orientation autonomy Suresh Kumar Koushal overruled constitutional morality Navtej 2018",
                "relevant_citations": ["(2018) 10 SCC 1"],
            },

            # ── Common Cause — Passive euthanasia, advance directive ───────
            {
                "query": "Common Cause advance directive living will passive euthanasia right to die dignity terminally ill withdrawal life support Article 21",
                "relevant_citations": ["(2018) 5 SCC 1"],
            },
            {
                "query": "Advance medical directive terminally ill five judge bench safeguards hospital procedure passive euthanasia 2018",
                "relevant_citations": ["(2018) 5 SCC 1"],
            },

            # ── Sabarimala review/reference (2019) — not original 5-judge ──
            # (2018) 8 SCC 1 (original bench) is NOT in corpus.
            # (2019) 11 SCC 1 (review/reference order) IS indexed.
            # Queries target the review content: 3:2 refers to 9-judge bench,
            # Articles 25/26 scope, essential religious practice, non-Hindu standing.
            {
                "query": "Sabarimala review petition 2019 refers nine judge bench essential religious practice Article 25 26 denomination exclusion reconsideration",
                "relevant_citations": ["(2019) 11 SCC 1"],
            },
            {
                "query": "Sabarimala review 3:2 majority reference larger bench Ayyappa denomination women exclusion non-Hindu standing 2019",
                "relevant_citations": ["(2019) 11 SCC 1"],
            },

            # ── D Velusamy — Live-in relationship, DV Act ─────────────────
            # Stored as "AIR 2011 SUPREME COURT 479"; SCC alias also accepted
            {
                "query": "Velusamy live-in relationship domestic violence Act wife eligibility relationship in the nature of marriage five conditions 2010",
                "relevant_citations": ["(2010) 10 SCC 469", "AIR 2011 SUPREME COURT 479"],
            },

            # ── Arnesh Kumar — Section 498A arrest safeguards ─────────────
            # Stored as "(2014) 8 SCC 273"
            {
                "query": "Arnesh Kumar Section 498A IPC arrest checklist nine point magistrate police guidelines automatic arrest dowry harassment 2014",
                "relevant_citations": ["(2014) 8 SCC 273"],
            },

            # ── Cadila Healthcare — Pharmaceutical trademark ───────────────
            {
                "query": "Cadila Healthcare Falcigo Falciwin pharmaceutical drug brand name deceptive similarity phonetic visual confusion consumer health risk trademark",
                "relevant_citations": ["(2001) 5 SCC 73"],
            },

            # ── Dasrath Rupsingh — Cheque bounce jurisdiction ─────────────
            {
                "query": "Dasrath Rupsingh cheque dishonour Section 138 NI Act territorial jurisdiction drawee bank branch place of presentation",
                "relevant_citations": ["(2014) 9 SCC 129"],
            },

            # ── Gurbaksh Singh Sibbia — Anticipatory bail ─────────────────
            {
                "query": "Gurbaksh Singh Sibbia anticipatory bail Section 438 CrPC discretion conditions liberal interpretation Punjab Haryana",
                "relevant_citations": ["(1980) 2 SCC 565"],
            },

            # ── Sarla Verma — Motor accident compensation ─────────────────
            {
                "query": "Sarla Verma motor accident compensation multiplier method future prospects dependency structured formula conventional heads",
                "relevant_citations": ["(2017) 16 SCC 680"],
            },
            {
                "query": "Fatal accident loss of dependency multiplier income future earnings death motor vehicle claim structured compensation",
                "relevant_citations": ["(2017) 16 SCC 680"],
            },

            # ── SP Gupta — PIL standing, judges transfer ──────────────────
            # Stored as "1981 SUPP (1) SCC 87"; SCC alias also accepted
            {
                "query": "SP Gupta judges transfer case public interest litigation epistolary jurisdiction locus standi third party bonafide petitioner",
                "relevant_citations": ["(1982) 1 SCC 618", "1981 SUPP (1) SCC 87"],
            },

            # ── ADM Jabalpur — OVERRULED, but in corpus ───────────────────
            # Stored as "1976 SCC (2) 521" (reporter-before-volume format)
            # Should surface with [OVERRULED] flag; both format variants accepted
            {
                "query": "ADM Jabalpur Shivkant Shukla habeas corpus Emergency suspension fundamental rights Article 21 1976 majority",
                "relevant_citations": ["1976 SCC (2) 521", "AIR 1976 SC 1207"],
            },

            # ── Precision tests — documents NOT in corpus ─────────────────
            # Bachan Singh (1980) 2 SCC 684: death penalty, 5-judge bench — NOT indexed.
            # Must NOT surface Gurbaksh Singh Sibbia or Mithu instead.
            {
                "query": "Bachan Singh five judge bench 1980 Section 302 IPC death penalty rarest of rare doctrine aggravating mitigating balance upheld constitutional validity",
                "relevant_citations": [],
            },
            {
                "query": "Bachan Singh death penalty Article 21 constitutional validity aggravating mitigating factors five judge 1980",
                "relevant_citations": [],
            },
            # Regulatory/statutory queries: corpus is judgments, not regulations
            {
                "query": "GST input tax credit ITC mismatch supplier default GSTR-3B Section 16 CGST recovery reversal",
                "relevant_citations": [],
            },
            {
                "query": "RERA Section 18 delayed possession homebuyer compensation builder penalty interest remedies",
                "relevant_citations": [],
            },
            {
                "query": "Cryptocurrency virtual digital asset FEMA PMLA seizure tracing blockchain regulation India enforcement",
                "relevant_citations": [],
            },

            # ── Extended queries: Puttaswamy — additional angles ──────────
            {
                "query": "Aadhaar biometric identity scheme privacy nine judge bench proportionality test 2017",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },
            {
                "query": "Informational self-determination surveillance data protection privacy jurisprudence India 2017 nine judges",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },

            # ── Extended: Kesavananda Bharati ─────────────────────────────
            {
                "query": "Article 368 amending power limitation constituent power basic structure thirteen judge 1973 Kerala",
                "relevant_citations": ["(1973) 4 SCC 225"],
            },
            {
                "query": "Fundamental rights property amendment Golaknath overruled parliamentary sovereignty basic structure 1973",
                "relevant_citations": ["(1973) 4 SCC 225"],
            },

            # ── Extended: Maneka Gandhi ───────────────────────────────────
            {
                "query": "Section 10(3)(c) Passports Act void audi alteram partem impound natural justice 1978 passport",
                "relevant_citations": ["AIR 1978 SC 597", "1978 SCC (1) 248"],
            },
            {
                "query": "Reasonableness procedure Article 21 just fair not arbitrary personal liberty Bhagwati Krishna Iyer 1978",
                "relevant_citations": ["AIR 1978 SC 597", "1978 SCC (1) 248"],
            },

            # ── Extended: Vishaka ─────────────────────────────────────────
            {
                "query": "Sexual harassment guidelines binding ICC Internal Complaints Committee employer liability 1997 Rajasthan",
                "relevant_citations": ["(1997) 6 SCC 241"],
            },
            {
                "query": "CEDAW gender equality duty employer workplace women safety Bhanwari Devi 1997 Supreme Court writ",
                "relevant_citations": ["(1997) 6 SCC 241"],
            },

            # ── Extended: MC Mehta ────────────────────────────────────────
            {
                "query": "Oleum gas leak absolute liability exceptions act of God third party not applicable enterprise hazardous 1987",
                "relevant_citations": ["AIR 1987 SC 1086", "(1987) 1 SCC 395"],
            },
            {
                "query": "Rylands Fletcher distinguished absolute liability evolve new principle enterprise polluter pays India 1987",
                "relevant_citations": ["AIR 1987 SC 1086", "(1987) 1 SCC 395"],
            },

            # ── Extended: Unni Krishnan ───────────────────────────────────
            {
                "query": "Right to education Article 21 free compulsory 14 years unaided private schools 1993 Andhra Pradesh",
                "relevant_citations": ["(1993) 1 SCC 645"],
            },
            {
                "query": "Education fundamental right implied Article 21 Unni Krishnan fee schedule private schools 1993",
                "relevant_citations": ["(1993) 1 SCC 645"],
            },

            # ── Extended: Romesh Thappar ──────────────────────────────────
            {
                "query": "Section 9(1-A) Madras Maintenance Public Order Act unconstitutional press ban cross roads magazine 1950",
                "relevant_citations": ["AIR 1950 SC 124", "1950 SCR 594"],
            },
            {
                "query": "Pre-censorship ban distribution freedom speech press restriction public order 1950 Supreme Court",
                "relevant_citations": ["AIR 1950 SC 124", "1950 SCR 594"],
            },

            # ── Extended: Shayara Bano ────────────────────────────────────
            {
                "query": "Muslim Personal Law Shariat Application Act triple talaq validity arbitrariness five judge bench",
                "relevant_citations": ["(2017) 9 SCC 1"],
            },
            {
                "query": "Triple talaq Article 25 religious practice manifestly arbitrary 3:2 void Shayara Bano 2017",
                "relevant_citations": ["(2017) 9 SCC 1"],
            },

            # ── Extended: Vineeta Sharma ──────────────────────────────────
            {
                "query": "Daughter coparcener birth right HUF Hindu Undivided Family father alive 2005 amendment retrospective 2020",
                "relevant_citations": ["(2020) 9 SCC 1"],
            },
            {
                "query": "Prakash Phulavati overruled Danamma Vineeta Sharma coparcenary daughter right by birth 2020",
                "relevant_citations": ["(2020) 9 SCC 1"],
            },

            # ── Extended: Shreya Singhal ──────────────────────────────────
            {
                "query": "Information Technology Act Section 66A vague overbroad constitutional challenge internet speech 2015",
                "relevant_citations": ["(2015) 5 SCC 1"],
            },
            {
                "query": "Online speech restriction annoyance menace grossly offensive Section 66A struck down free speech 2015",
                "relevant_citations": ["(2015) 5 SCC 1"],
            },

            # ── Extended: Navtej Singh Johar ──────────────────────────────
            {
                "query": "Section 377 IPC partially struck down consensual homosexuality dignity equal protection 2018 five bench",
                "relevant_citations": ["(2018) 10 SCC 1"],
            },
            {
                "query": "Constitutional morality Suresh Koushal overruled privacy dignity LGBTQ Section 377 Navtej 2018",
                "relevant_citations": ["(2018) 10 SCC 1"],
            },

            # ── Extended: Common Cause ────────────────────────────────────
            {
                "query": "Right to die with dignity passive euthanasia advance directive living will valid 2018 five judge",
                "relevant_citations": ["(2018) 5 SCC 1"],
            },
            {
                "query": "Aruna Shanbaug passive euthanasia overruled advance medical directive terminally ill safeguards 2018",
                "relevant_citations": ["(2018) 5 SCC 1"],
            },

            # ── Extended: Sabarimala (2019 review/reference) ─────────────
            {
                "query": "Sabarimala review nine judge bench essential religious practice Article 25 denomination women 2019",
                "relevant_citations": ["(2019) 11 SCC 1"],
            },
            {
                "query": "Kerala temple women exclusion Article 26 review reference larger bench Indu Malhotra 2019",
                "relevant_citations": ["(2019) 11 SCC 1"],
            },

            # ── Extended: D Velusamy ──────────────────────────────────────
            {
                "query": "Domestic Violence Act wife live-in relationship nature of marriage five conditions cohabitation 2010",
                "relevant_citations": ["(2010) 10 SCC 469", "AIR 2011 SUPREME COURT 479"],
            },
            {
                "query": "Partner domestic violence shared household live-in eligibility five conditions Supreme Court Velusamy 2010",
                "relevant_citations": ["(2010) 10 SCC 469", "AIR 2011 SUPREME COURT 479"],
            },

            # ── Extended: Arnesh Kumar ────────────────────────────────────
            {
                "query": "498A IPC automatic arrest nine-point checklist police Magistrate need checklist 2014 Supreme Court",
                "relevant_citations": ["(2014) 8 SCC 273"],
            },
            {
                "query": "498A dowry cruelty arrest guidelines cognizable non-bailable no arrest without satisfaction Arnesh 2014",
                "relevant_citations": ["(2014) 8 SCC 273"],
            },

            # ── Extended: Cadila Healthcare ───────────────────────────────
            {
                "query": "Pharmaceutical drug trademark deceptive similar phonetic visual trade dress consumer confusion Cadila 2001",
                "relevant_citations": ["(2001) 5 SCC 73"],
            },
            {
                "query": "Falcigo Falciwin drug name confusion health risk consumers phonetic similarity trademark India 2001",
                "relevant_citations": ["(2001) 5 SCC 73"],
            },

            # ── Extended: Dasrath Rupsingh ────────────────────────────────
            {
                "query": "Section 138 NI Act cheque dishonour territorial jurisdiction drawee bank branch presentation place 2014",
                "relevant_citations": ["(2014) 9 SCC 129"],
            },
            {
                "query": "Negotiable Instruments Act cheque bounce where filed jurisdiction drawee bank Dasrath Rupsingh 2014",
                "relevant_citations": ["(2014) 9 SCC 129"],
            },

            # ── Extended: Gurbaksh Singh Sibbia ──────────────────────────
            {
                "query": "Section 438 CrPC scope conditions time limit blanket bail directions anticipatory bail 1980 Punjab",
                "relevant_citations": ["(1980) 2 SCC 565"],
            },
            {
                "query": "Anticipatory bail liberal interpretation discretion conditions five judges Punjab Haryana Gurbaksh 1980",
                "relevant_citations": ["(1980) 2 SCC 565"],
            },

            # ── Extended: Sarla Verma ─────────────────────────────────────
            {
                "query": "Motor accident compensation structured assessment multiplier income dependency MACT formula 2017",
                "relevant_citations": ["(2017) 16 SCC 680"],
            },
            {
                "query": "Fatal accident loss of dependency conventional heads future prospects multiplier Motor Vehicles Act",
                "relevant_citations": ["(2017) 16 SCC 680"],
            },

            # ── Extended: SP Gupta ────────────────────────────────────────
            {
                "query": "PIL standing epistolary jurisdiction letter writ third party bonafide 1981 judges transfer",
                "relevant_citations": ["(1982) 1 SCC 618", "1981 SUPP (1) SCC 87"],
            },
            {
                "query": "Judges appointment executive immunity PIL locus standi public interest Bhagwati 1982 Supreme Court",
                "relevant_citations": ["(1982) 1 SCC 618", "1981 SUPP (1) SCC 87"],
            },

            # ── Extended: ADM Jabalpur ────────────────────────────────────
            {
                "query": "ADM Jabalpur majority opinion Ray CJ habeas corpus MISA Emergency 1976 four majority one dissent",
                "relevant_citations": ["1976 SCC (2) 521", "AIR 1976 SC 1207"],
            },
            {
                "query": "Shivkant Shukla Emergency 1976 Article 21 suspended fundamental rights National Emergency ADM Jabalpur overruled Puttaswamy",
                "relevant_citations": ["1976 SCC (2) 521", "AIR 1976 SC 1207"],
            },

            # ── Cross-case disambiguation queries ─────────────────────────
            {
                "query": "Which Supreme Court bench overruled ADM Jabalpur emergency ruling and declared privacy fundamental right",
                "relevant_citations": ["(2017) 10 SCC 1"],
            },
            {
                "query": "Constitutional validity passport impound procedure natural justice personal liberty golden triangle 1978",
                "relevant_citations": ["AIR 1978 SC 597", "1978 SCC (1) 248"],
            },
            {
                "query": "Section 438 CrPC anticipatory bail constitution bench five judge Punjab 1980 conditions scope",
                "relevant_citations": ["(1980) 2 SCC 565"],
            },
            {
                "query": "MISA habeas corpus maintainability Emergency national emergency 1976 majority Article 21 suspended ruling",
                "relevant_citations": ["1976 SCC (2) 521", "AIR 1976 SC 1207"],
            },

            # ── Additional precision tests — documents NOT in corpus ───────
            # Sabarimala original 5-judge bench NOT indexed (only review/reference is)
            {
                "query": "Indian Young Lawyers Association Sabarimala 2018 original bench women entry temple prohibition age 10 50 Indu Malhotra dissent",
                "relevant_citations": [],
            },
            # Puttaswamy-2 Aadhaar 2018 is a different judgment from the privacy ruling
            {
                "query": "Puttaswamy Aadhaar 2018 constitutional validity biometric identity scheme nine bench final Aadhaar Act",
                "relevant_citations": [],
            },
            # NALSA 2014 — third gender recognition — not in corpus
            {
                "query": "NALSA National Legal Services Authority third gender transgender hijra recognition fundamental rights Article 21 2014",
                "relevant_citations": [],
            },
            # Suresh Kumar Koushal 2013 — reversed by Navtej — not indexed
            {
                "query": "Suresh Kumar Koushal Section 377 homosexuality reversed criminalisation 2013 Delhi High Court overruled",
                "relevant_citations": [],
            },
            # Electoral Bonds 2024 — not in corpus
            {
                "query": "Electoral bonds scheme Association for Democratic Reforms ECI disclosure campaign finance unconstitutional 2024",
                "relevant_citations": [],
            },
            # Competition Act cartelisation — statutory, not a judgment in corpus
            {
                "query": "Competition Act 2002 dominance abuse market cartelization CCI penalty horizontal agreement jurisprudence",
                "relevant_citations": [],
            },
        ]

    # ── Compliance ────────────────────────────────────────────────────────────

    def _generate_compliance_benchmark(self) -> list[dict]:
        """DPDP Act 2023 compliance benchmark.

        Six annotated Indian data policies across industries.
        known_gaps reference only keys present in DPDP_SECTIONS:
          Section 4  Consent
          Section 5  Notice
          Section 6  Legitimate Uses (no-consent processing)
          Section 7  General Obligations (security, data quality)
          Section 8  Significant Data Fiduciary obligations
          Section 9  Rights of Data Principal
          Section 10 Right to Correction and Erasure
          Section 11 Right to Grievance Redressal
          Section 14 Duties of Data Principal
          Section 16 Cross-Border Transfer
          Section 17 Exemptions
          Section 18 Data Protection Board
        gap_type: missing | insufficient | conflicting
        """
        return [
            {
                "id": "policy_001",
                "name": "E-commerce Platform — Bundled Consent",
                "clauses": [
                    "We collect your name, email address, mobile number, delivery address, and purchase history to provide our services.",
                    "By using our platform, you agree to our data collection and processing practices.",
                    "We share your data with third-party marketing and analytics partners to improve your experience.",
                    "Your data is stored on servers in the United States and Singapore.",
                    "For data-related queries, contact support@example.com.",
                    "We use tracking cookies across our platform and partner websites.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "insufficient",
                        "description": "Bundled consent via 'by using our platform' — no granular, informed opt-in per purpose",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "missing",
                        "description": "No DPDP-compliant notice specifying data categories, processing purposes, or rights before collection",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "missing",
                        "description": "No mention of right to access, correction, erasure, grievance redressal, or nomination",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No mechanism for data principal to request correction or erasure of personal data",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "Support email provided but no designated grievance officer, response timeline, or escalation path",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "Transfer to US and Singapore without Central Government restriction check or notification compliance",
                    },
                ],
            },
            {
                "id": "policy_002",
                "name": "SaaS DPA — Significant Data Fiduciary, No DPO",
                "clauses": [
                    "The Processor shall process personal data solely on documented instructions from the Controller.",
                    "All personal data is encrypted at rest (AES-256) and in transit (TLS 1.3).",
                    "Data subjects may request access to their data by contacting the Controller.",
                    "The Processor shall notify the Controller of any personal data breach within 72 hours.",
                    "The Processor processes personal data of more than 10 million Data Principals annually.",
                    "Sub-processors may be engaged with prior written Controller consent.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 8",
                        "gap_type": "missing",
                        "description": "No DPO appointment despite processing data of >10 million Data Principals — qualifies as Significant Data Fiduciary",
                    },
                    {
                        "dpdp_section": "Section 8",
                        "gap_type": "missing",
                        "description": "No periodic data protection impact assessment or independent data audit mentioned",
                    },
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "insufficient",
                        "description": "No direct consent mechanism from Data Principals — only controller-to-processor contractual instruction",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "missing",
                        "description": "No grievance redressal mechanism for Data Principals to raise complaints directly with the Processor",
                    },
                    {
                        "dpdp_section": "Section 7",
                        "gap_type": "insufficient",
                        "description": "Encryption mentioned but no reference to data completeness, accuracy, or consistency obligations",
                    },
                ],
            },
            {
                "id": "policy_003",
                "name": "Healthcare App — Mostly Compliant, Minor Gaps",
                "clauses": [
                    "We collect health data including vitals, prescriptions, and diagnostic reports with your explicit written consent.",
                    "You may withdraw consent at any time through App Settings > Privacy. Processing stops within 7 days of withdrawal.",
                    "Your data is used for: (a) providing medical services, (b) appointment reminders, (c) anonymised population health research.",
                    "We retain your health data for 8 years as required by the Clinical Establishments Act.",
                    "Our Data Protection Officer, Dr. Ravi Kumar, can be reached at dpo@healthapp.in or +91-80-12345678.",
                    "You have the right to access, correct, and request deletion of your personal health data through the app.",
                    "We obtain separate explicit consent from parents or legal guardians for patients under 18.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice does not enumerate all data categories collected (e.g., location, device identifiers, usage logs)",
                    },
                    {
                        "dpdp_section": "Section 6",
                        "gap_type": "insufficient",
                        "description": "Research purpose (c) not clearly tied to anonymisation standards or a valid Section 6 legitimate-use exemption",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "No cross-border transfer policy — unclear whether health data is processed by foreign cloud providers",
                    },
                    {
                        "dpdp_section": "Section 14",
                        "gap_type": "missing",
                        "description": "No disclosure of Data Principal duties not to register false grievances or furnish inaccurate information",
                    },
                ],
            },
            {
                "id": "policy_004",
                "name": "Fintech KYC Policy — Minimal, High Risk",
                "clauses": [
                    "We collect Aadhaar number, PAN, bank account details, and a live selfie for KYC verification.",
                    "Your data is shared with our KYC verification partner and the relevant regulated bank.",
                    "We retain data until account closure plus 5 years as required by RBI and PMLA guidelines.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "missing",
                        "description": "No consent mechanism — no indication of how Data Principal consents to Aadhaar, PAN, and biometric collection",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "missing",
                        "description": "No formal notice specifying purposes, data fiduciary identity, or Data Principal rights before collection",
                    },
                    {
                        "dpdp_section": "Section 7",
                        "gap_type": "insufficient",
                        "description": "No security safeguards described for highly sensitive biometric and financial data",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "missing",
                        "description": "No Data Principal rights mentioned — access, correction, erasure, grievance redressal absent",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No correction or erasure mechanism described for KYC data",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "missing",
                        "description": "No grievance officer, contact details, or response timeline provided",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "KYC partner jurisdiction unknown — potential cross-border transfer without restriction compliance",
                    },
                ],
            },
            {
                "id": "policy_005",
                "name": "EdTech Student Data Policy — Consent and Erasure Gaps",
                "clauses": [
                    "We collect student name, age, grades, learning progress, and parent contact details.",
                    "Parents and guardians provide consent on behalf of minor students during school registration.",
                    "Data is used for personalised learning recommendations, progress reports, and curriculum improvement.",
                    "Aggregated and anonymised statistics are shared with school administrators and education boards.",
                    "Students and parents can request a copy of their data in machine-readable format by emailing privacy@edtech.in.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice does not describe all purposes, data categories, or DPDP-required rights disclosures",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "insufficient",
                        "description": "Only data export mentioned — right to correction, erasure, and grievance redressal absent",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No correction or erasure process described for inaccurate or outdated student records",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "missing",
                        "description": "No designated grievance officer, prescribed response timeline, or escalation mechanism",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "No cross-border transfer policy despite likely use of foreign SaaS infrastructure",
                    },
                ],
            },
            {
                "id": "policy_006",
                "name": "Corporate HR Employee Data Policy — Employment Context",
                "clauses": [
                    "We process employee personal data including name, contact details, bank account, Aadhaar, PAN, salary history, and performance reviews.",
                    "Data processing is necessary for employment contract performance and statutory compliance (PF, ESIC, TDS, payroll).",
                    "Employee data may be transferred to our parent company in Germany and our payroll processor in the Philippines.",
                    "Employees may access their personal data through the HRMS self-service portal.",
                    "Data is retained for 7 years post-employment for legal and statutory purposes.",
                    "The HR department handles all employee data-related complaints.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice lacks itemised description of all data categories and explicit statement of DPDP rights",
                    },
                    {
                        "dpdp_section": "Section 8",
                        "gap_type": "missing",
                        "description": "No assessment of whether organisation qualifies as Significant Data Fiduciary requiring DPO appointment",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "insufficient",
                        "description": "HRMS portal provides access but no explicit correction or erasure mechanism described",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "HR department named but no designated grievance officer, timeline, Board escalation path, or response SLA",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "Transfer to Germany and Philippines without restriction check or Central Government notification compliance",
                    },
                ],
            },
            {
                "id": "policy_007",
                "name": "Telecom / ISP — CDR Retention and Traffic Analysis",
                "clauses": [
                    "We retain call detail records (CDR), SMS metadata, and internet traffic logs for 2 years in accordance with DoT licensing conditions.",
                    "Traffic data may be analysed by our AI systems to detect fraud and personalise service plans.",
                    "Data is shared with law enforcement agencies upon receipt of a valid legal order.",
                    "Aggregated anonymised usage statistics may be sold to third-party market research firms.",
                    "Customers can raise billing-related complaints through our customer care helpline.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "missing",
                        "description": "No consent for AI-based traffic analysis or sale of anonymised statistics to third-party market research firms",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "missing",
                        "description": "No DPDP-compliant notice specifying purposes, AI profiling, third-party sale, or data principal rights",
                    },
                    {
                        "dpdp_section": "Section 7",
                        "gap_type": "insufficient",
                        "description": "CDR and traffic data are sensitive; no security safeguards, data quality, or accuracy obligations described",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "missing",
                        "description": "No data principal rights — access, correction, erasure, grievance redressal, nomination all absent",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "Only billing complaints addressed; no DPDP grievance officer, response timeline, or Board escalation for data complaints",
                    },
                    {
                        "dpdp_section": "Section 17",
                        "gap_type": "insufficient",
                        "description": "DoT licensing relied upon for retention but DPDP exemption applicability not assessed",
                    },
                ],
            },
            {
                "id": "policy_008",
                "name": "Health Insurance — Medical Underwriting Data Policy",
                "clauses": [
                    "We collect medical history, diagnostic reports, prescriptions, and lifestyle questionnaire responses for underwriting.",
                    "Your health data is shared with our panel hospitals, third-party administrators (TPAs), and reinsurers.",
                    "We may use your claims history and medical data to adjust your renewal premium at our discretion.",
                    "You consent to our collection and use by submitting the proposal form.",
                    "Data is retained for the duration of your policy plus 7 years post-expiry.",
                    "Disputes may be raised with our grievance team at grievance@insurer.in within 15 days.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "insufficient",
                        "description": "Bundled consent via proposal form — no granular opt-in for TPA sharing, reinsurer transfer, or premium adjustment profiling",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice does not enumerate all data categories, purposes, or rights at proposal stage",
                    },
                    {
                        "dpdp_section": "Section 7",
                        "gap_type": "insufficient",
                        "description": "Sensitive health data shared with multiple entities; no security safeguards or accuracy obligations stated",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No mechanism to correct inaccurate medical records or request erasure post-policy expiry",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "15-day dispute window but no DPDP grievance officer, SLA, or Board escalation path",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "Reinsurer transfer may be cross-border; no restriction check or Central Government notification compliance described",
                    },
                ],
            },
            {
                "id": "policy_009",
                "name": "Banking Mobile App — Digital Payments Privacy Policy",
                "clauses": [
                    "Our mobile banking app collects device identifiers, location data, transaction history, and behavioural patterns.",
                    "Transaction data is used to detect fraud, offer personalised financial products, and comply with RBI and FEMA regulations.",
                    "We share customer data with credit bureaus (CIBIL, Equifax), payment networks (NPCI, Visa), and marketing partners.",
                    "Customers can view and download their transaction statements from the app.",
                    "Data is processed in India and on cloud servers hosted by our technology partner.",
                    "For privacy-related complaints, contact our nodal officer at nodal@bank.in.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "insufficient",
                        "description": "No granular consent for marketing partner sharing or behavioural profiling; bundled with general app usage",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice does not specify location, device fingerprint, behavioural data categories, all purposes, or DPDP rights",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "insufficient",
                        "description": "Only statement download; correction, erasure of behavioural profiles, and nomination not addressed",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No mechanism to correct transaction categorisation errors or request deletion of behavioural analytics",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "Nodal officer email but no DPDP grievance officer, response timeline, or Board escalation",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "Cloud servers via technology partner not confirmed as India-based; potential cross-border transfer without restriction check",
                    },
                ],
            },
            {
                "id": "policy_010",
                "name": "Government e-Governance Portal — Citizen Data Policy",
                "clauses": [
                    "This portal collects Aadhaar number, PAN, address proof, income certificate, and caste certificate for processing government scheme applications.",
                    "Data is used to verify eligibility and disburse benefits under relevant Central and State government schemes.",
                    "Information may be shared with other government departments and local bodies for scheme administration.",
                    "Citizens may download their application status from the portal.",
                    "Data is retained in perpetuity for audit and government record purposes.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "missing",
                        "description": "No consent mechanism; portal assumes collection is mandated without informing citizens of processing basis",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "missing",
                        "description": "No DPDP-compliant notice on data categories, purposes, inter-departmental sharing, or rights",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "missing",
                        "description": "No citizen rights — access, correction, erasure, grievance redressal, nomination all absent",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No mechanism to correct errors in eligibility data such as income or caste certificates",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "missing",
                        "description": "No grievance officer, contact details, response timeline, or Board escalation",
                    },
                    {
                        "dpdp_section": "Section 17",
                        "gap_type": "insufficient",
                        "description": "Policy does not assess which DPDP exemptions apply to government processing or specify applicable provisions",
                    },
                ],
            },
            {
                "id": "policy_011",
                "name": "Logistics / Last-Mile Delivery — Delivery Partner and Customer Data Policy",
                "clauses": [
                    "We collect recipient name, delivery address, mobile number, and digital signature for operational purposes.",
                    "Delivery partner location is tracked in real time during active delivery assignments using GPS.",
                    "Customer delivery data is shared with merchant partners and analytics vendors to improve delivery SLAs.",
                    "Delivery partners' location history and performance data are retained for 3 years.",
                    "Recipients can reschedule deliveries by contacting customer support.",
                    "All data is stored on servers in India operated by our cloud infrastructure partner.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "missing",
                        "description": "No consent from recipients or delivery partners for real-time location tracking, analytics sharing, or extended retention",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "missing",
                        "description": "No DPDP notice to recipients or delivery partners on purposes, data categories, or rights",
                    },
                    {
                        "dpdp_section": "Section 7",
                        "gap_type": "insufficient",
                        "description": "Real-time location data is sensitive; no security safeguards, access controls, or accuracy obligations described",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "missing",
                        "description": "No rights mechanism for recipients or partners — access, correction, erasure, nomination absent",
                    },
                    {
                        "dpdp_section": "Section 10",
                        "gap_type": "missing",
                        "description": "No correction or deletion process for inaccurate delivery records or partner performance data",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "insufficient",
                        "description": "Operational support only; no DPDP grievance officer, privacy contact, or Board escalation",
                    },
                ],
            },
            {
                "id": "policy_012",
                "name": "Social Media Platform — User Generated Content and Profiling",
                "clauses": [
                    "We collect your profile data, posts, likes, shares, comments, messages, and inferred interests to provide our platform.",
                    "Your data is used to show personalised advertisements and content using inferred demographics and behaviour.",
                    "We share your data with advertisers, content measurement partners, and third-party plug-in providers.",
                    "You can download a copy of your data from Settings > Data Export.",
                    "Your data may be transferred to and processed on servers outside India where we or our partners operate.",
                    "Contact our support team for any account or privacy issue.",
                ],
                "known_gaps": [
                    {
                        "dpdp_section": "Section 4",
                        "gap_type": "insufficient",
                        "description": "Bundled consent via ToS — no granular opt-in for ad profiling, third-party sharing, or behavioural inference",
                    },
                    {
                        "dpdp_section": "Section 5",
                        "gap_type": "insufficient",
                        "description": "Notice buried in lengthy terms; inferred data, all purposes, DPDP rights, and grievance mechanism not clearly presented before consent",
                    },
                    {
                        "dpdp_section": "Section 6",
                        "gap_type": "missing",
                        "description": "No assessment of whether any processing qualifies as Section 6 legitimate use in lieu of consent",
                    },
                    {
                        "dpdp_section": "Section 9",
                        "gap_type": "insufficient",
                        "description": "Data export provided but correction of inferred profiles, erasure of ad data, and nomination rights not addressed",
                    },
                    {
                        "dpdp_section": "Section 11",
                        "gap_type": "missing",
                        "description": "No designated DPDP grievance officer, response timeline, or Board escalation — only generic support contact",
                    },
                    {
                        "dpdp_section": "Section 16",
                        "gap_type": "missing",
                        "description": "Cross-border transfer explicitly mentioned but no restriction check, Central Government notification compliance, or restricted-country screening",
                    },
                ],
            },
        ]

    # ── QA ────────────────────────────────────────────────────────────────────

    def _generate_qa_benchmark(self) -> list[dict]:
        """20 legal Q&A pairs spanning all four LexPilot agent capabilities.

        Covers:
          compliance  — DPDP Act 2023 provisions (DPDP Compliance Auditor)
          precedent   — Indian case law retrieval (Precedent Researcher)
          contract    — clause analysis and risk identification (Contract Analyst)
          risk        — legal risk assessment (Legal Risk Scorer)
        """
        return [
            # ── DPDP Compliance ───────────────────────────────────────────
            {
                "question": "What are the rights of a Data Principal under the DPDP Act 2023?",
                "ground_truth": (
                    "Under Section 9, Data Principals have the right to obtain information about "
                    "personal data being processed, right to correction of inaccurate data, right "
                    "to erasure, right to grievance redressal, and right of nomination. Section 10 "
                    "specifically governs correction and erasure, requiring the data fiduciary to "
                    "act on a valid request promptly."
                ),
                "category": "compliance",
            },
            {
                "question": "When can personal data be processed without the Data Principal's consent under DPDP 2023?",
                "ground_truth": (
                    "Section 6 (Legitimate Uses) permits processing without consent for: voluntary "
                    "provision of data where no benefit is sought in return; state and public functions; "
                    "compliance with a court order or law; medical or public health emergencies; and "
                    "employment-related processing. Each use must still be for a lawful purpose under "
                    "Section 4."
                ),
                "category": "compliance",
            },
            {
                "question": "What additional obligations apply to a Significant Data Fiduciary under DPDP?",
                "ground_truth": (
                    "Under Section 8, a Significant Data Fiduciary must: appoint a Data Protection "
                    "Officer based in India; conduct periodic data protection impact assessments; "
                    "undertake independent data audits; and comply with any further conditions "
                    "prescribed by the Central Government. The Central Government designates "
                    "Significant Data Fiduciaries based on volume of data processed, sensitivity, "
                    "and potential risk to Data Principals."
                ),
                "category": "compliance",
            },
            {
                "question": "What must a DPDP-compliant consent notice contain?",
                "ground_truth": (
                    "Under Section 5, a consent notice must: describe the personal data to be "
                    "processed; state the purpose of processing; explain how the Data Principal "
                    "can exercise their rights; and explain how they can raise a grievance. The "
                    "notice must be in clear, plain language and available in all Eighth Schedule "
                    "languages on request. It must be provided before or at the time of consent."
                ),
                "category": "compliance",
            },
            {
                "question": "What are the DPDP Act's rules on transferring personal data outside India?",
                "ground_truth": (
                    "Under Section 16, the Central Government may restrict transfer of personal "
                    "data to specified countries or territories by notification. Transfer is "
                    "generally permitted unless the destination country is restricted. Data "
                    "Fiduciaries must monitor applicable restriction orders and ensure no transfer "
                    "to blacklisted destinations."
                ),
                "category": "compliance",
            },
            {
                "question": "What penalties does the DPDP Act 2023 prescribe for a personal data breach?",
                "ground_truth": (
                    "The Act prescribes penalties up to Rs 250 crore for failure to implement "
                    "reasonable security safeguards leading to a personal data breach (Schedule 1). "
                    "Penalties are imposed by the Data Protection Board of India after inquiry. "
                    "For breach of child data processing obligations the penalty can reach Rs 200 crore."
                ),
                "category": "compliance",
            },
            # ── Precedent / Case Law ─────────────────────────────────────
            {
                "question": "What is the basic structure doctrine in Indian constitutional law and which case established it?",
                "ground_truth": (
                    "Established in Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225 by a "
                    "13-judge bench, the doctrine holds that Parliament's amending power under "
                    "Article 368 cannot be used to destroy or abrogate the Constitution's basic "
                    "structure. Essential features — such as judicial review, separation of powers, "
                    "supremacy of the Constitution, and fundamental rights — are immune from amendment."
                ),
                "category": "precedent",
            },
            {
                "question": "When did Indian courts recognise the right to privacy as a fundamental right?",
                "ground_truth": (
                    "In K.S. Puttaswamy v. Union of India (2017) 10 SCC 1, a 9-judge bench "
                    "unanimously declared privacy a fundamental right under Article 21. The court "
                    "overruled M.P. Sharma and Kharak Singh and held that privacy is intrinsic to "
                    "life and liberty. Any state intrusion must satisfy the proportionality test: "
                    "legality, legitimate aim, necessity, and proportionate balancing."
                ),
                "category": "precedent",
            },
            {
                "question": "What rule governs an employer's liability when a hazardous industry causes harm in India?",
                "ground_truth": (
                    "In M.C. Mehta v. Union of India (AIR 1987 SC 1086), Justice Bhagwati formulated "
                    "the absolute liability rule: an enterprise engaged in a hazardous or inherently "
                    "dangerous activity that escapes and causes harm is absolutely liable, with no "
                    "exceptions. This goes beyond strict liability (Rylands v. Fletcher) which allows "
                    "act-of-God and third-party defences. Compensation must match the enterprise's capacity."
                ),
                "category": "precedent",
            },
            {
                "question": "What criteria must a court consider when deciding an anticipatory bail application?",
                "ground_truth": (
                    "Per Gurbaksh Singh Sibbia v. State of Punjab (1980) 2 SCC 565, the court must "
                    "consider: the nature and gravity of the accusation; the applicant's antecedents "
                    "and criminal history; the possibility of the applicant fleeing justice; and "
                    "whether the accusation appears motivated by malice. The provision must be "
                    "interpreted liberally — conditions should not be so stringent as to render "
                    "anticipatory bail illusory."
                ),
                "category": "precedent",
            },
            {
                "question": "When did India's Supreme Court uphold the right to die with dignity and what safeguards were laid down?",
                "ground_truth": (
                    "In Common Cause v. Union of India (2018) 5 SCC 1, the 5-judge bench held that "
                    "the right to die with dignity is part of the right to life under Article 21. "
                    "The court upheld advance medical directives (living wills) and passive euthanasia "
                    "for terminally ill patients. Detailed safeguards were prescribed: a medical board "
                    "must certify terminal illness, a guardian or family member must be consulted, "
                    "and a Judicial Magistrate must be informed before life support is withdrawn."
                ),
                "category": "precedent",
            },
            {
                "question": "How did the Supreme Court's Arnesh Kumar guidelines change Section 498A IPC arrest practices?",
                "ground_truth": (
                    "In Arnesh Kumar v. State of Bihar (2014) 8 SCC 273, the court found that "
                    "Section 498A IPC was being misused for automatic arrests without application "
                    "of mind. The bench issued a 9-point checklist requiring police to satisfy "
                    "themselves that arrest is necessary before effecting one, and Magistrates to "
                    "independently apply mind before authorising detention. Summary trial "
                    "imprisonment without the checklist is a violation."
                ),
                "category": "precedent",
            },
            # ── Contract Analysis ─────────────────────────────────────────
            {
                "question": "What are the key data protection clauses to review in an Indian SaaS agreement?",
                "ground_truth": (
                    "Key clauses: (1) Processing scope and instructions — processor must act only on "
                    "controller instructions; (2) Sub-processor authorisation and audit rights; "
                    "(3) Security safeguards and breach notification timelines (72 hours to controller); "
                    "(4) Cross-border transfer restrictions under DPDP Section 16; "
                    "(5) DPO contact and direct grievance redressal for Data Principals under Section 11; "
                    "(6) Data retention and certified deletion on termination; "
                    "(7) DPDP Section 8 compliance if vendor qualifies as Significant Data Fiduciary."
                ),
                "category": "contract",
            },
            {
                "question": "What makes an indemnity clause one-sided in an Indian commercial contract?",
                "ground_truth": (
                    "A clause is one-sided when: indemnities flow only one way (vendor to client) "
                    "without reciprocal protection; it imposes unlimited liability on one party while "
                    "capping the other; it requires indemnity for matters outside the party's control "
                    "(e.g., third-party IP claims, regulatory changes); or it has broad hold-harmless "
                    "language with no carve-outs for fraud or gross negligence. Under the Indian "
                    "Contract Act 1872, courts may apply unconscionability doctrine but generally "
                    "enforce negotiated commercial terms between sophisticated parties."
                ),
                "category": "contract",
            },
            {
                "question": "What dispute resolution mechanisms are common in Indian B2B commercial contracts?",
                "ground_truth": (
                    "Common mechanisms: (1) Tiered escalation — senior management negotiation "
                    "(15-30 days), then mediation, then arbitration; (2) Arbitration under the "
                    "Arbitration and Conciliation Act 1996 (as amended 2015/2019) with DIAC, MCIA, "
                    "or ad hoc; (3) International arbitration with SIAC, ICC, or LCIA seat for "
                    "cross-border contracts; (4) Exclusive High Court jurisdiction for smaller "
                    "commercial disputes. Foreign arbitral awards are enforceable under the New York "
                    "Convention. SEBI-regulated entities may face mandatory arbitration for capital "
                    "market disputes."
                ),
                "category": "contract",
            },
            {
                "question": "What key clauses should be reviewed in an Indian employment agreement?",
                "ground_truth": (
                    "Key clauses: (1) Confidentiality and IP assignment — must survive termination; "
                    "(2) Non-compete and non-solicitation — Indian courts are reluctant to enforce "
                    "post-termination restraints as in restraint of trade under Section 27 of the "
                    "Contract Act; (3) Termination and notice provisions — compliance with relevant "
                    "labour legislation (Shops & Establishments Act, IDA); (4) Data protection "
                    "obligations — must align with DPDP Act 2023 Section 6 employment-context "
                    "legitimate use and employee data rights; (5) Jurisdiction and governing law."
                ),
                "category": "contract",
            },
            # ── Risk Assessment ───────────────────────────────────────────
            {
                "question": "What are the key legal risks of operating a digital lending platform in India without RBI authorisation?",
                "ground_truth": (
                    "Critical risks: (1) Criminal liability under Section 45S RBI Act for accepting "
                    "deposits without a licence; (2) Prosecution under state Money Lenders Acts; "
                    "(3) FEMA enforcement for cross-border fund flows without RBI approval; "
                    "(4) RBI's 2022 Digital Lending Guidelines require all platforms to partner "
                    "only with NBFC/bank licensees and prohibit direct pass-through of loan funds; "
                    "(5) SEBI action if securities are offered; (6) Personal liability for directors "
                    "as officers in default. Risk is Critical — immediate RBI registration or a "
                    "licensed NBFC partnership is required."
                ),
                "category": "risk",
            },
            {
                "question": "What DPDP compliance risks arise when an Indian company transfers employee data to a foreign parent entity?",
                "ground_truth": (
                    "Risks include: (1) Section 16 violation if the destination country is later "
                    "restricted by Central Government notification; (2) Failure to obtain adequate "
                    "consent or notice (Sections 4-5) for cross-border transfer; (3) Section 8 DPO "
                    "obligation if the Indian entity qualifies as Significant Data Fiduciary; "
                    "(4) Data Protection Board enforcement action; (5) Employee complaints. "
                    "Mitigation: transfer risk assessment, implement contractual safeguards, monitor "
                    "Central Government notifications, maintain transfer records."
                ),
                "category": "risk",
            },
            {
                "question": "What trademark risks should a pharmaceutical company assess before launching a new drug brand in India?",
                "ground_truth": (
                    "Per Cadila Healthcare Ltd v. Cadila Pharmaceuticals Ltd (2001) 5 SCC 73, "
                    "the deceptive similarity test for pharmaceutical marks is stricter than ordinary "
                    "goods because confusion can endanger health. Courts assess phonetic, visual, and "
                    "structural similarity. Risks: (1) Injunction and damages for infringement; "
                    "(2) Ex-parte ad-interim relief stopping launch; (3) Drug Controller rejection "
                    "if name conflicts with existing approved drugs; (4) Passing-off liability even "
                    "without registration. Mitigation: full phonetic and visual clearance search "
                    "before trademark filing and Drug Controller application."
                ),
                "category": "risk",
            },
            {
                "question": "What are the legal risks of a cheque bounce under Section 138 NI Act for a business?",
                "ground_truth": (
                    "Section 138 NI Act is a criminal offence (imprisonment up to 2 years and/or "
                    "fine up to twice the cheque amount). Risks: (1) Prosecution in the bank branch "
                    "jurisdiction per Dasrath Rupsingh (2014) 9 SCC 129 — territorial jurisdiction "
                    "is the drawee bank's branch location; (2) Criminal record if convicted; "
                    "(3) Adverse credit impact; (4) Civil recovery proceedings in parallel. "
                    "Defences: dishonour not due to insufficient funds; cheque not for legally "
                    "enforceable debt; notice not properly served; payment made within 15 days of notice."
                ),
                "category": "risk",
            },
            {
                "question": "What is the legal risk of relying on an overruled precedent in Indian court proceedings?",
                "ground_truth": (
                    "Citing an overruled precedent without disclosure risks: (1) Adverse costs order "
                    "for misrepresenting the law; (2) Loss of credibility with the bench; "
                    "(3) Weakening the client's case; (4) Professional conduct issues for the advocate. "
                    "ADM Jabalpur v. Shivkant Shukla (1976, overruled by Puttaswamy 2017) is a key "
                    "example — its habeas corpus position is no longer good law. LexPilot flags "
                    "overruled judgments with [OVERRULED] warnings and cites the overruling decision "
                    "so counsel can update their arguments."
                ),
                "category": "risk",
            },
        ]
