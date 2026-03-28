from pydantic import BaseModel, Field
from typing import Literal


class ComplianceGap(BaseModel):
    clause_text: str
    clause_location: str  # e.g., "Section 4.2, paragraph 3"
    dpdp_section: str     # Validated against known sections
    gap_type: Literal["missing", "insufficient", "conflicting"]
    risk_level: Literal["critical", "high", "medium", "low"]
    remediation: str
    confidence: float = Field(ge=0, le=1)


class ComplianceReport(BaseModel):
    document_id: str
    overall_risk: Literal["compliant", "minor_gaps", "major_gaps", "non_compliant"]
    total_clauses_analyzed: int
    gaps: list[ComplianceGap]
    sections_analyzed: list[str]
    executive_summary: str
