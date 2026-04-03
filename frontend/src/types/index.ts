export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  agentName?: string;
  toolCalls?: string[];
  citations?: Citation[];
  confidence?: ConfidenceData;
  complianceReport?: ComplianceReportData;
  timestamp: Date;
  isLoading?: boolean;
  loadingAgent?: string;
}

export interface Citation {
  text: string;
  status: "verified" | "overruled" | "removed";
}

export interface ConfidenceData {
  tier: "green" | "yellow" | "red";
  score: number;
  reasoning: string;
  sourcesCount: number;
  conflictingSources: boolean;
}

export interface ComplianceGap {
  clauseText: string;
  clauseLocation: string;
  dpdpSection: string;
  gapType: "missing" | "insufficient" | "conflicting";
  riskLevel: "critical" | "high" | "medium" | "low";
  remediation: string;
  confidence: number;
}

export interface ComplianceReportData {
  documentId: string;
  overallRisk: "compliant" | "minor_gaps" | "major_gaps" | "non_compliant";
  totalClausesAnalyzed: number;
  gaps: ComplianceGap[];
  sectionsAnalyzed: string[];
  executiveSummary: string;
}

export interface ResearchSessionData {
  id: string;
  userId: string;
  name: string;
  summary: Record<string, unknown>;
  precedentsFound: string[];
  clausesAnalyzed: Record<string, unknown>[];
  complianceFindings: Record<string, unknown>[];
  createdAt: string;
  updatedAt: string;
}

export interface SSEEvent {
  type: "text" | "agent_event" | "tool_call" | "final";
  content?: string;
  agent?: string;
  tool?: string;
}
