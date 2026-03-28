"use client";

import { useState } from "react";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ChevronDown,
  ChevronUp,
  Download,
} from "lucide-react";
import type { ComplianceReportData, ComplianceGap } from "@/types";

interface ComplianceReportProps {
  report: ComplianceReportData;
}

const RISK_CONFIG = {
  compliant: {
    icon: ShieldCheck,
    label: "Compliant",
    className: "bg-green-100 text-green-800",
  },
  minor_gaps: {
    icon: ShieldAlert,
    label: "Minor Gaps",
    className: "bg-yellow-100 text-yellow-800",
  },
  major_gaps: {
    icon: ShieldAlert,
    label: "Major Gaps",
    className: "bg-orange-100 text-orange-800",
  },
  non_compliant: {
    icon: ShieldX,
    label: "Non-Compliant",
    className: "bg-red-100 text-red-800",
  },
};

const GAP_RISK_COLORS = {
  critical: "border-l-red-500 bg-red-50",
  high: "border-l-orange-500 bg-orange-50",
  medium: "border-l-yellow-500 bg-yellow-50",
  low: "border-l-green-500 bg-green-50",
};

function GapCard({ gap }: { gap: ComplianceGap }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`border-l-4 rounded-r-lg p-3 ${GAP_RISK_COLORS[gap.riskLevel]}`}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wide text-gray-500">
              {gap.dpdpSection}
            </span>
            <span
              className={`text-xs px-1.5 py-0.5 rounded font-medium ${
                gap.riskLevel === "critical"
                  ? "bg-red-200 text-red-800"
                  : gap.riskLevel === "high"
                  ? "bg-orange-200 text-orange-800"
                  : gap.riskLevel === "medium"
                  ? "bg-yellow-200 text-yellow-800"
                  : "bg-green-200 text-green-800"
              }`}
            >
              {gap.riskLevel}
            </span>
            <span className="text-xs text-gray-500">{gap.gapType}</span>
          </div>
          <p className="text-sm text-gray-800 mt-1 truncate">
            {gap.clauseText}
          </p>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400 flex-shrink-0" />
        )}
      </button>
      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
          <div>
            <span className="text-xs font-medium text-gray-500">Location:</span>
            <p className="text-sm text-gray-700">{gap.clauseLocation}</p>
          </div>
          <div>
            <span className="text-xs font-medium text-gray-500">Full Clause:</span>
            <p className="text-sm text-gray-700">{gap.clauseText}</p>
          </div>
          <div>
            <span className="text-xs font-medium text-gray-500">Remediation:</span>
            <p className="text-sm text-gray-700">{gap.remediation}</p>
          </div>
          <div className="text-xs text-gray-500">
            Confidence: {Math.round(gap.confidence * 100)}%
          </div>
        </div>
      )}
    </div>
  );
}

export function ComplianceReport({ report }: ComplianceReportProps) {
  const riskConfig = RISK_CONFIG[report.overallRisk];
  const RiskIcon = riskConfig.icon;

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `compliance-report-${report.documentId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="font-semibold text-gray-900">DPDP Compliance Report</h3>
          <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${riskConfig.className}`}
          >
            <RiskIcon className="w-3.5 h-3.5" />
            {riskConfig.label}
          </span>
        </div>
        <button
          onClick={handleExport}
          className="flex items-center gap-1.5 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>

      {/* Executive Summary */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <p className="text-sm text-gray-700">{report.executiveSummary}</p>
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span>{report.totalClausesAnalyzed} clauses analyzed</span>
          <span>{report.gaps.length} gaps found</span>
          <span>{report.sectionsAnalyzed.length} DPDP sections checked</span>
        </div>
      </div>

      {/* Gaps */}
      {report.gaps.length > 0 && (
        <div className="px-4 py-3 space-y-2">
          {report.gaps.map((gap, i) => (
            <GapCard key={i} gap={gap} />
          ))}
        </div>
      )}
    </div>
  );
}
