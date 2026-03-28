"use client";

import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";
import type { Citation } from "@/types";

interface CitationBadgeProps {
  citation: Citation;
}

const STATUS_CONFIG = {
  verified: {
    icon: CheckCircle,
    label: "Verified",
    className: "bg-green-50 text-green-700 border-green-200",
    iconClassName: "text-green-500",
  },
  overruled: {
    icon: AlertTriangle,
    label: "Overruled",
    className: "bg-yellow-50 text-yellow-700 border-yellow-200",
    iconClassName: "text-yellow-500",
  },
  removed: {
    icon: XCircle,
    label: "Removed",
    className: "bg-red-50 text-red-700 border-red-200",
    iconClassName: "text-red-500",
  },
};

export function CitationBadge({ citation }: CitationBadgeProps) {
  const config = STATUS_CONFIG[citation.status];
  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${config.className}`}
      title={`${config.label}: ${citation.text}`}
    >
      <Icon className={`w-3 h-3 ${config.iconClassName}`} />
      <span className="font-mono">{citation.text}</span>
      <span className="opacity-75">{config.label}</span>
    </span>
  );
}
