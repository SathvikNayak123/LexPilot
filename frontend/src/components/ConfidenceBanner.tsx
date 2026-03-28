"use client";

import { ShieldCheck, ShieldAlert, ShieldX } from "lucide-react";
import type { ConfidenceData } from "@/types";

interface ConfidenceBannerProps {
  confidence: ConfidenceData;
}

const TIER_CONFIG = {
  green: {
    icon: ShieldCheck,
    className: "bg-green-50 border-green-200 text-green-800",
    iconClassName: "text-green-500",
    label: "High Confidence",
  },
  yellow: {
    icon: ShieldAlert,
    className: "bg-yellow-50 border-yellow-200 text-yellow-800",
    iconClassName: "text-yellow-500",
    label: "Moderate Confidence",
  },
  red: {
    icon: ShieldX,
    className: "bg-red-50 border-red-200 text-red-800",
    iconClassName: "text-red-500",
    label: "Low Confidence",
  },
};

export function ConfidenceBanner({ confidence }: ConfidenceBannerProps) {
  const config = TIER_CONFIG[confidence.tier];
  const Icon = config.icon;

  return (
    <div
      className={`flex items-center gap-3 rounded-lg border px-4 py-3 ${config.className}`}
    >
      <Icon className={`w-5 h-5 flex-shrink-0 ${config.iconClassName}`} />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{config.label}</span>
          <span className="text-xs opacity-75">
            ({Math.round(confidence.score * 100)}%)
          </span>
        </div>
        <p className="text-sm mt-0.5">{confidence.reasoning}</p>
      </div>
      <div className="text-xs text-right flex-shrink-0">
        <div>{confidence.sourcesCount} source(s)</div>
        {confidence.conflictingSources && (
          <div className="text-red-600 font-medium">Conflicts detected</div>
        )}
      </div>
    </div>
  );
}
