"use client";

import { AlertTriangle } from "lucide-react";

export function Disclaimer() {
  return (
    <div className="bg-amber-50 border-b border-amber-200 px-4 py-2.5 flex items-center gap-2">
      <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
      <p className="text-sm text-amber-800">
        <strong>LexPilot provides legal information and analysis, not legal advice.</strong>{" "}
        Always consult a qualified advocate before acting on any findings.
      </p>
    </div>
  );
}
