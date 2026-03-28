"use client";

import { MessageSquare } from "lucide-react";

interface SuggestedQueriesProps {
  onSelect: (query: string) => void;
}

const SUGGESTIONS = [
  "Find precedents on right to privacy",
  "Analyze this contract for DPDP compliance",
  "What are the risks in this NDA?",
  "Has ADM Jabalpur been overruled?",
  "Compare data protection obligations under DPDP vs GDPR",
  "Extract liability clauses from this agreement",
];

export function SuggestedQueries({ onSelect }: SuggestedQueriesProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {SUGGESTIONS.map((suggestion) => (
        <button
          key={suggestion}
          onClick={() => onSelect(suggestion)}
          className="flex items-center gap-2.5 text-left px-4 py-3 rounded-xl border border-gray-200 bg-white hover:border-primary-300 hover:bg-primary-50 transition-colors text-sm text-gray-700"
        >
          <MessageSquare className="w-4 h-4 text-gray-400 flex-shrink-0" />
          {suggestion}
        </button>
      ))}
    </div>
  );
}
