"use client";

import { Bot, Wrench, Loader2 } from "lucide-react";

interface AgentActivityProps {
  agentName: string | null;
  tools: string[];
}

export function AgentActivity({ agentName, tools }: AgentActivityProps) {
  return (
    <div className="border-t border-gray-100 bg-gray-50 px-4 py-2">
      <div className="max-w-3xl mx-auto flex items-center gap-3 text-sm text-gray-600">
        <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
        {agentName ? (
          <span className="flex items-center gap-1.5">
            <Bot className="w-3.5 h-3.5" />
            <span className="font-medium">{agentName}</span> is working...
          </span>
        ) : (
          <span>Processing...</span>
        )}
        {tools.length > 0 && (
          <span className="flex items-center gap-1.5 text-gray-500">
            <Wrench className="w-3.5 h-3.5" />
            {tools[tools.length - 1]}
          </span>
        )}
      </div>
    </div>
  );
}
