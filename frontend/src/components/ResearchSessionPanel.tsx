"use client";

import { useState, useEffect, useCallback } from "react";
import { Plus, FolderOpen, BookOpen, FileCheck, Loader2 } from "lucide-react";
import {
  createResearchSession,
  listResearchSessions,
} from "@/lib/research-sessions";
import type { ResearchSessionData } from "@/types";

interface ResearchSessionPanelProps {
  activeSessionId?: string;
  onSessionSelect: (sessionId: string | undefined) => void;
}

export function ResearchSessionPanel({
  activeSessionId,
  onSessionSelect,
}: ResearchSessionPanelProps) {
  const [sessions, setSessions] = useState<ResearchSessionData[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const userId = "default-user";

  const loadSessions = useCallback(async () => {
    try {
      const data = await listResearchSessions(userId);
      setSessions(data);
    } catch {
      // Backend may not be available yet
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setIsLoading(true);
    try {
      const session = await createResearchSession(userId, newName.trim());
      setSessions((prev) => [session, ...prev]);
      onSessionSelect(session.id);
      setNewName("");
      setIsCreating(false);
    } catch {
      // Handle error silently
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <h2 className="font-semibold text-gray-900 flex items-center gap-2">
            <FolderOpen className="w-4 h-4" />
            Research Sessions
          </h2>
          <button
            onClick={() => setIsCreating(!isCreating)}
            className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
            title="New session"
          >
            <Plus className="w-4 h-4 text-gray-600" />
          </button>
        </div>

        {isCreating && (
          <div className="flex gap-2 mt-2">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              placeholder="Session name..."
              className="flex-1 text-sm border border-gray-300 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary-500"
              autoFocus
            />
            <button
              onClick={handleCreate}
              disabled={!newName.trim() || isLoading}
              className="text-sm px-3 py-1.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Create"}
            </button>
          </div>
        )}
      </div>

      {/* No session (general chat) */}
      <div className="flex-1 overflow-y-auto">
        <button
          onClick={() => onSessionSelect(undefined)}
          className={`w-full text-left px-4 py-3 text-sm border-b border-gray-100 hover:bg-gray-50 transition-colors ${
            !activeSessionId ? "bg-primary-50 border-l-2 border-l-primary-500" : ""
          }`}
        >
          <div className="font-medium text-gray-900">General Chat</div>
          <div className="text-xs text-gray-500">No research session</div>
        </button>

        {sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => onSessionSelect(session.id)}
            className={`w-full text-left px-4 py-3 text-sm border-b border-gray-100 hover:bg-gray-50 transition-colors ${
              activeSessionId === session.id
                ? "bg-primary-50 border-l-2 border-l-primary-500"
                : ""
            }`}
          >
            <div className="font-medium text-gray-900">{session.name}</div>
            <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
              {session.precedentsFound.length > 0 && (
                <span className="flex items-center gap-1">
                  <BookOpen className="w-3 h-3" />
                  {session.precedentsFound.length} precedents
                </span>
              )}
              {session.complianceFindings.length > 0 && (
                <span className="flex items-center gap-1">
                  <FileCheck className="w-3 h-3" />
                  {session.complianceFindings.length} findings
                </span>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
