import type { ResearchSessionData } from "@/types";

const API_BASE = "/api/backend";

export async function createResearchSession(
  userId: string,
  name: string
): Promise<ResearchSessionData> {
  const response = await fetch(`${API_BASE}/research-sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, name }),
  });
  if (!response.ok) throw new Error("Failed to create session");
  return response.json();
}

export async function listResearchSessions(
  userId: string
): Promise<ResearchSessionData[]> {
  const response = await fetch(
    `${API_BASE}/research-sessions?user_id=${encodeURIComponent(userId)}`
  );
  if (!response.ok) throw new Error("Failed to list sessions");
  return response.json();
}

export async function getResearchSession(
  sessionId: string
): Promise<ResearchSessionData> {
  const response = await fetch(`${API_BASE}/research-sessions/${sessionId}`);
  if (!response.ok) throw new Error("Failed to get session");
  return response.json();
}
