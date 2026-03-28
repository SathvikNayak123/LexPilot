import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const response = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: body.messages[body.messages.length - 1].content,
      session_id: body.session_id,
      research_session_id: body.research_session_id,
    }),
  });

  if (!response.ok) {
    return new Response(
      JSON.stringify({ error: "Backend request failed" }),
      { status: response.status, headers: { "Content-Type": "application/json" } }
    );
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
