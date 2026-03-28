from agents import Runner
from agents.orchestrator import orchestrator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    research_session_id: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    async def stream():
        result = Runner.run_streamed(orchestrator, request.message)
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if hasattr(event.data, "delta") and event.data.delta:
                    yield f"data: {json.dumps({'type': 'text', 'content': event.data.delta})}\n\n"
            elif event.type == "agent_updated_stream_event":
                yield f"data: {json.dumps({'type': 'agent_event', 'agent': event.new_agent.name})}\n\n"
            elif event.type == "tool_called_stream_event":
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': event.tool_name})}\n\n"

        # Final result
        final = await result.final_output()
        yield f"data: {json.dumps({'type': 'final', 'content': str(final)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
