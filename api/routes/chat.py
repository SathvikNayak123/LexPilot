from agents import Runner
from agents.run_config import RunConfig
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from lexpilot_agents.orchestrator import orchestrator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from api.provider import openrouter_provider

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    research_session_id: str | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    async def stream():
        run_config = RunConfig(model_provider=openrouter_provider, tracing_disabled=True)
        try:
            result = Runner.run_streamed(orchestrator, request.message, run_config=run_config)
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    data = event.data
                    if hasattr(data, "delta") and data.delta:
                        yield f"data: {json.dumps({'type': 'text', 'content': data.delta})}\n\n"
                elif event.type == "agent_updated_stream_event":
                    yield f"data: {json.dumps({'type': 'agent_event', 'agent': event.new_agent.name})}\n\n"
                elif event.type == "run_item_stream_event":
                    if event.name == "tool_called":
                        tool_name = getattr(event.item, "name", str(event.item))
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name})}\n\n"

            final_text = result.final_output if isinstance(result.final_output, str) else ""
            yield f"data: {json.dumps({'type': 'final', 'content': final_text})}\n\n"

        except InputGuardrailTripwireTriggered as e:
            info = e.guardrail_result.output.output_info if e.guardrail_result else {}
            category = getattr(info, "category", "blocked") if hasattr(info, "category") else "blocked"
            if category == "injection":
                msg = "Your query was flagged as a potential prompt injection and has been blocked."
            elif category == "off_topic":
                msg = "This system handles legal research queries only. Please ask a legal question."
            else:
                msg = "Your query could not be processed. Please rephrase as a legal research question."
            yield f"data: {json.dumps({'type': 'final', 'content': msg})}\n\n"

        except OutputGuardrailTripwireTriggered:
            msg = ("The response contained unverifiable citations and was blocked. "
                   "Please try rephrasing your query for a more accurate answer.")
            yield f"data: {json.dumps({'type': 'final', 'content': msg})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'final', 'content': f'An error occurred: {str(e)}'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
