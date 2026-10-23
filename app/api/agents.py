"""FastAPI router for AI agent interactions and Server-Sent Events (SSE) streaming."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agents.runner import stream_agent_workflow
from app.schemas.agents import AgentChatRequest

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/chat")
async def chat_with_agents(request: AgentChatRequest) -> StreamingResponse:
    """Communicate with the Multi-Agent system with real-time SSE event streaming.

    Streams token chunks for typewriter UI rendering along with tool start/end execution events.

    Args:
        request (AgentChatRequest): User instruction and optional thread identifier.

    Returns:
        StreamingResponse: Real-time event stream formatted as text/event-stream.
    """
    return StreamingResponse(
        stream_agent_workflow(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
