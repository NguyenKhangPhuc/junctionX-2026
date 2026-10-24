"""Agent system prompt definitions package."""

from app.agents.prompts.career import CAREER_AGENT_SYSTEM_PROMPT
from app.agents.prompts.email import EMAIL_AGENT_SYSTEM_PROMPT
from app.agents.prompts.supervisor import SUPERVISOR_SYSTEM_PROMPT

__all__ = [
    "CAREER_AGENT_SYSTEM_PROMPT",
    "EMAIL_AGENT_SYSTEM_PROMPT",
    "SUPERVISOR_SYSTEM_PROMPT",
]
