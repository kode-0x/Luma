"""Chat/query API endpoints with streaming support."""

import json

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.dependencies import get_chat_service, get_settings_dep
from backend.core.config import Settings
from backend.core.exceptions import GenerationError, RetrievalError
from backend.models.queries import ChatRequest, ChatResponse, ErrorResponse
from backend.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Generation or retrieval error"},
    },
)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Ask a question and get an answer with citations.

    The query is processed through the full RAG pipeline:
    retrieval, reranking, and generation.

    Args:
        request: Chat request containing the query and optional document filters.
        service: Chat service (injected).

    Returns:
        ChatResponse with the answer and supporting citations.
    """
    try:
        response = await service.query(
            query=request.query,
            document_ids=request.document_ids or None,
            model=request.model,
        )
        return response
    except RetrievalError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except GenerationError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """Stream an answer token-by-token using Server-Sent Events.

    Produces a stream of SSE events:
    - `data: {"type": "token", "content": "..."}` for each generated token.
    - `data: {"type": "done", "content": ""}` when generation is complete.
    - `data: {"type": "error", "content": "..."}` on failure.

    Args:
        request: Chat request containing the query and optional document filters.
        service: Chat service (injected).

    Returns:
        StreamingResponse with SSE content type.
    """

    async def event_generator():
        """Generate SSE events from the streaming pipeline."""
        try:
            async for token in service.query_stream(
                query=request.query,
                document_ids=request.document_ids or None,
                model=request.model,
            ):
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

            done_event = json.dumps({"type": "done", "content": ""})
            yield f"data: {done_event}\n\n"

        except (RetrievalError, GenerationError) as exc:
            error_event = json.dumps({"type": "error", "content": exc.message})
            yield f"data: {error_event}\n\n"
        except Exception as exc:
            error_event = json.dumps({"type": "error", "content": str(exc)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models")
async def list_models(
    settings: Settings = Depends(get_settings_dep),
) -> list[dict]:
    """Fetch available models from OpenRouter.

    Returns a filtered list of chat-capable models with pricing info.
    Free models are prioritized at the top.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get("https://openrouter.ai/api/v1/models")

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to fetch models from OpenRouter")

        data = response.json()
        raw_models = data.get("data", [])

        models = []
        for m in raw_models:
            model_id = m.get("id", "")
            name = m.get("name", model_id)
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0") or "0")
            completion_price = float(pricing.get("completion", "0") or "0")
            is_free = prompt_price == 0 and completion_price == 0

            models.append(
                {
                    "id": model_id,
                    "name": name,
                    "prompt_price": prompt_price,
                    "completion_price": completion_price,
                    "is_free": is_free,
                }
            )

        models.sort(key=lambda x: (not x["is_free"], x["prompt_price"], x["name"]))

        return models

    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"OpenRouter request failed: {exc}") from exc
