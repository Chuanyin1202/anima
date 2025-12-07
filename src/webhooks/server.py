"""Webhook server for receiving external notifications.

This server runs as a background service to receive webhook notifications
from external providers (e.g., Apify) and triggers interaction cycles.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


class WebhookServer:
    """HTTP server for receiving webhook notifications."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        webhook_secret: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.webhook_secret = webhook_secret
        self.app = FastAPI(title="Anima Webhook Server")
        self.handlers: dict[str, Callable] = {}
        self._setup_routes()

    def _setup_routes(self):
        """Setup webhook routes."""

        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

        @self.app.post("/webhooks/{provider}")
        async def webhook_handler(provider: str, request: Request):
            """Generic webhook endpoint."""
            if provider not in self.handlers:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"No handler for provider: {provider}"},
                )

            # Verify secret if configured
            if self.webhook_secret:
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Missing or invalid authorization"},
                    )
                token = auth_header.split(" ", 1)[1]
                if token != self.webhook_secret:
                    return JSONResponse(
                        status_code=403,
                        content={"error": "Invalid webhook secret"},
                    )

            # Parse payload
            try:
                payload = await request.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("webhook_invalid_json", error=str(exc))
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid JSON payload"},
                )

            # Call handler
            try:
                handler = self.handlers[provider]
                await handler(payload)
                return {"status": "ok", "message": "Webhook processed"}
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "webhook_handler_failed",
                    provider=provider,
                    error=str(exc),
                    exc_info=True,
                )
                return JSONResponse(
                    status_code=500,
                    content={"error": "Webhook handler failed"},
                )

    def register_handler(self, provider: str, handler: Callable):
        """Register a webhook handler for a provider.

        Args:
            provider: Provider name (e.g., 'apify')
            handler: Async callable that processes webhook payload
        """
        self.handlers[provider] = handler
        logger.info("webhook_handler_registered", provider=provider)

    async def start(self):
        """Start the webhook server."""
        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)

        logger.info("webhook_server_starting", host=self.host, port=self.port)
        await server.serve()

    def run(self):
        """Run the webhook server (blocking)."""
        import uvicorn

        logger.info("webhook_server_starting", host=self.host, port=self.port)
        uvicorn.run(self.app, host=self.host, port=self.port)
