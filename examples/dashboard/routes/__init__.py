"""Dashboard route registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def register_routes(app: FastAPI) -> None:
    """Include all page routers."""
    from .pages import router

    app.include_router(router)
