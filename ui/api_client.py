"""Synchronous HTTP client for the BlogForge FastAPI backend."""

import os
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv(
    "BLOGFORGE_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


class APIError(Exception):
    """Raised when a BlogForge API request fails."""


class BlogForgeAPI:
    """Authenticated client for BlogForge API operations."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        headers: dict[str, str] = {}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            response = httpx.request(
                method,
                f"{API_BASE_URL}{path}",
                headers=headers,
                json=json,
                timeout=300.0,
            )
        except httpx.TimeoutException as exc:
            raise APIError(
                "The API request timed out. Blog generation can take "
                "several minutes. Check the backend before retrying."
            ) from exc
        except httpx.HTTPError as exc:
            raise APIError(
                f"Cannot connect to BlogForge API at {API_BASE_URL}. "
                "Make sure FastAPI is running."
            ) from exc

        if response.is_error:
            try:
                body = response.json()
                detail = body.get("detail", body)
            except ValueError:
                detail = response.text or "Unknown API error"

            raise APIError(
                f"HTTP {response.status_code}: {detail}"
            )

        return response

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    def register(
        self,
        email: str,
        password: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/register",
            json={
                "email": email,
                "password": password,
            },
        ).json()

    def login(
        self,
        email: str,
        password: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/auth/login",
            json={
                "email": email,
                "password": password,
            },
        ).json()

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/auth/me").json()

    def logout(self) -> None:
        self._request("POST", "/auth/logout")

    # --------------------------------------------------------
    # Conversations
    # --------------------------------------------------------

    def list_conversations(self) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/conversations",
        ).json()

    def create_conversation(
        self,
        title: str = "New blog",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/conversations",
            json={"title": title},
        ).json()

    def get_conversation(
        self,
        conversation_id: str,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/conversations/{conversation_id}",
        ).json()

    def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        self._request(
            "DELETE",
            f"/conversations/{conversation_id}",
        )

    def generate_blog(
        self,
        conversation_id: str,
        topic: str,
        as_of: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"topic": topic}

        if as_of:
            payload["as_of"] = as_of

        return self._request(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json=payload,
        ).json()

    # --------------------------------------------------------
    # Image attachments stored in PostgreSQL
    # --------------------------------------------------------

    def get_attachment(
        self,
        attachment_id: str,
    ) -> bytes:
        response = self._request(
            "GET",
            f"/attachments/{attachment_id}",
        )
        return response.content

    # --------------------------------------------------------
    # Generated images stored in the project's images folder
    # --------------------------------------------------------

    def get_generated_image(
        self,
        filename: str,
    ) -> bytes:
        """Retrieve a generated image using the current user's token."""

        safe_filename = filename.replace("\\", "/").split("/")[-1]

        if (
            not safe_filename
            or safe_filename in {".", ".."}
            or safe_filename != filename
        ):
            raise APIError("Invalid generated image filename.")

        encoded_filename = quote(safe_filename, safe="")

        response = self._request(
            "GET",
            f"/generated-images/{encoded_filename}",
        )

        content_type = response.headers.get("content-type", "")

        if not content_type.lower().startswith("image/"):
            raise APIError(
                f"The generated-image endpoint returned "
                f"an unexpected content type: {content_type or 'unknown'}."
            )

        return response.content