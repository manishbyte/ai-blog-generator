
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

INVOKE_URL = (
    "https://ai.api.nvidia.com/v1/genai/"
    "black-forest-labs/flux-3-image-klein"
)

OUTPUT_DIR = Path("images")
OUTPUT_FILE = OUTPUT_DIR / "flux_test.png"


def generate_image(prompt: str) -> Path:
    if not API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing. Add it to your .env file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "steps": 50,
    }

    print("Sending image-generation request to NVIDIA...")

    response = requests.post(
        INVOKE_URL,
        headers=headers,
        json=payload,
        timeout=300,
    )

    if not response.ok:
        raise RuntimeError(
            f"NVIDIA API failed with HTTP {response.status_code}:\n"
            f"{response.text}"
        )

    result = response.json()

    # NVIDIA image endpoints commonly return base64 image data
    # inside the first item of the "artifacts" array.
    artifacts = result.get("artifacts", [])

    if artifacts and artifacts[0].get("base64"):
        image_data = artifacts[0]["base64"]
    else:
        raise RuntimeError(
            "The request succeeded, but no base64 image was found "
            f"in the response:\n{result}"
        )

    image_bytes = base64.b64decode(image_data)
    OUTPUT_FILE.write_bytes(image_bytes)

    print("Image generated successfully.")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print(f"File size: {len(image_bytes):,} bytes")

    return OUTPUT_FILE


if __name__ == "__main__":
    prompt = (
        "A professional editorial illustration of an AI research lab, "
        "modern computers, cinematic blue lighting, realistic details, "
        "clean composition, high quality, no text."
    )

    try:
        generate_image(prompt)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Image generation failed: {exc}")

