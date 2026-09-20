from __future__ import annotations


class UserMessageError(RuntimeError):
    """Base class for user-facing errors with clean messages."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class PlaceholderError(UserMessageError):
    def __init__(self, placeholders):
        super().__init__(
            "Prompt still has unresolved placeholders: "
            f"{', '.join(sorted(placeholders))}. "
            "Please fill all required template variables."
        )


class LLMTimeoutError(UserMessageError):
    def __init__(self, host: str, model: str, timeout_s: int):
        super().__init__(
            f"LLM request timed out after {timeout_s}s (host={host}, model={model}). "
            "Retry later, use a closer/faster host, or lower the input size."
        )


class MissingKeyError(UserMessageError):
    def __init__(self, key_name: str, path: str):
        super().__init__(
            f"API key '{key_name}' not found. Provide it by either:\n"
            f"- placing the key file at: {path}\n"
            f"- OR exporting the env var {key_name} in your shell.\n"
            "Then rerun the command."
        )


class MissingChunksError(UserMessageError):
    def __init__(self, chunks_dir: str):
        super().__init__(
            f"No chunks.json files found in {chunks_dir}. "
            "Run ingestion first or add input files to data/raw."
        )


def format_error(e: Exception) -> str:
    return getattr(e, "message", str(e))
