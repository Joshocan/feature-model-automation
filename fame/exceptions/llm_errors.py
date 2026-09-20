from .user_messages import UserMessageError


class LLMTimeoutError(UserMessageError):
    """Raised when an LLM HTTP request times out."""

    def __init__(self, host: str, model: str, timeout_s: int):
        super().__init__(
            f"LLM request timed out after {timeout_s}s (host={host}, model={model}). "
            "Please retry or switch to a closer/faster host."
        )


class LLMHTTPError(UserMessageError):
    """Raised when an LLM HTTP request returns a non-2xx status."""

    def __init__(self, host: str, model: str, status: int, detail: str = ""):
        msg = (
            f"LLM request failed (status={status}, host={host}, model={model}). "
            "Please retry, check API key/quota, or switch host/model."
        )
        if detail:
            msg += f" Details: {detail}"
        super().__init__(msg)
