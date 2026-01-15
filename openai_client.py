import os
from functools import lru_cache

from openai import OpenAI


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _api_key_header_name() -> str:
    # For Azure API Management this is often `Ocp-Apim-Subscription-Key`.
    # For Azure OpenAI it is typically `api-key`.
    return (os.getenv("OPENAI_API_KEY_HEADER_NAME") or "api-key").strip()


def _get_required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your shell or in a local .env file."
        )
    return v


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Return a singleton OpenAI client configured for the **company Azure gateway**
    in OpenAI-compatible mode, matching `verify_openai_env.py`:
      client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    Notes:
    - Your `OPENAI_BASE_URL` should look like: https://<host>/openai/v1
    - Model names should be your Azure "deployment names" (e.g. `gpt-4o-model`).
    """
    # Load `.env` if python-dotenv is installed; otherwise rely on shell env vars.
    try:
        from dotenv import load_dotenv  # type: ignore
        # In local dev, it's common to update `.env` frequently; keep it consistent with `verify_openai_env.py`.
        load_dotenv(override=True)
    except Exception:
        pass
    api_key = _get_required_env("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None

    # ---- OpenAI-compatible / gateway support ----
    # Some Azure gateways / API management endpoints require a subscription header
    # rather than Bearer auth. Opt-in via:
    # - OPENAI_FORCE_API_KEY_HEADER=true
    # - OPENAI_API_KEY_HEADER_NAME=Ocp-Apim-Subscription-Key (or api-key)
    default_headers = None
    if _truthy(os.getenv("OPENAI_FORCE_API_KEY_HEADER")):
        default_headers = {_api_key_header_name(): api_key}

    # SSL / corporate proxy note:
    # If your network injects a custom root cert, Python/httpx may fail with
    # CERTIFICATE_VERIFY_FAILED. Provide a PEM bundle path via OPENAI_CA_BUNDLE
    # (or SSL_CERT_FILE / REQUESTS_CA_BUNDLE) to trust that root.
    ca_bundle = (
        os.getenv("OPENAI_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
    )
    if ca_bundle:
        try:
            import httpx

            http_client = httpx.Client(verify=ca_bundle, timeout=60.0)
            if base_url:
                return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client, default_headers=default_headers)
            return OpenAI(api_key=api_key, http_client=http_client, default_headers=default_headers)
        except Exception:
            # Fall back to default TLS settings
            pass

    # Intentionally do NOT set base_url here: default is OpenAI official endpoint.
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
    return OpenAI(api_key=api_key, default_headers=default_headers)

