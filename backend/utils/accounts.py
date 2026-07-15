"""Discover configured Alpaca accounts from environment variables."""

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


def _parse_paper_flag(value: Optional[str], default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    v = str(value).strip().lower()
    if v in {"0", "false", "no", "n", "live", "real"}:
        return False
    if v in {"1", "true", "yes", "y", "paper", "test"}:
        return True
    return default


def account_env_suffix(account_name: str) -> str:
    """Return the env-var suffix used for an account, or "" for Main."""
    return "" if account_name == "Main" else str(account_name).upper()


def account_display_name(env_suffix: str) -> str:
    """Turn an env suffix into the UI/config account name."""
    return env_suffix.capitalize()


def get_account_credentials(account_name: str) -> Dict[str, Optional[str]]:
    """
    Resolve credentials and env names for an account.

    Convention:
      Main account  -> APCA_API_KEY_ID + APCA_API_SECRET_KEY
      Extra account -> APCA_API_KEY_ID_<NAME> + APCA_API_SECRET_KEY_<NAME>

    Non-Main accounts intentionally do not fall back to Main credentials. That
    avoids accidentally sending a paper/test config to the real brokerage keys.
    """
    suffix = account_env_suffix(account_name)
    if suffix:
        key_env = f"APCA_API_KEY_ID_{suffix}"
        secret_env = f"APCA_API_SECRET_KEY_{suffix}"
        secret_alt_env = f"APCA_API_SECRET_{suffix}"
    else:
        key_env = "APCA_API_KEY_ID"
        secret_env = "APCA_API_SECRET_KEY"
        secret_alt_env = None

    secret_value = os.getenv(secret_env)
    secret_env_used = secret_env
    if not secret_value and secret_alt_env:
        secret_value = os.getenv(secret_alt_env)
        secret_env_used = secret_alt_env

    return {
        "api_key": os.getenv(key_env),
        "secret_key": secret_value,
        "key_env": key_env,
        "secret_env": secret_env_used,
    }


def get_account_paper(account_name: str) -> bool:
    """
    Resolve paper/live mode. Account-specific APCA_PAPER_<NAME> overrides the
    global APCA_PAPER flag; missing flags default to paper. PAPER_TRADING is
    accepted as a legacy alias.
    """
    suffix = account_env_suffix(account_name)
    if suffix:
        for key in (f"APCA_PAPER_{suffix}", f"PAPER_TRADING_{suffix}"):
            specific = os.getenv(key)
            if specific is not None:
                return _parse_paper_flag(specific, default=True)
    global_flag = os.getenv("APCA_PAPER")
    if global_flag is None:
        global_flag = os.getenv("PAPER_TRADING")
    return _parse_paper_flag(global_flag, default=True)


def get_configured_account_details() -> List[Dict[str, object]]:
    """Return configured accounts without exposing secret values."""
    accounts: List[Dict[str, object]] = []
    main_creds = get_account_credentials("Main")
    if main_creds["api_key"] and main_creds["secret_key"]:
        main_paper = get_account_paper("Main")
        accounts.append({
            "name": "Main",
            "key_env": main_creds["key_env"],
            "secret_env": main_creds["secret_env"],
            "paper": main_paper,
            "mode": "paper" if main_paper else "live",
        })

    suffixes = sorted(
        key[len("APCA_API_KEY_ID_"):]
        for key in os.environ
        if key.startswith("APCA_API_KEY_ID_")
    )
    for suffix in suffixes:
        name = account_display_name(suffix)
        creds = get_account_credentials(name)
        if creds["api_key"] and creds["secret_key"]:
            paper = get_account_paper(name)
            accounts.append({
                "name": name,
                "key_env": creds["key_env"],
                "secret_env": creds["secret_env"],
                "paper": paper,
                "mode": "paper" if paper else "live",
            })

    return accounts


def get_configured_accounts() -> List[str]:
    """Return account names whose API keys are present in .env."""
    return [a["name"] for a in get_configured_account_details()]
