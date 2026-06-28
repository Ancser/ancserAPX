"""Discover configured Alpaca accounts from environment variables."""

import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


def get_configured_accounts() -> List[str]:
    """
    Returns a list of account names whose API keys are present in .env.

    Convention:
      Main account  → APCA_API_KEY_ID + APCA_API_SECRET_KEY
      Extra account → APCA_API_KEY_ID_<NAME> + APCA_API_SECRET_KEY_<NAME>
    """
    accounts = []
    if os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY"):
        accounts.append("Main")

    # Scan for named accounts
    for key, _ in os.environ.items():
        if key.startswith("APCA_API_KEY_ID_"):
            name_upper = key[len("APCA_API_KEY_ID_"):]
            secret_key = f"APCA_API_SECRET_KEY_{name_upper}"
            secret_alt = f"APCA_API_SECRET_{name_upper}"
            if os.getenv(secret_key) or os.getenv(secret_alt):
                name = name_upper.capitalize()
                if name not in accounts:
                    accounts.append(name)

    return accounts
