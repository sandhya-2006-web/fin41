import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.aczen.in/nova-api/v1"

API_KEY = os.getenv("NOVA_API_KEY")

if not API_KEY:
    raise RuntimeError("NOVA_API_KEY is not set")

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {API_KEY}"
})


def nova_get(path, **params):
    """
    Make a GET request to Nova API.
    Handles pagination-related requests and basic retry behavior.
    """

    for attempt in range(4):

        response = session.get(
            BASE_URL + path,
            params=params,
            timeout=30
        )

        # Rate limit
        if response.status_code == 429:
            retry_after = int(
                response.headers.get("Retry-After", 60)
            )
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        # Temporary upstream error
        if response.status_code == 502:
            wait_time = 2 ** attempt
            print(
                f"Temporary API error. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)
            continue

        body = response.json()

        if not response.ok:
            error = body.get("error", {})

            raise RuntimeError(
                f"Nova API error {response.status_code}: "
                f"{error.get('code')} - "
                f"{error.get('message')} "
                f"(request {body.get('request_id')})"
            )

        return body

    raise RuntimeError("Nova API retries exhausted")


def list_all(path, **params):
    """
    Fetch all pages from a Nova list endpoint.
    """

    rows = []
    offset = 0
    limit = 200

    while True:

        page = nova_get(
            path,
            **params,
            limit=limit,
            offset=offset
        )

        rows.extend(page["data"])

        pagination = page["pagination"]

        print(
            f"Fetched {len(rows)} / "
            f"{pagination['total']} records"
        )

        if not pagination["has_more"]:
            break

        offset += limit

    return rows