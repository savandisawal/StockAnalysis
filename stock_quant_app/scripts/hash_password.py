"""Generate a bcrypt hash for an AUTH_USERS entry.

Usage:
    python -m scripts.hash_password
    python -m scripts.hash_password mypassword   (non-interactive)

Paste the output into .env as:  AUTH_USERS=youruser:<hash>
Multiple users are comma-separated.
"""

import getpass
import sys

from app.auth import hash_password


def main() -> None:
    if len(sys.argv) > 1:
        password = sys.argv[1]
    else:
        password = getpass.getpass("Password to hash: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.", file=sys.stderr)
            raise SystemExit(1)

    if len(password) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        raise SystemExit(1)

    print(hash_password(password))


if __name__ == "__main__":
    main()
