"""One-time Garmin Connect login for garmin_health.py.

Run this once in a terminal:
    python3 garmin_login.py            # password hidden while typing
    python3 garmin_login.py --visible  # password shown while typing

It asks for your Garmin email / password (and MFA code if enabled), then saves OAuth
tokens to ~/.garminconnect/garmin_tokens.json. The dashboard only ever reads those
tokens, so your password is never stored or typed into the Streamlit app. Tokens
refresh themselves; re-run this only if the dashboard says the session expired.

Using a hosted copy of the dashboard without typing your password into it: answer "y"
at the end to print a token string, and paste it into the app's "Use a token instead" tab.
Treat that string like a password — it gives access to your Garmin account.
"""
import getpass
import os
import sys
from pathlib import Path

from garminconnect import Garmin

TOKENSTORE = os.getenv("GARMINTOKENS_PATH", "~/.garminconnect")


def main():
    email = input("Garmin email: ").strip()
    if "--visible" in sys.argv:
        password = input("Garmin password (visible): ")
    else:
        password = getpass.getpass("Garmin password (hidden — nothing shows while typing): ")
    print(f"({len(password)} characters received, logging in…)")
    if not password:
        sys.exit("No password entered — run again.")
    garmin = Garmin(email=email, password=password, return_on_mfa=True)
    needs_mfa, state = garmin.login()
    if needs_mfa:
        code = input("MFA code: ").strip()
        garmin.resume_login(state, code)

    path = Path(TOKENSTORE).expanduser()
    garmin.client.dump(str(path))
    os.chmod(path / "garmin_tokens.json", 0o600)
    print(f"\nLogged in. Tokens saved to {path}/garmin_tokens.json")
    if input("Print a token string to paste into a hosted copy of the dashboard? [y/N] ").lower() == "y":
        print(garmin.client.dumps())


if __name__ == "__main__":
    main()
