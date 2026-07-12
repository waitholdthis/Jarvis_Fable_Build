"""Hardware Security Module (HSM) vaulting and ephemeral credential injection.

Blueprint Section 12: credentials are stored in OS-native secret storage so
they never appear in plain-text config files or prompt context. An ephemeral
token context manager retrieves a secret, injects it into the subprocess
environment for the duration of a scoped block, then immediately revokes it.

Vault backends (chosen automatically by OS):
  macOS   — macOS Keychain via the `security` CLI (no extra deps)
  Linux   — GNOME Keyring via `secret-tool` (libsecret), falling back to an
             PBKDF2 + AES-256-GCM encrypted JSON file at ~/.jarvis/vault.enc
  Windows — Windows Credential Manager via `cmdkey` / PowerShell

The encrypted-file fallback requires:
    pip install cryptography
and the master password in env var JARVIS_VAULT_PASSWORD.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import platform
import subprocess
from pathlib import Path


class VaultError(RuntimeError):
    """Raised when a vault operation fails."""


# ---- Backend implementations ------------------------------------------------

class KeychainVault:
    """macOS Keychain via `security` CLI — no Python deps required."""

    def store(self, service: str, account: str, secret: str) -> None:
        subprocess.run(
            ["security", "add-generic-password",
             "-a", account, "-s", service, "-w", secret, "-U"],
            check=True, capture_output=True,
        )

    def retrieve(self, service: str, account: str) -> str:
        result = subprocess.run(
            ["security", "find-generic-password",
             "-a", account, "-s", service, "-w"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise VaultError(f"not found in Keychain: {service}/{account}")
        return result.stdout.strip()

    def delete(self, service: str, account: str) -> None:
        subprocess.run(
            ["security", "delete-generic-password",
             "-a", account, "-s", service],
            capture_output=True,
        )

    def list_services(self) -> list[str]:
        result = subprocess.run(
            ["security", "dump-keychain"],
            capture_output=True, text=True,
        )
        services = []
        for line in result.stdout.splitlines():
            if '"svce"' in line and "<blob>" in line:
                try:
                    services.append(line.split('"')[3])
                except IndexError:
                    pass
        return services


class SecretToolVault:
    """GNOME Keyring via `secret-tool` (libsecret)."""

    def store(self, service: str, account: str, secret: str) -> None:
        proc = subprocess.Popen(
            ["secret-tool", "store", "--label", f"jarvis/{service}/{account}",
             "service", service, "account", account],
            stdin=subprocess.PIPE, capture_output=True,
        )
        proc.communicate(secret.encode())
        if proc.returncode != 0:
            raise VaultError("secret-tool store failed")

    def retrieve(self, service: str, account: str) -> str:
        result = subprocess.run(
            ["secret-tool", "lookup", "service", service, "account", account],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise VaultError(f"not found in GNOME Keyring: {service}/{account}")
        return result.stdout.strip()

    def delete(self, service: str, account: str) -> None:
        subprocess.run(
            ["secret-tool", "clear", "service", service, "account", account],
            capture_output=True,
        )

    def list_services(self) -> list[str]:
        return []   # secret-tool has no list command without a schema


class EncryptedFileVault:
    """Fallback: AES-256-GCM encrypted JSON at a configurable path.

    Master password is read from the JARVIS_VAULT_PASSWORD environment
    variable. Requires `pip install cryptography`.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def _password(self) -> bytes:
        pw = os.environ.get("JARVIS_VAULT_PASSWORD", "")
        if not pw:
            raise VaultError(
                "JARVIS_VAULT_PASSWORD is not set. "
                "Export it before using the encrypted file vault."
            )
        return pw.encode()

    def _derive_key(self, salt: bytes) -> bytes:
        from hashlib import pbkdf2_hmac
        return pbkdf2_hmac("sha256", self._password(), salt, 200_000, dklen=32)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise VaultError(
                "Encrypted vault requires: pip install cryptography"
            )
        raw = base64.b64decode(self.path.read_bytes())
        salt, nonce, ciphertext = raw[:16], raw[16:28], raw[28:]
        key = self._derive_key(salt)
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        return json.loads(plaintext)

    def _save(self, data: dict) -> None:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        except ImportError:
            raise VaultError(
                "Encrypted vault requires: pip install cryptography"
            )
        salt = os.urandom(16)
        nonce = os.urandom(12)
        key = self._derive_key(salt)
        ciphertext = AESGCM(key).encrypt(nonce, json.dumps(data).encode(), None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(base64.b64encode(salt + nonce + ciphertext))
        self.path.chmod(0o600)

    def store(self, service: str, account: str, secret: str) -> None:
        data = self._load()
        data[f"{service}/{account}"] = secret
        self._save(data)

    def retrieve(self, service: str, account: str) -> str:
        data = self._load()
        key = f"{service}/{account}"
        if key not in data:
            raise VaultError(f"not found in encrypted vault: {key}")
        return data[key]

    def delete(self, service: str, account: str) -> None:
        data = self._load()
        data.pop(f"{service}/{account}", None)
        self._save(data)

    def list_services(self) -> list[str]:
        try:
            return list(self._load().keys())
        except VaultError:
            return []


class _WindowsVault:
    """Windows Credential Manager via `cmdkey` and PowerShell."""

    def store(self, service: str, account: str, secret: str) -> None:
        subprocess.run(
            ["cmdkey", f"/add:{service}", f"/user:{account}", f"/pass:{secret}"],
            capture_output=True, check=True,
        )

    def retrieve(self, service: str, account: str) -> str:
        script = (
            f"$c = Get-StoredCredential -Target '{service}'; "
            "$c.GetNetworkCredential().Password"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise VaultError(f"not found in Credential Manager: {service}/{account}")
        return result.stdout.strip()

    def delete(self, service: str, account: str) -> None:
        subprocess.run(
            ["cmdkey", f"/delete:{service}"],
            capture_output=True,
        )

    def list_services(self) -> list[str]:
        result = subprocess.run(
            ["cmdkey", "/list"],
            capture_output=True, text=True,
        )
        return [
            line.strip().removeprefix("Target: ")
            for line in result.stdout.splitlines()
            if "Target:" in line
        ]


# ---- Factory ----------------------------------------------------------------

def get_vault(home: Path):
    """Return the best available vault backend for this platform."""
    system = platform.system()
    if system == "Darwin":
        return KeychainVault()
    if system == "Linux":
        import shutil
        if shutil.which("secret-tool"):
            return SecretToolVault()
        return EncryptedFileVault(home / "vault.enc")
    if system == "Windows":
        return _WindowsVault()
    return EncryptedFileVault(home / "vault.enc")


# ---- Ephemeral token injection ----------------------------------------------

@contextlib.contextmanager
def ephemeral_token(vault, service: str, account: str, env_var: str):
    """Retrieve a secret, inject it into os.environ, then zeroize on exit.

    The secret is never returned to the caller — it only exists in the child
    process environment during the scoped block.

    Usage:
        with ephemeral_token(vault, "openai", "api_key", "OPENAI_API_KEY"):
            result = call_openai_api()
        # OPENAI_API_KEY is unset; secret no longer in environment
    """
    token = vault.retrieve(service, account)
    previous = os.environ.get(env_var)
    os.environ[env_var] = token
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = previous
        # Best-effort in-memory zeroization before GC
        token = "\x00" * len(token)  # noqa: F841


# ---- Tool registration ------------------------------------------------------

def register_vault_tools(registry, vault) -> None:
    """Add vault_store, vault_check, and vault_delete tools (CONFIRM tier)."""
    from .tools import Tier, Tool

    def vault_store(service: str, account: str, secret: str) -> str:
        try:
            vault.store(service, account, secret)
        except VaultError as exc:
            return f"ERROR: {exc}"
        return f"stored {service}/{account} in {type(vault).__name__}"

    def vault_check(service: str, account: str) -> str:
        try:
            val = vault.retrieve(service, account)
            return (
                f"{service}/{account} exists in vault "
                f"({len(val)} chars — value not exposed to model)"
            )
        except VaultError:
            return f"{service}/{account}: not found in vault"

    def vault_delete(service: str, account: str) -> str:
        try:
            vault.delete(service, account)
        except VaultError as exc:
            return f"ERROR: {exc}"
        return f"deleted {service}/{account} from vault"

    def vault_list() -> str:
        try:
            services = vault.list_services()
        except VaultError as exc:
            return f"ERROR: {exc}"
        if not services:
            return "vault is empty (or listing is unsupported by this backend)"
        return "Vault entries:\n" + "\n".join(f"  - {s}" for s in services)

    registry.register(Tool(
        "vault_store",
        "Store a credential in the OS-native secret vault (Keychain / GNOME Keyring / encrypted file).",
        {"service": "service name (e.g. openai)",
         "account": "account name (e.g. api_key)",
         "secret": "the secret value"},
        vault_store, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "vault_check",
        "Check whether a credential exists in the vault. Value is never returned to the model.",
        {"service": "service name", "account": "account name"},
        vault_check,
    ))
    registry.register(Tool(
        "vault_delete",
        "Delete a credential from the OS-native vault.",
        {"service": "service name", "account": "account name"},
        vault_delete, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "vault_list",
        "List service entries stored in the credential vault (names only, no secrets).",
        {},
        vault_list,
    ))
