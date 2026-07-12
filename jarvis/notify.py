"""Native desktop notifications, best-effort, zero dependencies.

Linux: notify-send (libnotify) · macOS: osascript · Windows: PowerShell
balloon tip. Failure is silent — a missing notifier must never break a
reminder, which is also delivered in-band (terminal / web UI / speech).
"""

from __future__ import annotations

import platform
import shutil
import subprocess


def desktop_notify(title: str, body: str) -> bool:
    body = body[:400]
    system = platform.system()
    try:
        if system == "Linux" and shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", "--app-name=Jarvis", title, body],
                check=True, capture_output=True, timeout=10,
            )
            return True
        if system == "Darwin":
            script = (
                f"display notification {body!r} with title {title!r} sound name \"Glass\""
            )
            subprocess.run(
                ["osascript", "-e", script],
                check=True, capture_output=True, timeout=10,
            )
            return True
        if system == "Windows":
            script = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n = New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon = [System.Drawing.SystemIcons]::Information;"
                "$n.Visible = $true;"
                f"$n.ShowBalloonTip(10000, {title!r}, {body!r}, 'Info')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True, capture_output=True, timeout=15,
            )
            return True
    except Exception:
        pass
    return False
