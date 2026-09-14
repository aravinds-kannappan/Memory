from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from textwrap import dedent

from memory_agent.config import APP_ID, APP_NAME, base_url, data_dir, host, port


def agent_command() -> list[str]:
    # Prefer installed console script; fall back to python -m
    which = shutil.which("browser-memory")
    if which:
        return [which, "serve", "--open"]
    return [sys.executable, "-m", "memory_agent", "serve", "--open"]


def install_autostart() -> dict:
    system = platform.system()
    if system == "Darwin":
        return _install_mac_launch_agent()
    if system == "Linux":
        return _install_linux_autostart()
    return {"ok": False, "reason": f"unsupported_platform:{system}"}


def uninstall_autostart() -> dict:
    system = platform.system()
    removed = []
    if system == "Darwin":
        plist = Path.home() / "Library/LaunchAgents" / f"{APP_ID}.plist"
        if plist.exists():
            os.system(f'launchctl unload "{plist}" 2>/dev/null')
            plist.unlink()
            removed.append(str(plist))
    else:
        desktop = Path.home() / ".config/autostart/browser-memory.desktop"
        if desktop.exists():
            desktop.unlink()
            removed.append(str(desktop))
    return {"ok": True, "removed": removed}


def autostart_status() -> dict:
    system = platform.system()
    if system == "Darwin":
        plist = Path.home() / "Library/LaunchAgents" / f"{APP_ID}.plist"
        return {"enabled": plist.exists(), "path": str(plist), "platform": system}
    desktop = Path.home() / ".config/autostart/browser-memory.desktop"
    return {"enabled": desktop.exists(), "path": str(desktop), "platform": system}


def _install_mac_launch_agent() -> dict:
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    plist_path = agents / f"{APP_ID}.plist"
    cmd = agent_command()
    program_args = "\n".join(f"    <string>{c}</string>" for c in cmd)
    log = data_dir() / "launchd.out.log"
    err = data_dir() / "launchd.err.log"
    plist = dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
          <key>Label</key>
          <string>{APP_ID}</string>
          <key>ProgramArguments</key>
          <array>
        {program_args}
          </array>
          <key>RunAtLoad</key>
          <true/>
          <key>KeepAlive</key>
          <dict>
            <key>SuccessfulExit</key>
            <false/>
          </dict>
          <key>WorkingDirectory</key>
          <string>{data_dir()}</string>
          <key>StandardOutPath</key>
          <string>{log}</string>
          <key>StandardErrorPath</key>
          <string>{err}</string>
          <key>EnvironmentVariables</key>
          <dict>
            <key>MEMORY_HOST</key>
            <string>{host()}</string>
            <key>MEMORY_PORT</key>
            <string>{port()}</string>
          </dict>
        </dict>
        </plist>
        """
    )
    plist_path.write_text(plist)
    os.system(f'launchctl unload "{plist_path}" 2>/dev/null')
    rc = os.system(f'launchctl load "{plist_path}"')
    return {
        "ok": rc == 0,
        "path": str(plist_path),
        "command": cmd,
        "ui": base_url(),
        "message": f"{APP_NAME} will start when you log in.",
    }


def _install_linux_autostart() -> dict:
    autostart = Path.home() / ".config/autostart"
    autostart.mkdir(parents=True, exist_ok=True)
    desktop_path = autostart / "browser-memory.desktop"
    cmd = agent_command()
    exec_line = " ".join(cmd)
    content = dedent(
        f"""\
        [Desktop Entry]
        Type=Application
        Name={APP_NAME}
        Comment=Remember browsing so you can close tabs and pick up later
        Exec={exec_line}
        X-GNOME-Autostart-enabled=true
        StartupNotify=false
        Terminal=false
        Categories=Utility;
        """
    )
    desktop_path.write_text(content)
    try:
        desktop_path.chmod(0o755)
    except OSError:
        pass
    return {
        "ok": True,
        "path": str(desktop_path),
        "command": cmd,
        "ui": base_url(),
        "message": f"{APP_NAME} will start when you log in.",
    }
