"""Operational entrypoints and display terms only; inference profile stays immutable."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "workspaces/m5-cascade-v1"


def main():
    old = ROOT / "scripts/start.ps1"
    legacy = ROOT / "scripts/start_legacy.ps1"
    assert not legacy.exists()
    legacy.write_bytes(old.read_bytes())
    wrapper = """param([int]$Port = 8013, [switch]$EnableDeepSeek, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'start_main.ps1') -Port $Port -EnableDeepSeek:$EnableDeepSeek -CheckOnly:$CheckOnly
"""
    old.write_text(wrapper, encoding="utf-8")
    (ROOT / "scripts/start_m5.ps1").write_text(wrapper, encoding="utf-8")
    p = WS / "backend/cpr_agent/ports.py"
    s = p.read_text(encoding="utf-8")
    assert s.count("轻量RGB识别与预算化采样") == 1
    p.write_text(
        s.replace("轻量RGB识别与预算化采样", "质量门控与特征依赖驱动的轻量化视觉级联"), encoding="utf-8"
    )
    p = WS / "backend/cpr_agent/api.py"
    s = p.read_text(encoding="utf-8")
    marker = '    dist = ROOT / "frontend" / "dist"'
    assert s.count(marker) == 1
    s = s.replace(
        marker,
        "    from .mainline_demo import router as mainline_demo_router\n    app.include_router(mainline_demo_router(runtime, settings))\n\n"
        + marker,
    )
    p.write_text(s, encoding="utf-8")
    print("Unified entrypoints and current M5 title; numerical profile unchanged")


if __name__ == "__main__":
    main()
