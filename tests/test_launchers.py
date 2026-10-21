from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


class WindowsLauncherTests(unittest.TestCase):
    def test_windows_timezone_dependency_is_declared_and_checked_at_startup(self) -> None:
        from zoneinfo import ZoneInfo

        project_root = Path(__file__).resolve().parents[1]
        requirements = (project_root / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(requirements, r"(?m)^tzdata>=")
        startup = (project_root / "start.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("ZoneInfo('Asia/Shanghai')", startup)
        self.assertIn("'install.ps1') -SkipDws", startup)
        self.assertEqual("Asia/Shanghai", ZoneInfo("Asia/Shanghai").key)

    def test_business_launchers_resolve_their_powershell_entrypoints(self) -> None:
        if os.name != "nt":
            self.skipTest("Windows launchers are only executable on Windows")

        project_root = Path(__file__).resolve().parents[1]
        launchers = (
            "01_首次使用.cmd",
            "02_日常启动.cmd",
            "03_重新登录钉钉.cmd",
            "04_演示模式.cmd",
        )
        environment = os.environ.copy()
        environment["DINGTALK_REPORT_LAUNCHER_DRY_RUN"] = "1"

        for launcher_name in launchers:
            launcher = project_root / launcher_name
            self.assertTrue(launcher.is_file(), launcher_name)
            completed = subprocess.run(
                ["cmd.exe", "/d", "/c", str(launcher)],
                cwd=project_root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            self.assertEqual(
                0,
                completed.returncode,
                f"{launcher_name}: {completed.stdout}\n{completed.stderr}",
            )
            self.assertIn("LAUNCHER_OK", completed.stdout)
