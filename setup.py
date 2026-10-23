"""Pin documentation links in copied package data to the distribution's version."""

import re
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class _BuildPy(build_py):
    """Render installed reference links without changing the source reference."""

    def run(self) -> None:
        """Copy package files and pin documentation links in the built reference."""
        super().run()
        if self.editable_mode:
            return
        version = self.distribution.get_version()
        revision = re.search(r"\+g([0-9a-f]+)(?:\.|$)", version)
        ref = revision[1] if revision else f"v{version}"
        reference = Path(self.build_lib) / "phasesweep" / "config_reference.yaml"
        reference.write_text(
            re.sub(
                r"https://github\.com/pszemraj/phasesweep/blob/[^/]+/",
                f"https://github.com/pszemraj/phasesweep/blob/{ref}/",
                reference.read_text(encoding="utf-8"),
            ),
            encoding="utf-8",
        )


setup(cmdclass={"build_py": _BuildPy})
