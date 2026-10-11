"""Import-safe configuration for native MLflow tracking."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator, model_validator

from phasesweep.config.common import ConfigFloat, _Frozen


class MlflowSettings(_Frozen):
    """One source and destination experiment for the trainer and PhaseSweep."""

    tracking_uri: str = Field(min_length=1)
    experiment_id: str | None = Field(default=None, min_length=1)
    experiment_name: str | None = Field(default=None, min_length=1)
    run_mode: Literal["managed", "trainer"] = "managed"
    poll_seconds: ConfigFloat = Field(default=2.0, gt=0, allow_inf_nan=False)
    timeout_seconds: ConfigFloat = Field(default=120.0, ge=1, allow_inf_nan=False)

    @field_validator("tracking_uri")
    @classmethod
    def _target(cls, value: str) -> str:
        """Normalize HTTP targets and preserve native Databricks profile URIs.

        :param str value: Tracking URI supplied by the operator.
        :raises ValueError: Unsupported scheme or embedded HTTP credentials.
        :return str: Normalized tracking target.
        """
        if value == "databricks":
            return value
        parsed = urlsplit(value)
        if parsed.scheme == "databricks":
            # MLflow rejects whitespace in profile names at the first request.
            if (
                parsed.netloc
                and not any(c.isspace() for c in parsed.netloc)
                and not any(
                    (parsed.path, parsed.query, parsed.fragment, parsed.username, parsed.password)
                )
            ):
                return value
        elif (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not any((parsed.username, parsed.password, parsed.query, parsed.fragment))
        ):
            return urlunsplit(
                (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")
            )
        raise ValueError(
            "MLflow tracking_uri requires HTTP(S), databricks, or databricks://profile (no spaces in the profile name) without embedded credentials, query, or fragment."
        )

    @model_validator(mode="after")
    def _experiment(self) -> MlflowSettings:
        """Require exactly one existing experiment selector.

        :raises ValueError: Both or neither experiment selectors were supplied.
        :return MlflowSettings: Validated settings.
        """
        if (self.experiment_id is None) == (self.experiment_name is None):
            raise ValueError("MLflow requires exactly one of experiment_id or experiment_name.")
        return self

    def evidence_identity(self) -> dict[str, str | None]:
        """Return fields that determine the scientific source.

        :return dict[str, str | None]: Source identity, excluding polling budgets.
        """
        return {
            "tracking_uri": self.tracking_uri,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "run_mode": self.run_mode,
        }
