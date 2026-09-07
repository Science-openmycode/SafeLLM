from __future__ import annotations

from aloepri.cloud.ssh import SSHProfile, SSHSession

_REMOTE_CHECKS: tuple[tuple[str, list[str]], ...] = (
    ("architecture", ["uname", "-m"]),
    (
        "tdx_guest_device",
        ["sh", "-lc", "test -e /dev/tdx_guest -o -e /dev/tdx-guest"],
    ),
    (
        "tdx_quote_tool",
        [
            "sh",
            "-lc",
            "command -v tdx-attest >/dev/null 2>&1 || "
            "command -v trustauthority-cli >/dev/null 2>&1",
        ],
    ),
    ("tongsuo", ["sh", "-lc", "command -v tongsuo >/dev/null 2>&1"]),
    (
        "nvidia",
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    ),
    (
        "iommu",
        [
            "sh",
            "-lc",
            "test -d /sys/kernel/iommu_groups && "
            "test -n \"$(ls -A /sys/kernel/iommu_groups 2>/dev/null)\"",
        ],
    ),
    (
        "time_sync",
        [
            "sh",
            "-lc",
            "timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -qx yes",
        ],
    ),
)


async def inspect_remote_tdx(profile: SSHProfile) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    async with SSHSession(profile) as session:
        for name, command in _REMOTE_CHECKS:
            result = await session.run(command, timeout_seconds=30)
            checks[name] = result
        fingerprint = session.fingerprint
    architecture_value = checks["architecture"].get("stdout")
    architecture = architecture_value.strip() if isinstance(architecture_value, str) else ""
    failures = [
        name
        for name, result in checks.items()
        if not isinstance(result.get("exit_code"), int) or result["exit_code"] != 0
    ]
    if architecture != "x86_64":
        failures.append("architecture_x86_64")
    trusted = profile.host_key_fingerprint is not None
    if not trusted:
        failures.append("host_key_unconfirmed")
    return {
        "tee_backend": "intel_tdx",
        "pass": not failures,
        "hardware_attested": False,
        "quote_generated": False,
        "host_key_fingerprint": fingerprint,
        "trusted": trusted,
        "checks": checks,
        "failures": sorted(set(failures)),
        "note": "preflight is not a DCAP quote; deployment must still attest at runtime",
    }
