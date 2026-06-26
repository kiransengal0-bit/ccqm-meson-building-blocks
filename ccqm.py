#!/usr/bin/env python3
"""
CCQM Stage-1 Building-Block Calculator

User-facing command-line interface for coupling constants, decay constants,
and current-resolved P -> P / P -> V form factors.

Scope: Stage 1 only. This program does not calculate decay widths,
branching fractions, CKM/Wilson coefficient amplitudes, or angular observables.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "2.0.0-stage1-scientific-precision"
SUPPORTED_FINAL_KINDS = {"P", "V"}
SUPPORTED_CURRENTS = {
    "P": {"scalar", "pseudoscalar", "vector", "axial", "v_minus_a", "v_plus_a", "tensor"},
    "V": {"scalar", "pseudoscalar", "vector", "axial", "v_minus_a", "v_plus_a", "tensor_plus", "tensor_minus"},
}
PRECISION_PRESETS = {
    "quick": 4,
    "standard": 8,
    "high": 12,
    "very_high": 16,
    "research": 20,
}


class UserInputError(Exception):
    """Friendly user-facing input error."""


@dataclass
class LineRecord:
    text: str
    line_no: int


def load_backend(path: Path):
    if not path.exists():
        raise UserInputError(f"Backend file not found: {path}")
    spec = importlib.util.spec_from_file_location("ccqm_stage1_backend", str(path))
    if spec is None or spec.loader is None:
        raise UserInputError(f"Could not load backend from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def split_row(line: str) -> list[str]:
    if "," in line:
        return [p.strip() for p in line.split(",") if p.strip()]
    return [p.strip() for p in line.split() if p.strip()]


def parse_bool(x: Any, default: bool | None = None) -> bool:
    if x is None:
        if default is None:
            raise UserInputError("Boolean value is missing.")
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s == "" and default is not None:
        return default
    if s in ("true", "t", "yes", "y", "1", "on"):
        return True
    if s in ("false", "f", "no", "n", "0", "off"):
        return False
    raise UserInputError(f"Cannot parse boolean value: {x!r}. Use yes/no or true/false.")


def maybe_number(x: str) -> Any:
    s = str(x).strip()
    if s.lower() in ("true", "false", "yes", "no", "on", "off"):
        return parse_bool(s)
    if s.lower() in ("q2max", "qmax", "max", "physical", "physical_max"):
        return "q2max"
    try:
        if re.fullmatch(r"[+-]?\d+", s):
            return int(s)
        return float(s)
    except ValueError:
        return s


def parse_key_value(line: str, *, line_no: int | None = None) -> tuple[str, Any]:
    if "=" not in line:
        loc = f" on line {line_no}" if line_no else ""
        raise UserInputError(f"Expected 'key = value'{loc}, got: {line}")
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        loc = f" on line {line_no}" if line_no else ""
        raise UserInputError(f"Missing key before '='{loc}.")
    return key, maybe_number(value.strip())


def parse_text_input(path: Path) -> dict[str, Any]:
    """Parse the human-readable Stage-1 text file."""
    data: dict[str, Any] = {
        "global": {},
        "mesons": [],
        "decay_constants": [],
        "transitions_by_current": [],
    }
    section: str | None = None
    current_transition: dict[str, Any] | None = None

    def finish_transition():
        nonlocal current_transition
        if current_transition is not None:
            if "name" not in current_transition:
                current_transition["name"] = f"transition_{len(data['transitions_by_current'])+1}"
            data["transitions_by_current"].append(current_transition)
            current_transition = None

    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines, start=1):
        line = strip_comment(raw)
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            finish_transition()
            sec = line[1:-1].strip()
            sec_low = sec.lower()
            if sec_low.startswith("transition"):
                parts = sec.split(maxsplit=1)
                section = "transition"
                current_transition = {"_line": i}
                if len(parts) > 1 and parts[1].strip():
                    current_transition["name"] = parts[1].strip()
            elif sec_low in {"global", "mesons", "decay_constants"}:
                section = sec_low
            else:
                raise UserInputError(f"Unknown section [{sec}] on line {i}. Supported: [global], [mesons], [decay_constants], [transition NAME].")
            continue

        if section is None:
            raise UserInputError(f"Line {i} is outside any section: {line}")

        if section == "global":
            key, value = parse_key_value(line, line_no=i)
            data["global"][key] = value

        elif section == "mesons":
            parts = split_row(line)
            if not parts or parts[0].lower() in ("name", "meson"):
                continue
            if len(parts) < 6:
                raise UserInputError(
                    f"Bad meson row on line {i}. Required columns: name kind M m1 m2 Lambda [optional g]. Got: {line}"
                )
            try:
                meson: dict[str, Any] = {
                    "name": parts[0],
                    "kind": parts[1].upper(),
                    "M": float(parts[2]),
                    "m1": float(parts[3]),
                    "m2": float(parts[4]),
                    "Lambda": float(parts[5]),
                    "_line": i,
                }
                extras = parts[6:]
                if extras and "=" not in extras[0]:
                    meson["g"] = float(extras[0])
                    extras = extras[1:]
                for item in extras:
                    if "=" in item:
                        key, value = parse_key_value(item, line_no=i)
                        meson[key] = value
            except ValueError as exc:
                raise UserInputError(f"Could not parse numeric meson data on line {i}: {line}") from exc
            data["mesons"].append(meson)

        elif section == "decay_constants":
            for token in split_row(line):
                if token.lower() not in ("name", "meson"):
                    data["decay_constants"].append(token)

        elif section == "transition":
            if current_transition is None:
                current_transition = {"_line": i}
            key, value = parse_key_value(line, line_no=i)
            current_transition[key] = value

    finish_transition()
    return data


def is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    return str(x).strip() == ""


def parse_currents(x: Any) -> list[str] | None:
    if x is None:
        return None
    if isinstance(x, list):
        out = [str(v).strip() for v in x if str(v).strip()]
    else:
        s = str(x).replace(";", ",").replace("\n", ",")
        out = [p.strip() for p in s.split(",") if p.strip()]
    return out or None


def validate_input(data: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    mesons = data.get("mesons", [])
    if not mesons:
        raise UserInputError("Input must contain a non-empty [mesons] section.")

    names: set[str] = set()
    for m in mesons:
        name = m.get("name")
        if not name:
            raise UserInputError("Every meson row must have a name.")
        if name in names:
            raise UserInputError(f"Duplicate meson name: {name}")
        names.add(name)
        kind = str(m.get("kind", "")).upper()
        if kind not in {"P", "V"}:
            raise UserInputError(f"Meson {name}: kind must be P or V, got {kind!r}.")
        for key in ("M", "m1", "m2", "Lambda"):
            if float(m[key]) <= 0:
                raise UserInputError(f"Meson {name}: {key} must be positive.")

    for req in data.get("decay_constants", []):
        if isinstance(req, str):
            if req not in names:
                raise UserInputError(f"Decay constant requested for unknown meson: {req}")

    for t in data.get("transitions_by_current", []):
        tname = t.get("name", "unnamed_transition")
        for key in ("initial", "final", "m1", "m2", "m3"):
            if key not in t:
                raise UserInputError(f"Transition {tname}: missing required field '{key}'.")
        if t["initial"] not in names:
            raise UserInputError(f"Transition {tname}: unknown initial meson {t['initial']!r}.")
        if t["final"] not in names:
            raise UserInputError(f"Transition {tname}: unknown final meson {t['final']!r}.")
        final_kind = str(t.get("final_kind", "")).upper() or next(m["kind"] for m in mesons if m["name"] == t["final"])
        if final_kind not in SUPPORTED_FINAL_KINDS:
            raise UserInputError(f"Transition {tname}: final_kind must be P or V.")
        currents = parse_currents(t.get("currents"))
        if not currents:
            raise UserInputError(f"Transition {tname}: currents field is required, for example currents = vector, tensor")
        invalid = [c for c in currents if c not in SUPPORTED_CURRENTS[final_kind]]
        if invalid:
            raise UserInputError(
                f"Transition {tname}: invalid current(s) for final_kind={final_kind}: {', '.join(invalid)}. "
                f"Allowed: {', '.join(sorted(SUPPORTED_CURRENTS[final_kind]))}."
            )
        q2_mode = str(t.get("q2_mode", "list")).lower()
        if q2_mode not in {"list", "range"}:
            raise UserInputError(f"Transition {tname}: q2_mode must be 'list' or 'range'.")
        if q2_mode == "list" and "q2_values" not in t:
            raise UserInputError(f"Transition {tname}: q2_mode=list requires q2_values.")
        if q2_mode == "range":
            for key in ("q2_min", "q2_max", "q2_points"):
                if key not in t:
                    raise UserInputError(f"Transition {tname}: q2_mode=range requires {key}.")
            if int(t.get("q2_points")) < 1:
                raise UserInputError(f"Transition {tname}: q2_points must be >= 1.")
        if final_kind == "V" and currents and any(c.startswith("tensor") for c in currents):
            q2_has_zero = False
            if q2_mode == "list":
                for tok in re.split(r"[,;\s]+", str(t.get("q2_values", ""))):
                    if tok.strip():
                        try:
                            if abs(float(tok)) < 1e-14:
                                q2_has_zero = True
                        except ValueError:
                            pass
            elif q2_mode == "range":
                try:
                    q2_has_zero = abs(float(t.get("q2_min"))) < 1e-14 and parse_bool(t.get("endpoint"), default=True)
                except Exception:
                    q2_has_zero = False
            if q2_has_zero:
                warnings.append(f"Transition {tname}: P->V tensor form factors are singular at q2=0; use q2_min > 0 or extrapolate.")
    return warnings


def apply_precision_preset(data: dict[str, Any], preset: str | None) -> None:
    if preset is None:
        return
    if preset not in PRECISION_PRESETS:
        raise UserInputError(f"Unknown precision preset {preset!r}. Use one of: {', '.join(PRECISION_PRESETS)}")
    data.setdefault("global", {})["n_quad"] = PRECISION_PRESETS[preset]


def build_results(ccqm, data: dict[str, Any], *, progress: bool = True) -> dict[str, Any]:
    glob = data.get("global", {})
    lambda_ir = float(glob.get("lambda_ir", 0.181))
    n_quad = int(glob.get("n_quad", 12))
    Nc = int(glob.get("Nc", 3))

    mesons_in = {m["name"]: dict(m) for m in data.get("mesons", [])}

    results: dict[str, Any] = {
        "program": {"name": "CCQM Stage-1 Building-Block Calculator", "version": VERSION},
        "scope": "stage_1_building_blocks_only",
        "global": {"lambda_ir": lambda_ir, "Nc": Nc, "n_quad_default": n_quad},
        "couplings": {},
        "decay_constants": {},
        "transitions_by_current": {},
        "warnings": [],
        "notes": [
            "Stage 1 only: coupling constants, decay constants, and current-resolved form factors.",
            "No leptonic, semileptonic, nonleptonic, rare decay widths, branching ratios, CKM factors, or Wilson coefficients are calculated.",
            "Decay constants f_H are independent of q2; q2-dependent plots are form-factor plots.",
        ],
    }

    if progress:
        print(f"Using n_quad={n_quad}, lambda_ir={lambda_ir}, Nc={Nc}")

    # Couplings g_H
    if progress:
        print("Step 1/3: coupling constants")
    for name, m in mesons_in.items():
        if "g" in m and not is_blank(m["g"]):
            results["couplings"][name] = {"name": name, "kind": m["kind"], "g": float(m["g"]), "source": "input"}
        else:
            mc = ccqm.MesonConfig(
                name=name,
                kind=m["kind"],
                M=float(m["M"]),
                m1=float(m["m1"]),
                m2=float(m["m2"]),
                Lambda=float(m["Lambda"]),
                lambda_ir=float(m.get("lambda_ir", lambda_ir)),
                Nc=int(m.get("Nc", Nc)),
                n_quad=int(m.get("n_quad", n_quad)),
            )
            results["couplings"][name] = ccqm.compute_coupling_constant(mc, derivative_step=float(m.get("derivative_step", 1e-4)))

    # Decay constants f_H
    if progress:
        print("Step 2/3: decay constants")
    for req in data.get("decay_constants", []):
        if isinstance(req, str):
            out_name = req
            meson_name = req
        else:
            meson_name = req.get("meson", req.get("name"))
            out_name = req.get("name", meson_name)
        m = mesons_in[meson_name]
        dc = ccqm.DecayConstantConfig(
            kind=m["kind"],
            M=float(m["M"]),
            m1=float(m["m1"]),
            m2=float(m["m2"]),
            Lambda=float(m["Lambda"]),
            g=float(results["couplings"][meson_name]["g"]),
            lambda_ir=float(m.get("lambda_ir", lambda_ir)),
            Nc=int(m.get("Nc", Nc)),
            n_quad=int(m.get("n_quad", n_quad)),
        )
        results["decay_constants"][out_name] = ccqm.compute_decay_constant(dc)

    # Current-resolved form factors
    transitions = data.get("transitions_by_current", data.get("transitions", []))
    if progress:
        print(f"Step 3/3: form factors for {len(transitions)} transition(s)")
    for i, t in enumerate(transitions, start=1):
        tname = t.get("name", f"transition_{i}")
        if progress:
            print(f"  [{i}/{len(transitions)}] {tname}")
        initial_name = t.get("initial")
        final_name = t.get("final")
        initial = mesons_in[initial_name]
        final = mesons_in[final_name]
        final_kind = str(t.get("final_kind", final.get("kind"))).strip().upper()
        cfg = ccqm.CCQMConfig(
            Mi=float(t.get("Mi", initial["M"])),
            Mf=float(t.get("Mf", final["M"])),
            m1=float(t["m1"]),
            m2=float(t["m2"]),
            m3=float(t["m3"]),
            Lambda_i=float(t.get("Lambda_i", initial["Lambda"])),
            Lambda_f=float(t.get("Lambda_f", final["Lambda"])),
            g_i=float(t.get("g_i", results["couplings"][initial_name]["g"])),
            g_f=float(t.get("g_f", results["couplings"][final_name]["g"])),
            lambda_ir=float(t.get("lambda_ir", lambda_ir)),
            Nc=int(t.get("Nc", Nc)),
            n_quad=int(t.get("n_quad", n_quad)),
        )
        try:
            grid_result = ccqm.compute_form_factors_by_current_grid(
                cfg,
                final_kind,
                currents=parse_currents(t.get("currents")),
                q2_mode=t.get("q2_mode", "list"),
                q2_values=t.get("q2_values"),
                q2_min=t.get("q2_min"),
                q2_max=t.get("q2_max"),
                q2_points=int(t.get("q2_points", 1)),
                endpoint=parse_bool(t.get("endpoint"), default=True),
                validate_physical=parse_bool(t.get("validate_physical"), default=True),
                include_raw_amplitudes=parse_bool(t.get("include_raw_amplitudes"), default=False),
                pv_tensor_i_prefactor=parse_bool(t.get("pv_tensor_i_prefactor"), default=True),
                va_current_factor=float(t.get("va_current_factor", 1.0)),
            )
        except Exception as exc:
            # Keep other transitions usable and produce a clear report.
            grid_result = {"error": str(exc), "q2_results": {}, "q2_grid": []}
            results["warnings"].append(f"Transition {tname} failed: {exc}")
        grid_result.update({
            "initial": initial_name,
            "final": final_name,
            "final_kind": final_kind,
            "masses": {"Mi": cfg.Mi, "Mf": cfg.Mf, "m1": cfg.m1, "m2": cfg.m2, "m3": cfg.m3},
            "lambdas": {"Lambda_i": cfg.Lambda_i, "Lambda_f": cfg.Lambda_f},
            "couplings": {"g_i": cfg.g_i, "g_f": cfg.g_f},
            "currents_requested": parse_currents(t.get("currents")),
        })
        results["transitions_by_current"][tname] = grid_result

    return ccqm._jsonable(results)


def numeric_value(value: Any, *, complex_policy: str = "real") -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, dict) and "re" in value and "im" in value:
        re_val = float(value["re"])
        im_val = float(value["im"])
        if complex_policy == "abs":
            return math.hypot(re_val, im_val)
        if abs(im_val) <= 1e-10 * max(1.0, abs(re_val)):
            return re_val
        if complex_policy == "real":
            return re_val
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def fmt_value(value: Any, digits: int = 8) -> str:
    if value == "" or value is None:
        return "-"
    if isinstance(value, dict) and "re" in value and "im" in value:
        return f"{float(value['re']):.{digits}g} {float(value['im']):+.{digits}g}j"
    val = numeric_value(value)
    if val is not None:
        return f"{val:.{digits}g}"
    return str(value)


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") or "item"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def flatten_form_factors(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tname, tres in results.get("transitions_by_current", {}).items():
        for q2_str, qres in tres.get("q2_results", {}).items():
            for current, cres in qres.get("by_current", {}).items():
                ffs = cres.get("form_factors", {})
                if not ffs and (cres.get("warning") or cres.get("error")):
                    rows.append({"transition": tname, "q2": q2_str, "current": current, "quantity": "warning_or_error", "value": cres.get("warning", cres.get("error", ""))})
                for quantity, value in ffs.items():
                    rows.append({"transition": tname, "q2": q2_str, "current": current, "quantity": quantity, "value": fmt_value(value, digits=12)})
    return rows


def coupling_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, c in results.get("couplings", {}).items():
        rows.append({"meson": name, "kind": c.get("kind", ""), "g_H": fmt_value(c.get("g"), 12), "Z_check": fmt_value(c.get("Z_check", ""), 12), "source": c.get("source", "computed")})
    return rows


def decay_constant_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, d in results.get("decay_constants", {}).items():
        rows.append({"meson": name, "kind": d.get("kind", ""), "f_GeV": fmt_value(d.get("f_GeV", d.get("f")), 12), "f_MeV": fmt_value(d.get("f_MeV"), 12)})
    return rows


def warning_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for w in results.get("warnings", []):
        rows.append({"location": "global", "message": w})
    for tname, tres in results.get("transitions_by_current", {}).items():
        if tres.get("error"):
            rows.append({"location": f"transition:{tname}", "message": tres["error"]})
        for q2_str, qres in tres.get("q2_results", {}).items():
            for current, cres in qres.get("by_current", {}).items():
                if cres.get("warning"):
                    rows.append({"location": f"{tname}/q2={q2_str}/{current}", "message": cres["warning"]})
                if cres.get("error"):
                    rows.append({"location": f"{tname}/q2={q2_str}/{current}", "message": cres["error"]})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["message"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def transition_q2_values(tres: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    for key in ("q2_grid", "q2_values", "q2_values_used"):
        raw = tres.get(key)
        if isinstance(raw, list):
            for item in raw:
                val = numeric_value(item)
                if val is not None:
                    vals.append(val)
    if not vals:
        for q2_str in tres.get("q2_results", {}).keys():
            val = numeric_value(q2_str)
            if val is not None:
                vals.append(val)
    return sorted(set(vals))


def collect_series(results: dict[str, Any]) -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    series: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for tname, tres in results.get("transitions_by_current", {}).items():
        for q2_str, qres in tres.get("q2_results", {}).items():
            q2 = numeric_value(q2_str)
            if q2 is None:
                continue
            for current, cres in qres.get("by_current", {}).items():
                for quantity, value in cres.get("form_factors", {}).items():
                    y = numeric_value(value)
                    if y is None:
                        continue
                    series.setdefault((tname, current, quantity), []).append((q2, y))
    for key, pts in list(series.items()):
        dedup: dict[float, float] = {}
        for q2, y in pts:
            dedup[q2] = y
        series[key] = sorted(dedup.items())
    return series



def collect_trend_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Return form-factor rows sorted for trend inspection.

    This is intentionally different from the compact report table: here the
    primary grouping is transition/current/form_factor, then q2.  Decay
    constants are not included because they are q2-independent.
    """
    rows: list[dict[str, Any]] = []
    for (tname, current, quantity), pts in collect_series(results).items():
        for q2, y in pts:
            rows.append({
                "transition": tname,
                "current": current,
                "form_factor": quantity,
                "q2_GeV2": fmt_value(q2, digits=12),
                "value": fmt_value(y, digits=12),
            })
    rows.sort(key=lambda r: (r["transition"], r["current"], r["form_factor"], float(r["q2_GeV2"])))
    return rows


def collect_trend_groups(results: dict[str, Any]) -> dict[tuple[str, str, str], list[tuple[float, float]]]:
    """Collect trend data by (transition, current, form_factor)."""
    return dict(sorted(collect_series(results).items(), key=lambda item: item[0]))


def write_trend_wide_csv(path: Path, results: dict[str, Any]) -> None:
    """Write a wide matrix: one form factor per row, q2 values as columns.

    This is convenient for users who want to copy a row into their own plotting
    program, spreadsheet, Mathematica, Origin, gnuplot, etc.
    """
    groups = collect_trend_groups(results)
    all_q2: list[float] = sorted({q2 for pts in groups.values() for q2, _ in pts})
    q2_cols = [f"q2={fmt_value(q2, digits=12)}" for q2 in all_q2]
    rows: list[dict[str, Any]] = []
    for (tname, current, quantity), pts in groups.items():
        by_q2 = {q2: y for q2, y in pts}
        row: dict[str, Any] = {"transition": tname, "current": current, "form_factor": quantity}
        for q2, col in zip(all_q2, q2_cols):
            row[col] = fmt_value(by_q2[q2], digits=12) if q2 in by_q2 else ""
        rows.append(row)
    write_csv(path, rows, ["transition", "current", "form_factor"] + q2_cols)


def write_trend_text_report(path: Path, results: dict[str, Any]) -> None:
    """Write a trend-oriented text report.

    Layout:
        Transition
          Current
            Form factor
                q2    value
    """
    lines: list[str] = []
    lines.append("CCQM FORM-FACTOR TREND REPORT")
    lines.append("=" * 88)
    lines.append("")
    lines.append("This report is for q²-dependent form factors only.")
    lines.append("Decay constants f_H are q²-independent and are intentionally not repeated here.")
    lines.append("Use this report to inspect trends or copy values for your own plots.")
    lines.append("")

    groups = collect_trend_groups(results)
    if not groups:
        lines.append("No q²-dependent form-factor data found.")
        ensure_dir(path.parent)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    last_transition: str | None = None
    last_current: str | None = None
    for (tname, current, quantity), pts in groups.items():
        if tname != last_transition:
            lines.append("")
            lines.append(f"Transition: {tname}")
            lines.append("-" * 88)
            last_transition = tname
            last_current = None
        if current != last_current:
            lines.append("")
            lines.append(f"  Current: {current}")
            last_current = current
        lines.append(f"\n    {quantity}")
        rows = [[fmt_value(q2, digits=12), fmt_value(y, digits=12)] for q2, y in pts]
        lines.extend("    " + line for line in text_table(["q² [GeV²]", "value"], rows, widths=[18, 24]))

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trend_html_report(path: Path, results: dict[str, Any]) -> None:
    groups = collect_trend_groups(results)
    sections: list[str] = []
    if not groups:
        sections.append("<p>No q²-dependent form-factor data found.</p>")
    else:
        current_transition: str | None = None
        current_current: str | None = None
        for (tname, current, quantity), pts in groups.items():
            if tname != current_transition:
                sections.append(f"<h2>Transition: {html.escape(tname)}</h2>")
                current_transition = tname
                current_current = None
            if current != current_current:
                sections.append(f"<h3>Current: {html.escape(current)}</h3>")
                current_current = current
            rows = [[fmt_value(q2, digits=12), fmt_value(y, digits=12)] for q2, y in pts]
            sections.append(f"<h4>{html.escape(quantity)}</h4>")
            sections.append(html_table(["q² [GeV²]", "value"], rows))

    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>CCQM Form-Factor Trend Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #222; }}
h1 {{ margin-bottom: 0; }} .subtitle {{ color: #555; margin-top: 6px; }}
table {{ border-collapse: collapse; margin: 10px 0 28px 0; min-width: 420px; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: left; }}
th {{ background: #f5f5f5; }} h2 {{ margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
h3 {{ margin-top: 22px; }} h4 {{ margin-bottom: 4px; }}
.note {{ background: #f7f9fc; padding: 12px 16px; border-left: 4px solid #789; }}
</style></head><body>
<h1>CCQM Form-Factor Trend Report</h1>
<p class='subtitle'>Grouped as transition → current → form factor → q² values.</p>
<div class='note'>Decay constants f_H are q²-independent and are intentionally not repeated here.</div>
{''.join(sections)}
</body></html>"""
    ensure_dir(path.parent)
    path.write_text(html_text, encoding="utf-8")


def write_plots(results: dict[str, Any], plots_dir: Path, *, fmt: str = "png") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(plots_dir)
    written: list[Path] = []

    dc = results.get("decay_constants", {})
    if dc:
        names: list[str] = []
        values: list[float] = []
        for name, d in dc.items():
            val = numeric_value(d.get("f_MeV"))
            if val is not None:
                names.append(str(name))
                values.append(val)
        if names:
            plt.figure(figsize=(max(6, 0.7 * len(names)), 4.2))
            plt.bar(names, values)
            plt.ylabel(r"$f_H$ [MeV]")
            plt.title("Decay constants")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            out = plots_dir / f"decay_constants_bar.{fmt}"
            plt.savefig(out, dpi=180)
            plt.close()
            written.append(out)

    for (tname, current, quantity), pts in collect_series(results).items():
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.figure(figsize=(6.4, 4.2))
        plt.plot(xs, ys, marker="o")
        plt.xlabel(r"$q^2$ [GeV$^2$]")
        plt.ylabel(quantity)
        plt.title(f"{tname}: {current} / {quantity}")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        out = plots_dir / f"ff_{safe_name(tname)}__{safe_name(current)}__{safe_name(quantity)}.{fmt}"
        plt.savefig(out, dpi=180)
        plt.close()
        written.append(out)

    # Decay constants as q2-independent reference lines.
    for tname, tres in results.get("transitions_by_current", {}).items():
        q2_grid = transition_q2_values(tres)
        if len(q2_grid) < 2:
            continue
        refs: list[tuple[str, float]] = []
        for name in (tres.get("initial"), tres.get("final")):
            if name in dc:
                val = numeric_value(dc[name].get("f_MeV"))
                if val is not None:
                    refs.append((str(name), val))
        if not refs:
            continue
        xmin, xmax = min(q2_grid), max(q2_grid)
        plt.figure(figsize=(6.4, 4.2))
        for name, val in refs:
            plt.plot([xmin, xmax], [val, val], marker="o", label=f"{name}: f_H")
        plt.xlabel(r"$q^2$ [GeV$^2$]")
        plt.ylabel(r"$f_H$ [MeV]")
        plt.title(f"{tname}: decay constants are q²-independent")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()
        out = plots_dir / f"decay_constants_vs_q2_reference__{safe_name(tname)}.{fmt}"
        plt.savefig(out, dpi=180)
        plt.close()
        written.append(out)

    return written


def text_table(headers: list[str], rows: list[list[Any]], widths: list[int] | None = None) -> list[str]:
    if widths is None:
        widths = []
        for i, h in enumerate(headers):
            max_len = len(str(h))
            for row in rows:
                max_len = max(max_len, len(str(row[i])))
            widths.append(min(max(max_len, 10), 24))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    for row in rows:
        out.append(fmt.format(*[str(v)[:w] for v, w in zip(row, widths)]))
    return out


def write_text_report(path: Path, results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("CCQM STAGE-1 BUILDING-BLOCK REPORT")
    lines.append("=" * 88)
    lines.append("")
    lines.append("Scope: coupling constants, decay constants, and current-resolved form factors only.")
    lines.append("Not included: decay widths, branching ratios, CKM factors, Wilson coefficients, angular observables.")
    lines.append("")

    warns = warning_rows(results)
    if warns:
        lines.append("WARNINGS")
        lines.append("-" * 88)
        for w in warns:
            lines.append(f"[{w['location']}] {w['message']}")
        lines.append("")

    lines.append("1. COUPLING CONSTANTS")
    lines.append("-" * 88)
    rows = [[r["meson"], r["kind"], r["g_H"], r["Z_check"], r["source"]] for r in coupling_rows(results)]
    lines.extend(text_table(["Meson", "Kind", "g_H", "Z_check", "Source"], rows))
    lines.append("")

    conv = convergence_rows(results)
    diag_rows = diagnostic_rows(results)
    if conv:
        lines.append("NUMERICAL CONVERGENCE SUMMARY")
        lines.append("-" * 88)
        levels = results.get("convergence", {}).get("levels", [])
        lines.append("Quadrature levels checked: " + ", ".join(str(x) for x in levels))
        status_count = {}
        for r in conv:
            status_count[r.get("status", "unknown")] = status_count.get(r.get("status", "unknown"), 0) + 1
        lines.append("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_count.items())))
        lines.append("See tables/convergence.csv for per-form-factor estimates.")
        lines.append("")

    if diag_rows:
        lines.append("PROJECTION DIAGNOSTICS SUMMARY")
        lines.append("-" * 88)
        status_count = {}
        for r in diag_rows:
            status_count[r.get("projection_status", "unknown")] = status_count.get(r.get("projection_status", "unknown"), 0) + 1
        lines.append("Projection status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(status_count.items())))
        lines.append("See tables/diagnostics.csv for condition numbers and endpoint flags.")
        lines.append("")

    lines.append("2. DECAY CONSTANTS")
    lines.append("-" * 88)
    lines.append("Decay constants f_H are q²-independent.")
    rows = [[r["meson"], r["kind"], r["f_GeV"], r["f_MeV"]] for r in decay_constant_rows(results)]
    lines.extend(text_table(["Meson", "Kind", "f [GeV]", "f [MeV]"], rows))
    lines.append("")

    lines.append("3. CURRENT-RESOLVED FORM FACTORS")
    lines.append("-" * 88)
    for tname, tres in results.get("transitions_by_current", {}).items():
        lines.append("")
        lines.append(f"Transition: {tname}")
        lines.append(f"  Initial: {tres.get('initial')}    Final: {tres.get('final')}    final_kind: {tres.get('final_kind')}")
        masses = tres.get("masses", {})
        lines.append("  Masses: " + ", ".join(f"{k}={fmt_value(v)}" for k, v in masses.items()))
        lines.append("  Couplings: " + ", ".join(f"{k}={fmt_value(v)}" for k, v in tres.get("couplings", {}).items()))
        lines.append("  q² grid: " + ", ".join(fmt_value(q) for q in transition_q2_values(tres)))
        if tres.get("error"):
            lines.append(f"  ERROR: {tres['error']}")
            continue
        for q2_str, qres in tres.get("q2_results", {}).items():
            lines.append(f"\n  q² = {q2_str}")
            ff_rows: list[list[str]] = []
            for current, cres in qres.get("by_current", {}).items():
                if cres.get("warning"):
                    ff_rows.append([current, "WARNING", cres["warning"]])
                if cres.get("error"):
                    ff_rows.append([current, "ERROR", cres["error"]])
                for quantity, value in cres.get("form_factors", {}).items():
                    ff_rows.append([current, quantity, fmt_value(value, digits=12)])
            if ff_rows:
                lines.extend(text_table(["Current", "Quantity", "Value"], ff_rows, widths=[16, 20, 28]))
            else:
                lines.append("    No form-factor rows produced.")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in row) + "</tr>")
    return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html_report(path: Path, results: dict[str, Any], plots_dir: Path | None = None) -> None:
    coupling = [[r["meson"], r["kind"], r["g_H"], r["Z_check"], r["source"]] for r in coupling_rows(results)]
    decay = [[r["meson"], r["kind"], r["f_GeV"], r["f_MeV"]] for r in decay_constant_rows(results)]
    ff = flatten_form_factors(results)
    ff_rows = [[r["transition"], r["q2"], r["current"], r["quantity"], r["value"]] for r in ff[:1000]]
    warnings = [[r["location"], r["message"]] for r in warning_rows(results)]
    diag = diagnostic_rows(results)[:200]
    diag_html_rows = [[r["transition"], r["q2"], r["current"], r["projection_condition_number"], r["projection_status"]] for r in diag]
    conv = convergence_rows(results)[:200]
    conv_html_rows = [[r["transition"], r["q2"], r["current"], r["quantity"], r["rel_diff"], r["status"]] for r in conv]
    plot_html = ""
    if plots_dir and plots_dir.exists():
        imgs = sorted(plots_dir.glob("*.png"))[:80]
        if imgs:
            parts = []
            for img in imgs:
                rel = img.relative_to(path.parent) if img.is_relative_to(path.parent) else img
                parts.append(f"<figure><img src='{html.escape(str(rel))}' alt='{html.escape(img.name)}'><figcaption>{html.escape(img.name)}</figcaption></figure>")
            plot_html = "<h2>Plots</h2><div class='plots'>" + "".join(parts) + "</div>"
    warn_html = f"<h2>Warnings</h2>{html_table(['Location','Message'], warnings)}" if warnings else ""
    ff_note = "" if len(ff) <= 1000 else f"<p>Showing first 1000 of {len(ff)} form-factor rows. See tables/form_factors.csv for all rows.</p>"
    html_text = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>CCQM Stage-1 Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.45; color: #222; }}
h1 {{ margin-bottom: 0; }} .subtitle {{ color: #555; margin-top: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px 0; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 7px 9px; text-align: left; }}
th {{ background: #f5f5f5; }} code {{ background: #f5f5f5; padding: 1px 4px; }}
.plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 22px; }}
figure {{ margin: 0; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }}
img {{ max-width: 100%; height: auto; }} figcaption {{ font-size: 12px; color: #666; word-break: break-all; }}
.note {{ background: #f7f9fc; padding: 12px 16px; border-left: 4px solid #789; }}
</style></head><body>
<h1>CCQM Stage-1 Building-Block Report</h1>
<p class='subtitle'>Coupling constants, decay constants, and current-resolved form factors only.</p>
<div class='note'><b>Not included:</b> decay widths, branching ratios, CKM factors, Wilson coefficients, or angular observables.</div>
{warn_html}
<h2>Coupling constants</h2>{html_table(['Meson','Kind','g_H','Z_check','Source'], coupling)}
<h2>Decay constants</h2><p>Decay constants are independent of q².</p>{html_table(['Meson','Kind','f [GeV]','f [MeV]'], decay)}
<h2>Projection diagnostics</h2><p>Showing up to 200 rows. See <code>tables/diagnostics.csv</code> for all condition numbers.</p>{html_table(['Transition','q²','Current','Condition','Status'], diag_html_rows)}
<h2>Convergence diagnostics</h2><p>Showing up to 200 rows. See <code>tables/convergence.csv</code> for all estimates.</p>{html_table(['Transition','q²','Current','Quantity','Rel. diff','Status'], conv_html_rows)}
<h2>Form factors</h2>{ff_note}{html_table(['Transition','q²','Current','Quantity','Value'], ff_rows)}
{plot_html}
</body></html>"""
    ensure_dir(path.parent)
    path.write_text(html_text, encoding="utf-8")




def _form_factor_value_map(results: dict[str, Any]) -> dict[tuple[str, str, str, str], float]:
    """Map (transition, q2, current, quantity) -> numeric value for convergence checks."""
    mp: dict[tuple[str, str, str, str], float] = {}
    for tname, tres in results.get("transitions_by_current", {}).items():
        for q2_str, qres in tres.get("q2_results", {}).items():
            q2_num = numeric_value(q2_str)
            q2_key = f"{q2_num:.12g}" if q2_num is not None else str(q2_str)
            for current, cres in qres.get("by_current", {}).items():
                for quantity, value in cres.get("form_factors", {}).items():
                    val = numeric_value(value)
                    if val is not None:
                        mp[(str(tname), q2_key, str(current), str(quantity))] = val
    return mp


def resolve_convergence_levels(data: dict[str, Any], mode: str) -> list[int]:
    """Return quadrature orders used for convergence diagnostics."""
    if mode in ("off", "none", "false", "0"):
        return []
    n = int(data.get("global", {}).get("n_quad", 12))
    if mode == "basic":
        if n <= 4:
            return [4, 8]
        return sorted(set([max(4, n - 4), n]))
    if mode == "strong":
        return sorted(set([max(4, n - 8), max(4, n - 4), n]))
    if mode == "research":
        return sorted(set([max(4, n - 12), max(4, n - 8), max(4, n - 4), n, n + 4]))
    # explicit comma separated string, e.g. "8,12,16"
    try:
        levels = sorted(set(int(x.strip()) for x in str(mode).split(",") if x.strip()))
    except Exception as exc:
        raise UserInputError("Bad convergence mode. Use off, basic, strong, research, or comma-separated levels like 8,12,16.") from exc
    if any(v < 2 for v in levels):
        raise UserInputError("Convergence n_quad levels must be >= 2.")
    return levels


def compute_convergence(ccqm, data: dict[str, Any], levels: list[int], *, progress: bool = True) -> dict[str, Any]:
    """Run the same input at several n_quad values and compare form factors."""
    if not levels:
        return {"enabled": False, "levels": [], "rows": []}
    results_by_level: dict[int, dict[str, Any]] = {}
    maps: dict[int, dict[tuple[str, str, str, str], float]] = {}
    for n in levels:
        if progress:
            print(f"  convergence check: n_quad={n}")
        d = copy.deepcopy(data)
        d.setdefault("global", {})["n_quad"] = n
        r = build_results(ccqm, d, progress=False)
        results_by_level[n] = r
        maps[n] = _form_factor_value_map(r)
    rows: list[dict[str, Any]] = []
    if len(levels) >= 2:
        low, high = levels[-2], levels[-1]
        keys = sorted(set(maps[low]).intersection(maps[high]))
        for key in keys:
            v_low = maps[low][key]
            v_high = maps[high][key]
            abs_diff = abs(v_high - v_low)
            rel_diff = abs_diff / max(1.0, abs(v_high))
            if rel_diff < 1e-5:
                status = "excellent"
            elif rel_diff < 1e-4:
                status = "stable"
            elif rel_diff < 1e-3:
                status = "usable"
            else:
                status = "check"
            transition, q2, current, quantity = key
            rows.append({
                "transition": transition,
                "q2": q2,
                "current": current,
                "quantity": quantity,
                "n_low": low,
                "value_low": v_low,
                "n_high": high,
                "value_high": v_high,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "status": status,
            })
    return {"enabled": True, "levels": levels, "rows": rows}


def diagnostic_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tname, tres in results.get("transitions_by_current", {}).items():
        for q2_str, qres in tres.get("q2_results", {}).items():
            # one row per current because the relevant projection depends on the current
            for current, cres in qres.get("by_current", {}).items():
                diag = cres.get("diagnostics", {})
                rows.append({
                    "transition": tname,
                    "q2": q2_str,
                    "current": current,
                    "projection_condition_number": fmt_value(diag.get("projection_condition_number"), 12),
                    "projection_status": diag.get("projection_status", ""),
                    "near_q2_zero": diag.get("near_q2_zero", ""),
                    "near_zero_recoil": diag.get("near_zero_recoil", ""),
                })
    return rows


def convergence_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in results.get("convergence", {}).get("rows", []):
        rows.append({
            "transition": r.get("transition", ""),
            "q2": r.get("q2", ""),
            "current": r.get("current", ""),
            "quantity": r.get("quantity", ""),
            "n_low": r.get("n_low", ""),
            "value_low": fmt_value(r.get("value_low"), 12),
            "n_high": r.get("n_high", ""),
            "value_high": fmt_value(r.get("value_high"), 12),
            "abs_diff": fmt_value(r.get("abs_diff"), 6),
            "rel_diff": fmt_value(r.get("rel_diff"), 6),
            "status": r.get("status", ""),
        })
    return rows

def write_outputs(results: dict[str, Any], output_dir: Path, *, plot_format: str = "png", include_plots: bool = True) -> dict[str, Any]:
    ensure_dir(output_dir)
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    ensure_dir(tables_dir)

    paths: dict[str, Any] = {}
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["json"] = json_path

    write_csv(tables_dir / "couplings.csv", coupling_rows(results), ["meson", "kind", "g_H", "Z_check", "source"])
    write_csv(tables_dir / "decay_constants.csv", decay_constant_rows(results), ["meson", "kind", "f_GeV", "f_MeV"])
    write_csv(tables_dir / "form_factors.csv", flatten_form_factors(results), ["transition", "q2", "current", "quantity", "value"])
    write_csv(tables_dir / "form_factor_trends.csv", collect_trend_rows(results), ["transition", "current", "form_factor", "q2_GeV2", "value"])
    write_trend_wide_csv(tables_dir / "form_factor_trends_wide.csv", results)
    write_csv(tables_dir / "warnings.csv", warning_rows(results), ["location", "message"])
    write_csv(tables_dir / "diagnostics.csv", diagnostic_rows(results), ["transition", "q2", "current", "projection_condition_number", "projection_status", "near_q2_zero", "near_zero_recoil"])
    write_csv(tables_dir / "convergence.csv", convergence_rows(results), ["transition", "q2", "current", "quantity", "n_low", "value_low", "n_high", "value_high", "abs_diff", "rel_diff", "status"])
    paths["tables"] = tables_dir

    txt_path = output_dir / "report.txt"
    write_text_report(txt_path, results)
    paths["txt"] = txt_path

    trend_txt_path = output_dir / "form_factor_trends.txt"
    write_trend_text_report(trend_txt_path, results)
    paths["trend_txt"] = trend_txt_path

    plot_paths: list[Path] = []
    if include_plots:
        plot_paths = write_plots(results, plots_dir, fmt=plot_format)
        paths["plots"] = plots_dir
        paths["plot_count"] = len(plot_paths)

    html_path = output_dir / "report.html"
    write_html_report(html_path, results, plots_dir if include_plots else None)
    paths["html"] = html_path

    trend_html_path = output_dir / "form_factor_trends.html"
    write_trend_html_report(trend_html_path, results)
    paths["trend_html"] = trend_html_path
    return paths


def copy_template(target: Path, *, force: bool = False) -> None:
    if target.exists() and not force:
        raise UserInputError(f"File already exists: {target}. Use --force to overwrite.")
    target.write_text(DEFAULT_TEMPLATE, encoding="utf-8")


def load_input(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise UserInputError(f"Input file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return parse_text_input(path)


def print_summary(results: dict[str, Any], paths: dict[str, Any], elapsed: float) -> None:
    n_c = len(results.get("couplings", {}))
    n_d = len(results.get("decay_constants", {}))
    n_t = len(results.get("transitions_by_current", {}))
    n_ff = len(flatten_form_factors(results))
    n_warn = len(warning_rows(results))
    n_diag = len(diagnostic_rows(results))
    n_conv = len(convergence_rows(results))
    print("\nDone.")
    print(f"  Couplings:       {n_c}")
    print(f"  Decay constants: {n_d}")
    print(f"  Transitions:     {n_t}")
    print(f"  Form-factor rows:{n_ff}")
    print(f"  Warnings:        {n_warn}")
    print(f"  Diagnostics:     {n_diag}")
    print(f"  Convergence rows:{n_conv}")
    print(f"  Time:            {elapsed:.2f} s")
    print("\nOutput folder:")
    print(f"  {paths['json'].parent}")
    print("Open first:")
    print(f"  {paths['txt']}")
    print(f"  {paths['html']}")
    print("Trend report:")
    print(f"  {paths['trend_txt']}")
    print(f"  {paths['trend_html']}")


def command_run(args: argparse.Namespace) -> int:
    t0 = time.time()
    data = load_input(args.input)
    apply_precision_preset(data, args.precision)
    pre_warnings = validate_input(data)
    ccqm = load_backend(args.backend)
    results = build_results(ccqm, data, progress=not args.quiet)
    results.setdefault("warnings", []).extend(pre_warnings)
    levels = resolve_convergence_levels(data, args.convergence)
    if levels:
        if not args.quiet:
            print("Step 4/4: numerical convergence diagnostics")
        results["convergence"] = compute_convergence(ccqm, data, levels, progress=not args.quiet)
    else:
        results["convergence"] = {"enabled": False, "levels": [], "rows": []}
    paths = write_outputs(results, args.output, plot_format=args.plot_format, include_plots=not args.no_plots)
    if not args.quiet:
        print_summary(results, paths, time.time() - t0)
    return 0


def command_validate(args: argparse.Namespace) -> int:
    data = load_input(args.input)
    apply_precision_preset(data, args.precision)
    warnings = validate_input(data)
    print("Input is valid.")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


def command_template(args: argparse.Namespace) -> int:
    copy_template(args.output, force=args.force)
    print(f"Wrote template input: {args.output}")
    return 0


def command_schema(args: argparse.Namespace) -> int:
    print("Supported currents")
    print("  final_kind = P:", ", ".join(sorted(SUPPORTED_CURRENTS["P"])))
    print("  final_kind = V:", ", ".join(sorted(SUPPORTED_CURRENTS["V"])))
    print("\nPrecision presets")
    for k, v in PRECISION_PRESETS.items():
        print(f"  {k:<10} n_quad={v}")
    print("\nConvergence modes")
    print("  off       no convergence check")
    print("  basic     compare final n_quad with one lower level")
    print("  strong    compare three levels")
    print("  research  compare several levels including n_quad+4")
    print("  8,12,16   explicit levels")
    return 0


DEFAULT_TEMPLATE = """# CCQM Stage-1 input file
# Scope: coupling constants, decay constants, and current-resolved form factors only.
# No decay widths, branching ratios, CKM factors, Wilson coefficients, or observables.

[global]
lambda_ir = 0.181
Nc = 3
n_quad = 8        # quick=4, standard=8, high=12, very_high=16

[mesons]
# name    kind   M       m1      m2      Lambda    optional_g
B         P      5.279   5.09    0.235   1.88
K         P      0.494   0.424   0.235   1.04
Bs        P      5.367   5.09    0.424   1.95
phi       V      1.019   0.424   0.424   0.88

[decay_constants]
B
K
Bs
phi

[transition B_to_K]
initial = B
final = K
final_kind = P
m1 = 5.09       # initial active quark
m2 = 0.424      # final active quark
m3 = 0.235      # spectator quark
currents = scalar, vector, tensor
q2_mode = list
q2_values = 0, 1, 5

[transition Bs_to_phi]
initial = Bs
final = phi
final_kind = V
m1 = 5.09
m2 = 0.424
m3 = 0.424
currents = v_minus_a, tensor_plus
q2_mode = range
q2_min = 1.0       # P->V tensor basis is singular at q2=0
q2_max = q2max
q2_points = 5
endpoint = false
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccqm.py", description="CCQM Stage-1 building-block calculator")
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a Stage-1 calculation")
    run.add_argument("input", type=Path, help="Input .txt or .json file")
    run.add_argument("--output", "-o", type=Path, default=Path("ccqm_output"), help="Output directory")
    run.add_argument("--backend", type=Path, default=Path(__file__).with_name("ccqm_stage1_backend.py"), help="Backend file")
    run.add_argument("--precision", choices=sorted(PRECISION_PRESETS), default=None, help="Override n_quad with a preset")
    run.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png")
    run.add_argument("--convergence", default="basic", help="Convergence check: off, basic, strong, research, or explicit levels like 8,12,16")
    run.add_argument("--no-plots", action="store_true", help="Do not generate plots")
    run.add_argument("--quiet", action="store_true", help="Only print errors")
    run.set_defaults(func=command_run)

    val = sub.add_parser("validate", help="Validate an input file without running integrals")
    val.add_argument("input", type=Path)
    val.add_argument("--precision", choices=sorted(PRECISION_PRESETS), default=None)
    val.set_defaults(func=command_validate)

    templ = sub.add_parser("template", help="Write a starter input file")
    templ.add_argument("output", type=Path, nargs="?", default=Path("ccqm_input.txt"))
    templ.add_argument("--force", action="store_true")
    templ.set_defaults(func=command_template)

    schema = sub.add_parser("schema", help="Print supported currents and precision presets")
    schema.set_defaults(func=command_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserInputError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
