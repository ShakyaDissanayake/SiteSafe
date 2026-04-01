"""Compliance Report Generator.

Converts ComplianceReport dataclass into JSON, text, CSV, and HTML formats.

Typical usage:
    reporter = ReportGenerator(output_dir="./reports")
    reporter.save_json(report, "frame_001.json")
    reporter.print_summary(report)
"""

import csv
import json
from pathlib import Path
from typing import Optional

from inference import ComplianceReport, Violation


class ReportGenerator:
    """Generates and saves compliance reports in multiple formats."""

    def __init__(self, output_dir: str = "./reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self, report: ComplianceReport) -> dict:
        """Convert a ComplianceReport to a JSON-serializable dict."""
        return {
            "frame_id": report.frame_id,
            "timestamp": report.timestamp,
            "scene_verdict": report.scene_verdict,
            "overall_confidence": round(report.overall_confidence, 4),
            "worker_count": report.worker_count,
            "compliant_workers": report.compliant_workers,
            "violation_count": report.violation_count,
            "violations": [self._viol_dict(v) for v in report.violations],
            "scene_brightness": round(report.scene_brightness, 1),
            "low_confidence_flags": report.low_confidence_flags,
        }

    def save_json(self, report: ComplianceReport, filename: str) -> Path:
        """Save report as a JSON file."""
        fp = self.output_dir / filename
        with open(fp, "w") as f:
            json.dump(self.to_dict(report), f, indent=2)
        return fp

    def print_summary(self, report: ComplianceReport) -> str:
        """Print and return a formatted text summary."""
        icons = {"SAFE": "🟢", "WARNING": "🟡", "UNSAFE": "🔴"}
        sev_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}

        lines = ["=" * 60, "  SAFETY COMPLIANCE REPORT", "=" * 60]
        lines.append(f"  Frame: {report.frame_id}  |  Time: {report.timestamp}")
        ic = icons.get(report.scene_verdict, "❓")
        lines.append(f"  Verdict: {ic} {report.scene_verdict}  |  Confidence: {report.overall_confidence:.1%}")
        lines.append(f"  Workers: {report.worker_count}  |  Compliant: {report.compliant_workers}  |  Violations: {report.violation_count}")

        if report.violations:
            lines.append("\n  ── VIOLATIONS ──")
            for v in report.violations:
                si = sev_icons.get(v.severity, "❓")
                lines.append(f"  {si} [{v.rule_id}] {v.description}")
                lines.append(f"      Confidence: {v.confidence:.1%}  |  Action: {v.suggested_action}")

        if report.low_confidence_flags:
            lines.append("\n  ── LOW CONFIDENCE WARNINGS ──")
            for flag in report.low_confidence_flags:
                lines.append(f"  ⚠️  {flag}")

        lines.append("=" * 60)
        summary = "\n".join(lines)
        print(summary)
        return summary

    def append_csv(self, report: ComplianceReport, csv_filename: str = "compliance_log.csv") -> Path:
        """Append report summary as a CSV row."""
        fp = self.output_dir / csv_filename
        exists = fp.exists()
        headers = ["frame_id", "timestamp", "scene_verdict", "overall_confidence",
                    "worker_count", "compliant_workers", "violation_count",
                    "critical_violations", "high_violations", "scene_brightness"]
        crit = sum(1 for v in report.violations if v.severity == "CRITICAL")
        high = sum(1 for v in report.violations if v.severity == "HIGH")
        row = [report.frame_id, report.timestamp, report.scene_verdict,
               f"{report.overall_confidence:.4f}", report.worker_count,
               report.compliant_workers, report.violation_count, crit, high,
               f"{report.scene_brightness:.1f}"]
        with open(fp, "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(headers)
            w.writerow(row)
        return fp

    @staticmethod
    def _viol_dict(v: Violation) -> dict:
        return {"worker_id": v.worker_id, "rule_id": v.rule_id,
                "rule_name": v.rule_name, "severity": v.severity,
                "description": v.description, "confidence": round(v.confidence, 4),
                "bbox": list(v.bbox), "suggested_action": v.suggested_action}


class ShiftAggregator:
    """Aggregates compliance reports across a work shift."""

    def __init__(self) -> None:
        self.reports: list[ComplianceReport] = []
        self.shift_start: Optional[str] = None

    def add_report(self, report: ComplianceReport) -> None:
        if not self.shift_start:
            self.shift_start = report.timestamp
        self.reports.append(report)

    def get_summary(self) -> dict:
        if not self.reports:
            return {"status": "no_data"}
        total_v = sum(r.violation_count for r in self.reports)
        total_f = len(self.reports)
        unsafe = sum(1 for r in self.reports if r.scene_verdict == "UNSAFE")
        vc: dict[str, int] = {}
        for r in self.reports:
            for v in r.violations:
                vc[v.rule_id] = vc.get(v.rule_id, 0) + 1
        top = sorted(vc.items(), key=lambda x: x[1], reverse=True)[:5]
        return {"shift_start": self.shift_start, "shift_end": self.reports[-1].timestamp,
                "total_frames": total_f, "unsafe_frames": unsafe,
                "unsafe_rate": unsafe / max(total_f, 1),
                "total_violations": total_v,
                "avg_violations_per_frame": total_v / max(total_f, 1),
                "top_violations": [{"rule_id": r, "count": c} for r, c in top]}
