"""Unit tests for the ReportGenerator and ShiftAggregator."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import ComplianceReport, Violation
from inference.reporter import ReportGenerator, ShiftAggregator


def make_report(
    verdict: str = "SAFE", violations: int = 0, workers: int = 2
) -> ComplianceReport:
    viols = []
    for i in range(violations):
        viols.append(Violation(
            worker_id=i, rule_id=f"PPE-{i+1:03d}",
            rule_name=f"Test Rule {i+1}",
            severity="CRITICAL" if i == 0 else "HIGH",
            description=f"Test violation {i+1}",
            confidence=0.85, bbox=(100, 100, 300, 500),
            suggested_action="Fix it.",
        ))
    return ComplianceReport(
        frame_id="test_frame", timestamp="2026-04-01 12:00:00",
        scene_verdict=verdict, overall_confidence=0.85,
        worker_count=workers, compliant_workers=workers - violations,
        violation_count=violations, violations=viols,
    )


class TestReportGenerator:
    def test_to_dict(self):
        report = make_report("UNSAFE", violations=1)
        rg = ReportGenerator(output_dir=tempfile.mkdtemp())
        d = rg.to_dict(report)
        assert d["scene_verdict"] == "UNSAFE"
        assert d["violation_count"] == 1
        assert len(d["violations"]) == 1

    def test_save_json(self):
        report = make_report()
        tmp = tempfile.mkdtemp()
        rg = ReportGenerator(output_dir=tmp)
        fp = rg.save_json(report, "test.json")
        assert fp.exists()
        with open(fp) as f:
            data = json.load(f)
        assert data["frame_id"] == "test_frame"

    def test_print_summary_safe(self, capsys):
        report = make_report("SAFE")
        rg = ReportGenerator(output_dir=tempfile.mkdtemp())
        s = rg.print_summary(report)
        assert "SAFE" in s

    def test_print_summary_unsafe(self, capsys):
        report = make_report("UNSAFE", violations=2)
        rg = ReportGenerator(output_dir=tempfile.mkdtemp())
        s = rg.print_summary(report)
        assert "UNSAFE" in s
        assert "VIOLATIONS" in s

    def test_csv_append(self):
        tmp = tempfile.mkdtemp()
        rg = ReportGenerator(output_dir=tmp)
        r1 = make_report("SAFE")
        r2 = make_report("UNSAFE", violations=1)
        rg.append_csv(r1, "log.csv")
        rg.append_csv(r2, "log.csv")
        fp = Path(tmp) / "log.csv"
        lines = fp.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_violation_dict_fields(self):
        v = Violation(worker_id=1, rule_id="PPE-001",
                      rule_name="Hard Hat", severity="CRITICAL",
                      description="Missing", confidence=0.9,
                      bbox=(10, 20, 30, 40), suggested_action="Fix")
        d = ReportGenerator._viol_dict(v)
        assert d["rule_id"] == "PPE-001"
        assert d["bbox"] == [10, 20, 30, 40]


class TestShiftAggregator:
    def test_empty(self):
        agg = ShiftAggregator()
        s = agg.get_summary()
        assert s["status"] == "no_data"

    def test_aggregation(self):
        agg = ShiftAggregator()
        agg.add_report(make_report("SAFE"))
        agg.add_report(make_report("UNSAFE", violations=2))
        agg.add_report(make_report("SAFE"))
        s = agg.get_summary()
        assert s["total_frames"] == 3
        assert s["unsafe_frames"] == 1
        assert s["total_violations"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
