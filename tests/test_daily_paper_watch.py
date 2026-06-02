import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from src.daily_paper_watch import (
    Paper,
    build_current_meta,
    generate_site,
    load_config,
    render_report,
    score_papers,
    write_report,
)


class DailyPaperWatchTests(unittest.TestCase):
    def test_load_config_fallback_yaml(self):
        config = load_config(__import__("pathlib").Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertIn("cs.RO", config["categories"])
        self.assertIn("embodied navigation", config["priority_keywords"])

    def test_score_prioritizes_embodied_navigation(self):
        now = datetime.now(timezone.utc)
        papers = [
            Paper(
                paper_id="2601.00001",
                title="Language-Guided Embodied Navigation with Semantic Maps",
                authors=("A. Researcher",),
                summary="We study vision-language navigation for robot navigation in 3D scenes.",
                published=now,
                updated=now,
                categories=("cs.RO", "cs.CV"),
                abs_url="https://arxiv.org/abs/2601.00001",
                pdf_url="https://arxiv.org/pdf/2601.00001",
            ),
            Paper(
                paper_id="2601.00002",
                title="Protein Structure Prediction with Transformers",
                authors=("B. Researcher",),
                summary="A molecular biology paper about proteins.",
                published=now,
                updated=now,
                categories=("cs.LG",),
                abs_url="https://arxiv.org/abs/2601.00002",
                pdf_url="https://arxiv.org/pdf/2601.00002",
            ),
        ]
        config = {
            "priority_keywords": ["embodied navigation", "vision-language navigation", "robot navigation"],
            "keywords": ["semantic maps", "semantic map"],
            "related_keywords": ["3D scenes"],
            "exclude_keywords": ["protein"],
            "min_score": 2.0,
        }
        scored = score_papers(papers, config)
        self.assertEqual([item.paper.paper_id for item in scored], ["2601.00001"])
        self.assertIn("embodied navigation", scored[0].matched_keywords)

    def test_score_rejects_generic_robotics_without_navigation(self):
        now = datetime.now(timezone.utc)
        papers = [
            Paper(
                paper_id="2601.00004",
                title="Composable World Models for Robot Data Synthesis",
                authors=("E. Researcher",),
                summary="We synthesize robot manipulation data with visual scene generation and policy learning.",
                published=now,
                updated=now,
                categories=("cs.RO", "cs.CV"),
                abs_url="https://arxiv.org/abs/2601.00004",
                pdf_url="https://arxiv.org/pdf/2601.00004",
            )
        ]
        config = {
            "priority_keywords": ["embodied navigation", "vision-language navigation", "robot navigation"],
            "keywords": ["semantic map", "visual navigation"],
            "core_focus_keywords": ["navigation", "VLN", "objectnav", "pointnav", "robot navigation"],
            "embodiment_context_keywords": ["embodied", "robot", "agent", "3D scene"],
            "related_keywords": ["robot"],
            "exclude_keywords": [],
            "strict_focus": True,
            "min_score": 2.0,
        }
        self.assertEqual(score_papers(papers, config), [])

    def test_generate_static_site(self):
        now = datetime.now(timezone.utc)
        paper = Paper(
            paper_id="2601.00003",
            title="Vision-Language Navigation with Persistent Semantic Memory",
            authors=("C. Researcher", "D. Researcher"),
            summary="A robot navigation system for embodied navigation with semantic map memory.",
            published=now,
            updated=now,
            categories=("cs.RO", "cs.CV"),
            abs_url="https://arxiv.org/abs/2601.00003",
            pdf_url="https://arxiv.org/pdf/2601.00003",
        )
        config = {
            "project_name": "Embodied Nav Paper Watch",
            "domain_name": "具身智能导航",
            "timezone": "Asia/Shanghai",
            "categories": ["cs.RO", "cs.CV"],
            "priority_keywords": ["embodied navigation", "vision-language navigation", "robot navigation"],
            "keywords": ["semantic map"],
            "related_keywords": [],
            "exclude_keywords": [],
            "min_score": 2.0,
        }
        scored = score_papers([paper], config)
        tz = ZoneInfo("Asia/Shanghai")
        run_time = datetime(2026, 6, 2, 1, 0, tzinfo=timezone.utc)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "out"
            site_dir = tmp_path / "site"
            report = render_report(
                scored,
                config=config,
                scanned_count=1,
                candidate_count=1,
                fetch_warnings=[],
                run_time=run_time,
                tz=tz,
                max_items=5,
            )
            report_path = write_report(report, out_dir, run_time, tz)
            current_meta = build_current_meta(
                scored,
                config=config,
                scanned_count=1,
                candidate_count=1,
                fetch_warnings=[],
                run_time=run_time,
                tz=tz,
                max_items=5,
            )
            generate_site(
                report_path=report_path,
                out_dir=out_dir,
                site_dir=site_dir,
                current_meta=current_meta,
                config=config,
            )

            index_html = (site_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("Vision-Language Navigation", index_html)
            self.assertIn("解决的问题", index_html)
            self.assertIn("TL;DR", index_html)
            self.assertIn("核心贡献", index_html)
            self.assertIn("局限性", index_html)
            self.assertTrue((site_dir / "reports" / "2026-06-02.html").exists())
            self.assertTrue((site_dir / "archive" / "index.html").exists())
            self.assertTrue((site_dir / "daily" / "2026-06-02" / "index.html").exists())
            self.assertTrue(any((site_dir / "papers" / "2026-06-02").glob("*.html")))
            self.assertTrue((site_dir / "docs" / "202606" / "02" / "README.md").exists())
            self.assertTrue((site_dir / "docs" / "202606" / "02" / "papers.meta.json").exists())
            self.assertTrue((site_dir / "assets" / "app.css").exists())
            self.assertTrue((site_dir / "data" / "reports.json").exists())


if __name__ == "__main__":
    unittest.main()
