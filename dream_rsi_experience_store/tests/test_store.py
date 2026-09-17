import json
import sqlite3
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "experience_store" / "experience.db"


class SimplifiedExperienceStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        cls.db.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_only_three_tables_exist(self):
        tables = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        self.assertEqual(tables, {"runs", "experiences", "raw_events"})

    def test_required_indexes_exist(self):
        indexes = {
            row[0]
            for row in self.db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
        self.assertEqual(
            indexes,
            {
                "idx_experiences_run_id",
                "idx_experiences_parent_id",
                "idx_raw_events_node_id",
                "idx_raw_events_run_id",
                "idx_raw_events_sequence_no",
            },
        )

    def test_foreign_keys_are_valid(self):
        self.assertEqual(list(self.db.execute("PRAGMA foreign_key_check")), [])

    def test_tree_has_one_declared_root_per_run(self):
        for run in self.db.execute("SELECT run_id,root_node_id FROM runs"):
            roots = list(
                self.db.execute(
                    "SELECT node_id FROM experiences WHERE run_id=? AND parent_id IS NULL",
                    (run["run_id"],),
                )
            )
            self.assertEqual([row["node_id"] for row in roots], [run["root_node_id"]])

    def test_tree_has_no_orphan_or_cycle(self):
        orphan_count = self.db.execute(
            """SELECT count(*) FROM experiences child
               LEFT JOIN experiences parent ON parent.node_id=child.parent_id
               WHERE child.parent_id IS NOT NULL AND parent.node_id IS NULL"""
        ).fetchone()[0]
        self.assertEqual(orphan_count, 0)
        for run_id, in self.db.execute("SELECT run_id FROM runs"):
            parents = dict(
                self.db.execute(
                    "SELECT node_id,parent_id FROM experiences WHERE run_id=?", (run_id,)
                )
            )
            for start in parents:
                seen, current = set(), start
                while current:
                    self.assertNotIn(current, seen, f"cycle in {run_id}")
                    seen.add(current)
                    current = parents.get(current)

    def test_experience_aggregates_are_valid_json(self):
        for row in self.db.execute(
            "SELECT metrics_json,artifacts_json,commit_hashes_json,source_files_json,metadata_json FROM experiences"
        ):
            self.assertIsInstance(json.loads(row["metrics_json"]), dict)
            self.assertIsInstance(json.loads(row["artifacts_json"]), list)
            self.assertIsInstance(json.loads(row["commit_hashes_json"]), list)
            self.assertIsInstance(json.loads(row["source_files_json"]), list)
            self.assertIsInstance(json.loads(row["metadata_json"]), dict)

    def test_score_is_preserved_in_metrics(self):
        for row in self.db.execute(
            "SELECT score,score_name,metrics_json FROM experiences WHERE score IS NOT NULL"
        ):
            metric = json.loads(row["metrics_json"])[row["score_name"]]
            value = metric["value"] if isinstance(metric, dict) else metric
            self.assertAlmostEqual(value, row["score"], places=12)

    def test_event_counts_match_raw_events(self):
        mismatches = self.db.execute(
            """SELECT count(*) FROM experiences x
               WHERE x.event_count != (
                 SELECT count(*) FROM raw_events e WHERE e.node_id=x.node_id
               )"""
        ).fetchone()[0]
        self.assertEqual(mismatches, 0)

    def test_bound_event_has_matching_run(self):
        mismatches = self.db.execute(
            """SELECT count(*) FROM raw_events e
               JOIN experiences x ON x.node_id=e.node_id
               WHERE e.run_id IS NOT x.run_id"""
        ).fetchone()[0]
        self.assertEqual(mismatches, 0)

    def test_raw_payload_and_source_are_retained(self):
        missing = self.db.execute(
            """SELECT count(*) FROM raw_events
               WHERE raw_json IS NULL OR raw_json='' OR source_file IS NULL OR source_file=''"""
        ).fetchone()[0]
        self.assertEqual(missing, 0)


if __name__ == "__main__":
    unittest.main()
