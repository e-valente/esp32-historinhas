"""Streaming helpers for the pinned Historinhas preparation pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from research import prepare


class TestPreparation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / "historinhas_train.json"
        self.original_raw = prepare.RAW
        prepare.RAW = self.raw
        self.addCleanup(setattr, prepare, "RAW", self.original_raw)

    def write_rows(self, rows):
        self.raw.write_text(json.dumps(rows, ensure_ascii=False))

    def test_documents_stream_text_field_and_normalize_line_endings(self):
        self.write_rows([
            {"texto": "Ação\r\ncoração"},
            {"texto": "\t"},
            {"texto": "João\rviu o céu"},
        ])
        self.assertEqual(list(prepare.documents()),
                         ["Ação\ncoração", "João\nviu o céu"])

    def test_missing_text_field_is_rejected(self):
        self.write_rows([{"titulo": "sem texto"}])
        with self.assertRaisesRegex(SystemExit, "texto"):
            list(prepare.documents())

    def test_truncated_json_is_rejected(self):
        self.raw.write_text('[{"texto": "Era uma vez"}')
        with self.assertRaisesRegex(SystemExit, "closing"):
            list(prepare.documents())

    def test_document_hash_split_is_stable(self):
        self.assertEqual(prepare.is_validation("Ação e coração"),
                         prepare.is_validation("Ação e coração"))


if __name__ == "__main__":
    unittest.main()
