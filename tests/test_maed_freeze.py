"""Small archive tests; do not instantiate or train recommendation models."""
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.freeze_maed_v1 import inside, sha256, verify_inventory


class FreezeTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'tmp').mkdir(exist_ok=True)

    def test_exact_hash_detects_same_size_mutation(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            path = root / 'checkpoint.pt'
            path.write_bytes(b'abcd')
            entries = {'checkpoint.pt': {'sha256': sha256(path), 'bytes': 4}}
            verify_inventory(root, entries)
            path.write_bytes(b'abce')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_inventory(root, entries)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                verify_inventory(root, entries)

    def test_inventory_cannot_escape_root(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tmp') as directory:
            root = Path(directory)
            self.assertEqual(inside(root, 'nested/file'), root.resolve() / 'nested/file')
            for relative in ('../outside', str(root.resolve() / 'absolute')):
                with self.assertRaises(ValueError):
                    inside(root, relative)


if __name__ == '__main__':
    unittest.main()
