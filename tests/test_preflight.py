import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load('scripts/preflight.py', 'preflight')
download = load('data/cached_challenge_fineweb.py', 'download')


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def shard(self, name, tokens=4):
        path = self.root / name
        path.write_bytes(struct.pack('<iii', 20240520, 1, tokens) + bytes(1012) + bytes(tokens * 2))
        return path

    def test_complete_training_inputs_and_truncation(self):
        self.shard('fineweb_train_000000.bin')
        val = self.shard('fineweb_val_000000.bin')
        tokenizer = self.root / 'tokenizer.model'
        tokenizer.write_bytes(b'fixture')
        self.assertEqual(preflight.check_inputs(self.root, tokenizer)['train']['tokens'], 4)
        val.write_bytes(val.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'truncated'):
            preflight.check_inputs(self.root, tokenizer)

    def test_missing_validation_data_fails_before_training(self):
        self.shard('fineweb_train_000000.bin')
        tokenizer = self.root / 'tokenizer.model'
        tokenizer.write_bytes(b'fixture')
        with self.assertRaisesRegex(ValueError, 'no val shards'):
            preflight.check_inputs(self.root, tokenizer)

    def test_manifest_cannot_escape_local_data_directory(self):
        for name in ('../outside.bin', '/tmp/outside.bin', 'datasets/../../outside.bin', 'C:\\outside.bin'):
            with self.assertRaises(ValueError):
                download.local_path_for_remote(name)

    def test_failed_copy_is_not_published_or_reused(self):
        cached = self.root / 'cache.bin'
        cached.write_bytes(b'complete')
        destination = self.root / 'target.bin'
        def interrupted(source, target):
            Path(target).write_bytes(b'partial')
            raise OSError('disk full')
        with patch.object(download, 'local_path_for_remote', return_value=destination), patch.object(download, 'hf_hub_download', return_value=str(cached)):
            with patch.object(download.shutil, 'copyfile', side_effect=interrupted), self.assertRaises(OSError):
                download.get('datasets/target.bin')
            self.assertFalse(destination.exists())
            self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['cache.bin'])
            download.get('datasets/target.bin')
            destination.write_bytes(b'edited')
            self.assertEqual(cached.read_bytes(), b'complete')


if __name__ == '__main__':
    unittest.main()
