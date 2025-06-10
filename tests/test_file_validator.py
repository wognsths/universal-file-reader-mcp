import importlib.util
import sys
import pathlib

# Load validators module with correct package to resolve relative imports
package_name = 'src.document_reader.core'
validators_path = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'document_reader' / 'core' / 'validators.py'

spec = importlib.util.spec_from_file_location(
    package_name + '.validators', validators_path,
    submodule_search_locations=[],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
FileValidator = module.FileValidator


def test_validate_file_basic_nonexistent(tmp_path):
    path = tmp_path / "missing.txt"
    result = FileValidator.validate_file_basic(str(path))
    assert not result["is_valid"]
    assert "File does not exist" in result["errors"][0]


def test_validate_file_basic_unreadable(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("hidden")
    path.chmod(0)
    try:
        import os
        original_access = os.access
        os.access = lambda *_: False
        result = FileValidator.validate_file_basic(str(path))
    finally:
        os.access = original_access
        path.chmod(0o644)
    assert not result["is_valid"]
    assert "File read permission denied" in result["errors"][0]
