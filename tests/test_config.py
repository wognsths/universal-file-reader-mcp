import importlib.util
import sys
import pathlib

package_name = 'src.document_reader.core'
config_path = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'document_reader' / 'core' / 'config.py'

spec = importlib.util.spec_from_file_location(
    package_name + '.config', config_path,
    submodule_search_locations=[],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
ProcessorConfig = module.ProcessorConfig


def test_max_page_per_process_env(monkeypatch):
    monkeypatch.setenv("MAX_PAGE_PER_PROCESS", "3")
    config = ProcessorConfig.from_env()
    assert config.ocr_config.MAX_PAGE_PER_PROCESS == 3


def test_timeout_seconds_env(monkeypatch):
    monkeypatch.setenv("TIMEOUT_SECONDS", "300")
    config = ProcessorConfig.from_env()
    assert config.global_config.TIMEOUT_SECONDS == 300


def test_model_name_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "gpt-test")
    config = ProcessorConfig.from_env()
    assert config.ocr_config.MODEL_NAME == "gpt-test"
