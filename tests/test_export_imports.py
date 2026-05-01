import importlib


def test_export_modules_import():
    # ensure export modules import and provide helper functions
    mod_names = [
        "export.export_torchscript",
        "export.export_onnx",
        "export.quantize_dynamic",
        "export.benchmark_cpu",
    ]

    for name in mod_names:
        m = importlib.import_module(name)
        # lookup a common helper used for dry-run
        assert any(hasattr(m, fn) for fn in ("export_torchscript", "export_onnx", "quantize_dynamic", "benchmark")) or hasattr(m, "export_torchscript")
