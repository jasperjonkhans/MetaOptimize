"""Offline export regressions and memory-checked original C chat/tokenizer paths.

HF tests stub only LlamaConfig (not a Transformers roundtrip); tensors and the
small Transformer are real Torch. C executables are always built in temp dirs.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "codes/MetaOptimize/tinystories"
sys.path.insert(0, str(TINY))
import torch
from model import ModelArgs, Transformer

spec = importlib.util.spec_from_file_location("tinystories_export_safety", TINY / "export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ConfigStub:
    def __init__(self, **kwargs):
        self.values = kwargs

    def save_pretrained(self, path):
        (Path(path) / "config.json").write_text(json.dumps(self.values))


def hf_config_stub():
    module = types.ModuleType("transformers.models.llama.configuration_llama")
    module.LlamaConfig = ConfigStub
    return mock.patch.dict(sys.modules, {module.__name__: module})


class ExportSafetyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = Transformer(ModelArgs(dim=8, n_layers=1, n_heads=2,
                                          n_kv_heads=2, vocab_size=16,
                                          hidden_dim=16, multiple_of=4, max_seq_len=8))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_legacy_export_preserves_tied_and_untied_model(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                if not tied:
                    self.model.output.weight = torch.nn.Parameter(self.model.output.weight.detach().clone() + 1)
                params = copy.deepcopy(vars(self.model.params))
                state = {k: v.clone() for k, v in self.model.state_dict().items()}
                identities = [id(p) for p in self.model.parameters()]
                with contextlib.redirect_stdout(io.StringIO()):
                    export.legacy_export(self.model, self.path / "first.bin")
                    export.legacy_export(self.model, self.path / "second.bin")
                first = (self.path / "first.bin").read_bytes()
                self.assertEqual(first, (self.path / "second.bin").read_bytes())
                self.assertEqual(struct.unpack("7i", first[:28])[5], 16 if tied else -16)
                self.assertEqual(vars(self.model.params), params)
                self.assertEqual([id(p) for p in self.model.parameters()], identities)
                for key, value in self.model.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]), key)

    def test_normal_import_does_not_require_transformers(self):
        real_import = __import__

        def without_transformers(name, *args, **kwargs):
            if name == "transformers" or name.startswith("transformers."):
                raise ImportError("Transformers deliberately unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=without_transformers):
            runpy.run_path(str(TINY / "export.py"), run_name="export_without_transformers")

    def test_model_export_dispatches_named_torch_dtype(self):
        for dtype in (torch.float16, torch.float32):
            with mock.patch.object(export, "hf_export") as target:
                export.model_export(self.model, self.path, -1, dtype)
                target.assert_called_once_with(self.model, self.path, dtype=dtype)

    def test_hf_saved_tensor_dtypes_with_config_stub(self):
        with hf_config_stub():
            for dtype in (torch.float16, torch.float32):
                export.model_export(self.model, self.path, -1, dtype)
                state = torch.load(self.path / "pytorch_model.bin", weights_only=True)
                self.assertTrue(all(t.dtype == dtype for t in state.values()))
                self.assertEqual(json.loads((self.path / "config.json").read_text())["vocab_size"], 16)

    def test_cli_passes_converted_dtype_with_config_stub(self):
        checkpoint = {"model_args": vars(self.model.params), "model": self.model.state_dict()}
        with hf_config_stub(), mock.patch.object(torch, "load", return_value=checkpoint), mock.patch.object(
            sys, "argv", ["export.py", str(self.path), "--version", "-1", "--dtype", "fp16", "--checkpoint", "unused.pt"]
        ):
            runpy.run_path(str(TINY / "export.py"), run_name="__main__")
        state = torch.load(self.path / "pytorch_model.bin", weights_only=True)
        self.assertTrue(all(t.dtype == torch.float16 for t in state.values()))

    def test_export_rejects_unsupported_options_before_writing(self):
        for version, dtype in ((3, torch.float32), (-1, "fp16"), (-1, torch.bfloat16),
                               (0, torch.float16), (1, torch.float16), (2, torch.float16)):
            with self.subTest(version=version, dtype=dtype), self.assertRaises(ValueError):
                export.model_export(self.model, self.path / "bad", version, dtype)
        self.assertFalse((self.path / "bad").exists())
        with self.assertRaises(ValueError):
            export.hf_export(self.model, self.path / "bad", dtype="fp16")

    def test_cli_rejects_invalid_options_before_loading_model(self):
        for options in (("--version", "3"), ("--dtype", "bad"), ("--dtype", "fp16")):
            with self.subTest(options=options), mock.patch.object(torch, "load") as load, mock.patch.object(
                sys, "argv", ["export.py", str(self.path), "--checkpoint", "unused.pt", *options]
            ), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runpy.run_path(str(TINY / "export.py"), run_name="__main__")
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_quantization_shape_error_names_weight_not_unbound_i(self):
        self.model.layers[0].attention.wq.weight = torch.nn.Parameter(torch.ones(3, 3))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "weight 1 has numel 9"):
            export.version2_export(self.model, self.path / "bad", group_size=8)
        self.assertFalse((self.path / "bad").exists())

    def test_quantization_rejects_invalid_group_sizes(self):
        for group_size in (0, -2, 1.5, True):
            with self.subTest(group_size=group_size), self.assertRaisesRegex(ValueError, "positive integer"):
                export.version2_export(self.model, self.path / "bad", group_size)


# Includes the complete, original functions, with no encode/chat/forward stubs.
C_HARNESS = r'''
#define TESTING
#include "SOURCE"
int main(int argc, char **argv) {
    if (argc != 4) return 10;
    Transformer t = {0};
    t.config = (Config){.dim=2, .hidden_dim=2, .n_layers=0, .n_heads=1,
                        .n_kv_heads=1, .vocab_size=32000, .seq_len=2};
    malloc_run_state(&t.state, &t.config);
    t.weights.token_embedding_table = malloc(32000 * 2 * sizeof(float));
    for (int i=0; i<32000*2; i++) t.weights.token_embedding_table[i] = 1.0f;
    float norm[2] = {1,1};
    t.weights.rms_final_weight = norm;
#ifdef QUANTIZED
    GS = 2;
    QuantizedTensor classifier = {.q=calloc(32000*2, 1), .s=calloc(32000, sizeof(float))};
    t.weights.wcls = &classifier;
    if (atoi(argv[3]) > 1) { classifier.q[4] = 1; classifier.s[2] = 1.0f; }
#else
    t.weights.wcls = calloc(32000*2, sizeof(float));
    if (atoi(argv[3]) > 1) t.weights.wcls[4] = 1.0f;
#endif
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", 32000);
    Sampler sampler;
    build_sampler(&sampler, 32000, 0.0f, 0.9f, 1);
    char *user = strcmp(argv[1], "STDIN") == 0 ? NULL : argv[1];
    char *system = strcmp(argv[2], "STDIN") == 0 ? NULL : argv[2];
    chat(&t, &tokenizer, &sampler, user, system, atoi(argv[3]));
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_run_state(&t.state);
    free(t.weights.token_embedding_table);
#ifdef QUANTIZED
    free(classifier.q); free(classifier.s);
#else
    free(t.weights.wcls);
#endif
    return 0;
}
'''


class CChatSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.path = Path(cls.temp.name)
        cls.env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0", UBSAN_OPTIONS="halt_on_error=1")
        # Leak detection is disabled for rejected prompts that exit immediately;
        # ASan bounds checking and UBSan remain enabled in every subprocess.
        cls.flags = None
        for compiler, flags, label in (
            ("clang", ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"], "Clang ASan+UBSan"),
            ("cc", ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"], "GCC ASan+UBSan"),
            ("cc", ["-D_FORTIFY_SOURCE=2", "-fstack-protector-all"], "fortify fallback"),
        ):
            if not shutil.which(compiler):
                continue
            probe = cls.path / "probe.c"
            probe.write_text("int main(void) { return 0; }\n")
            build = subprocess.run([compiler, "-O1", *flags, str(probe), "-o", str(cls.path / "probe")], capture_output=True)
            if build.returncode == 0 and subprocess.run([str(cls.path / "probe")], env=cls.env, capture_output=True).returncode == 0:
                cls.compiler, cls.flags, cls.checker = compiler, flags, label
                break
        if cls.flags is None:
            raise RuntimeError("No memory-checking compiler available")
        print("C memory checker:", cls.checker)
        cls.binaries = []
        for source in ("run.c", "runq.c"):
            harness = cls.path / (source + ".harness.c")
            harness.write_text(C_HARNESS.replace("SOURCE", source))
            binary = cls.path / (source + ".harness")
            command = [cls.compiler, "-O1", "-g", *cls.flags, "-I", str(TINY), str(harness), "-lm", "-o", str(binary)]
            if source == "runq.c":
                command.insert(1, "-DQUANTIZED")
            subprocess.run(command, check=True, capture_output=True)
            cls.binaries.append(binary)

    def run_chat(self, binary, user, system, text="", steps=1):
        result = subprocess.run([str(binary), user, system, str(steps)], input=text,
                                text=True, capture_output=True, cwd=TINY, env=self.env, timeout=10)
        self.assertNotIn("AddressSanitizer", result.stderr)
        self.assertNotIn("runtime error:", result.stderr)
        return result

    def test_cli_role_byte_boundaries(self):
        for binary in self.binaries:
            for role in ("user", "system"):
                for length in (0, 511, 512, 600):
                    with self.subTest(source=binary.name, role=role, length=length):
                        user, system = ("x" * length, "") if role == "user" else ("", "x" * length)
                        result = self.run_chat(binary, user, system)
                        self.assertEqual(result.returncode, 0 if length < 512 else 1, result.stderr)
                        if length >= 512:
                            self.assertIn("maximum 511 bytes", result.stderr)

    def test_combined_maximum_rendered_prompt(self):
        for binary in self.binaries:
            result = self.run_chat(binary, "x" * 511, "y" * 511)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Assistant:", result.stdout)

    def test_stdin_role_boundaries_without_truncation(self):
        for binary in self.binaries:
            for role in ("user", "system"):
                for length in (0, 511, 512, 600):
                    for ending in ("\n", ""):
                        with self.subTest(source=binary.name, role=role, length=length, ending=ending):
                            user, system = ("STDIN", "") if role == "user" else ("", "STDIN")
                            result = self.run_chat(binary, user, system, "x" * length + ending)
                            self.assertEqual(result.returncode, 0 if length < 512 else 1, result.stderr)
                            if length >= 512:
                                self.assertIn("maximum 511 bytes", result.stderr)
                            elif length or ending:
                                self.assertIn("Assistant:", result.stdout)

    def test_eof_ends_chat_without_inference(self):
        for binary in self.binaries:
            for user, system, text in (("STDIN", "", ""), ("", "STDIN", ""), ("STDIN", "STDIN", "\n")):
                result = self.run_chat(binary, user, system, text)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("Assistant:", result.stdout)

    def test_eof_after_assistant_turn_does_not_reuse_prompt(self):
        for binary in self.binaries:
            result = self.run_chat(binary, "hi", "", steps=128)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("Assistant:"), 1)
            self.assertIn("User:", result.stdout)

    def test_original_five_c_tokenizer_fixtures(self):
        # Preserve all five fixtures; also exercise runq's tokenizer implementation.
        for source in ("run.c", "runq.c"):
            binary = self.path / ("testc-" + source)
            fixture_source = self.path / ("fixtures-" + source)
            fixture_source.write_text((TINY / "test.c").read_text().replace('#include "run.c"', f'#include "{source}"'))
            subprocess.run([self.compiler, "-O1", "-g", *self.flags, "-I", str(TINY), str(fixture_source), "-lm", "-o", str(binary)], check=True, capture_output=True)
            result = subprocess.run([str(binary)], cwd=TINY, env=self.env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("AddressSanitizer", result.stderr)
            self.assertNotIn("runtime error:", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
