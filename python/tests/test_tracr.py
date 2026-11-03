import pytest

pytest.importorskip("tracr")
from weightscript.backends.tracr import (
    compile_program,
    patch,
    verify,
    recover,
    Compiled,
)
from weightscript.symbolic import previous_program, count_program


def test_real_tracr_closed_loop_and_weights_only_runtime(tmp_path):
    source = previous_program((0,), vocab=(0, 1), max_seq_len=3)
    original, native = compile_program(source)
    assert original.predict([1, 0, 1]) == native.apply(["BOS", 1, 0, 1]).decoded[1:]
    assert verify(original, source, 3)["passed"]
    changed = previous_program((0, 1), vocab=(0, 1), max_seq_len=3)
    edited, report = patch(original, changed)
    assert report["changed_scalars"] > 0 and report["architecture_preserved"]
    assert verify(edited, changed, 3)["passed"]
    # Recovery must not consult embedded source.
    original.metadata.pop("source")
    assert recover(original, [source, changed])["program"] == source
    edited.metadata.pop("source")
    edited.save(tmp_path / "model")
    loaded = Compiled.load(tmp_path / "model")
    assert loaded.predict([0, 1, 0]) == edited.predict([0, 1, 0])
    loaded.params[next(iter(loaded.params))].flat[0] += 1
    assert loaded.fingerprint() != edited.fingerprint()


def test_count_primitive_compiles():
    source = count_program(0.5, False, max_seq_len=5)
    compiled, native = compile_program(source)
    for tokens in ([2, 0, 1, 0, 3], [2, 1, 0, 1, 3]):
        assert (
            compiled.predict(tokens) == native.apply(["BOS"] + list(tokens)).decoded[1:]
        )
        assert compiled.predict(tokens)[-1] == int(tokens.count(1) > tokens.count(0))
