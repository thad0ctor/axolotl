# Fix: make Liger Gemma 4 RMSNorm deepcopy-safe

When `liger_rms_norm: true` is set on a Gemma 4 VLM and any module
containing `Gemma4RMSNorm` instances (typically `vision_tower`) is added
to `lora_modules_to_save`, axolotl crashes at adapter setup time inside
PEFT's `ModulesToSaveWrapper.update`, which calls `copy.deepcopy` on the
wrapped module. The Liger replacement class for `Gemma4RMSNorm` requires
`dim` on `__new__`/`__init__`, which is incompatible with how Python's
pickle/deepcopy reconstruction protocol invokes `cls.__new__(cls)`.

This PR makes `dim` optional and short-circuits `__init__` when called
with the pickle-protocol empty signature. Normal construction is
unchanged.

## Symptom

```
File ".../peft/utils/other.py", line 619, in update
    self.modules_to_save[adapter_name] = copy.deepcopy(self.original_module)
...
File ".../copy.py", line 259, in _reconstruct
    state = deepcopy(state, memo)
...
File ".../copyreg.py", line 99, in __newobj__
    return cls.__new__(cls, *args)
TypeError: LigerPlugin.pre_model_load.<locals>._LigerGemma4RMSNorm.__new__()
    missing 1 required positional argument: 'dim'
```

Stack: PEFT's `ModulesToSaveWrapper.update` →
`copy.deepcopy(vision_tower)` (creating a full-precision trainable copy)
→ recursive deepcopy descends into every `Gemma4RMSNorm` instance in the
vision tower → pickle protocol calls `cls.__new__(cls)` with no args →
crash.

## Repro (minimal)

```yaml
base_model: google/gemma-4-E2B-it      # or any Gemma 4 VLM
adapter: lora
processor_type: AutoProcessor
chat_template: gemma4
multimodal: true
freeze_mm_modules: false

plugins:
  - axolotl.integrations.liger.LigerPlugin
liger_rms_norm: true

lora_r: 16
lora_target_modules: 'model\.language_model\.layers\.[\d]+\.(mlp|self_attn)\.(up|down|gate|q|k|v|o)_proj'
lora_modules_to_save:
  - vision_tower            # any module containing Gemma4RMSNorm triggers it
```

## Root cause

`src/axolotl/integrations/liger/plugin.py` defines `_LigerGemma4RMSNorm`
as a nested class inside `LigerPlugin.pre_model_load`, overriding
`Gemma4RMSNorm` globally:

```python
class _LigerGemma4RMSNorm(LigerRMSNorm):
    def __new__(cls, dim, eps=1e-6, with_scale=True):   # dim required
        ...
    def __init__(self, dim, eps=1e-6, with_scale=True): # dim required
        ...
```

`dim` is required on both `__new__` and `__init__`. The pickle /
`copy.deepcopy` protocol reconstructs instances via
`copyreg.__newobj__(cls)`, which calls `cls.__new__(cls)` with an empty
`*args`. The Gemma3n and Gemma3 RMSNorm bindings don't hit this because
they aren't installed into a tree that PEFT deepcopies — it's the
`modules_to_save` + vision-tower-containing-many-RMSNorms combination
that makes it appear.

## Fix

Make `dim` optional and short-circuit `__init__` when called with the
pickle-protocol empty signature:

```python
def __new__(cls, dim=None, eps=1e-6, with_scale=True):
    if not with_scale:
        return _OrigGemma4RMSNorm(dim, eps, with_scale=False)
    return super().__new__(cls)

def __init__(self, dim=None, eps=1e-6, with_scale=True):
    if dim is None or not with_scale:
        return
    super().__init__(dim, eps, offset=0.0, casting_mode="llama", in_place=False)
```

Why this is safe:

- Normal construction (from
  `modeling_gemma4.Gemma4RMSNorm(dim, eps, with_scale)`) still passes
  `dim` explicitly, so runtime behavior is unchanged.
- `copy.deepcopy` restores instance state via `__setstate__` / direct
  `__dict__` assignment after `__new__`, so the no-op `__init__` path
  doesn't strand any uninitialized attributes.
- `LigerRMSNorm.__init__` is skipped only when `dim is None`; all of its
  parameters are restored by deepcopy's state transfer.
- The `with_scale=False` fast path still delegates to the stock HF
  `Gemma4RMSNorm` exactly as before.

## Files

- `src/axolotl/integrations/liger/plugin.py` (+17 / -5)
- `tests/integrations/test_liger_gemma4_rmsnorm.py` (new, +118)

## Tests

Seven regression tests in
`tests/integrations/test_liger_gemma4_rmsnorm.py` exercise the Liger
plugin end-to-end via `LigerPlugin.pre_model_load` and lock the deepcopy
invariant in place:

- `test_new_accepts_no_args` — `cls.__new__(cls)` without `dim` works.
- `test_init_no_args_is_noop` — `__init__()` short-circuits when `dim`
  is missing.
- `test_deepcopy_bare_instance` — direct repro of the original crash.
- `test_deepcopy_preserves_state` — weight tensor round-trips through
  deepcopy.
- `test_normal_construction_unchanged` — keyword/positional construction
  still initializes fully.
- `test_with_scale_false_delegates_to_original` — the `with_scale=False`
  fast path still returns the stock HF class.
- `test_deepcopy_of_containing_module` — deepcopy of a wrapper module
  containing nested RMSNorms (the real PEFT call pattern) succeeds.

```
tests/integrations/test_liger_gemma4_rmsnorm.py::test_new_accepts_no_args PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_init_no_args_is_noop PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_deepcopy_bare_instance PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_deepcopy_preserves_state PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_normal_construction_unchanged PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_with_scale_false_delegates_to_original PASSED
tests/integrations/test_liger_gemma4_rmsnorm.py::test_deepcopy_of_containing_module PASSED
```

## Related

A separate PR
([`fix/peft-modules-to-save-kwargs`](../../pulls?q=is%3Apr+head%3Afix%2Fpeft-modules-to-save-kwargs))
addresses a second crash in the same workflow: PEFT's
`ModulesToSaveWrapper.forward` requires a positional `x`, which breaks
Gemma 4's kwargs-only `vision_tower(pixel_values=...)` /
`embed_vision(inputs_embeds=...)` calls at the first forward pass. The
two fixes are independent — each can land on its own — but a Gemma 4 VLM
full-FT vision workflow needs both.

## Checklist

- [x] Fix is minimal and scoped to the Liger Gemma 4 RMSNorm replacement
- [x] Backward compatible: normal construction unchanged
- [x] Test coverage for deepcopy, state preservation, normal
      construction, and the `with_scale=False` fast path
- [x] No public API changes
- [x] Repro config included above
