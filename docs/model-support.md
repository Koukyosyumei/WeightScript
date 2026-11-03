# Model adapter support

Semantic extraction executes tensor operations independently of the model's native
`forward`. Generic raw tensor transactions are available separately; they do not
constitute semantic extraction for an unsupported architecture.

| Family | Semantic adapter | RoPE | Attention |
| --- | --- | --- | --- |
| GPT-2 | Existing standard GPT2LMHeadModel | Learned positions | Causal multi-head; CPU float32 |
| Llama | Standard LlamaForCausalLM | Default, linear, dynamic NTK, Llama-3 frequency scaling | Causal GQA |
| Qwen2 | Standard Qwen2ForCausalLM | Default, linear, dynamic NTK, Llama-3 frequency scaling | Causal GQA; configured full/sliding layer types |
| Mistral | Standard MistralForCausalLM | Default, linear, dynamic NTK, Llama-3 frequency scaling | Causal GQA; sliding window or full attention |
| Qwen3.5 / hybrid decoders | Unsupported | Unsupported | No hybrid semantic adapter |

The rotary adapters accept homogeneous float32, float16, or bfloat16 weights on
one CPU or CUDA device, in evaluation mode. Quantization, distributed/device-map
execution, partial rotary dimensions, custom decoder subclasses, and other RoPE
variants (including YaRN and LongRoPE) are rejected. GPT-2 retains its existing
CPU float32 restrictions. `load_model(source, device="cpu", dtype=torch.float32)`
retains its defaults and can explicitly load the rotary families in other supported
precisions/devices. No dependency upgrade or remote custom code is required.

Inputs are nonempty rank-two integer token tensors on the model device. Optional
masks must be binary, match the token shape, and share its device. Execution uses
full-sequence causal inference, starting positions at zero, with no KV cache or
custom position IDs. Left and right padding use absolute sequence positions,
matching native forward without explicit position IDs. Sliding attention permits
keys whose distance from a query is less than the configured window. Qwen2 mixed
layers follow `config.layer_types`, including its generated defaults. In the pinned
Transformers version Llama native forward does not implement sliding windows;
windowed Llama configurations are explicitly rejected rather than misrepresented.

Rotary scaling is computed from configuration with independent formulas. Dynamic
NTK uses the current full sequence length and resets naturally on shorter inputs.
Transformers maintains a mutable rotary frequency cache: after a long invocation,
an intermediate length above the original context can reuse the earlier longer
length's frequencies. Extracted execution is stateless and does not reproduce that
history-dependent cache reuse. Native comparisons should use fresh native rotary
state per sequence length, or lengths that grow or reset below the original context.
Mistral dynamic NTK in Transformers 4.57.1 requires an explicit numeric `head_dim`;
its default `None` fails in the native rotary initializer. Tests supply that field.

## Numerical evidence

Generated tiny models exercise all three rotary families, four rotary variants,
GQA (four query heads/two KV heads), native and edited forward, padding masks,
float32/float16/bfloat16, dynamic context expansion/reset, Qwen2 mixed windows,
and Mistral windows. Native comparisons use eager attention. RMSNorm and softmax
accumulate in float32, then cast to the model dtype, as native implementations do.
The tests use absolute/relative tolerances of 2e-6 for float32, 8e-4 for float16,
and 6e-3 for bfloat16. These are tiny-model regression tolerances, not universal
error guarantees. Long contexts, different hardware, large weights, or different
native attention kernels require measured tolerances on the actual workload.
The existing evaluation helper's default float32 tolerances are unchanged.
CUDA has a conditional native comparison; a CPU-only test run does not validate it.

Checkpoint tests verify edited bfloat16 reloads, fingerprints, audits and embedding
weight ties for all three families. Existing float32 checkpoints retain the same
fingerprint byte encoding; bfloat16 hashes use raw contiguous bytes without an
unsupported NumPy bfloat16 conversion. Persistent weights/configuration determine
identity; transient rotary caches are excluded as before. No large downloads are
needed. These tests establish implementation equivalence, not model quality or
performance improvements over prior work.

The pinned Transformers 4.57.1 configuration mapping contains `qwen3_next` but no
`qwen3_5` entry. A different registered family is not evidence of Qwen3.5 support.
Hybrid gated/recurrent attention would need its own extraction semantics, state
handling, compiler, and independent tests. Neither dependency upgrades nor opaque
native-forward fallback are used to claim support.

## Atomic edits and storage costs

`raw.transaction(model, edits)` returns a fully isolated model by default. It supports
shape-preserving set/add/scale selections and low-rank matrix updates on real
floating tensors, including device-correct selectors and edit values. Failed edits,
overflow, and nonfinite results leave the source untouched. Exact tied aliases
receive cumulative edits on one copied storage and retain their ties. Audits count
that storage once. Integer/complex target tensors are rejected.

`raw.transaction_plan(model, edits)` reports unique storage bytes, bytes to copy in
sparse mode, shared bytes, and addressed tensor names. It includes buffers, including
nonpersistent caches. These are storage estimates; Python/module metadata,
allocator overhead, selected-value temporaries and low-rank products are additional.
Editing one scalar still copies its containing tensor, not only that scalar.

`raw.transaction(..., share_untouched=True)` and
`compile_edits(..., share_untouched=True)` copy edited parameter storage and all
buffers, while sharing untouched parameter storage. This opt-in mode avoids
allocating a full checkpoint copy for sparse edits. Both models' shared parameters
must be treated as read-only, including during training or later manual edits.
This is explicit structural sharing, not automatic copy-on-write interception of
arbitrary PyTorch writes. Use the default for unrestricted independent ownership.
Non-identical views on overlapping storage are rejected in sparse mode. Module
structure and configuration are copied; native rotary cache updates cannot mutate
the source buffers. Tests check tied aliases, accumulation, unaffected tensors,
buffer isolation, memory estimates, dtype/domain errors, and rollback after overflow.
