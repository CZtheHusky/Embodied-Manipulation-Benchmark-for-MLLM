# API model control

The MLLM agent closes the benchmark loop through an OpenAI-compatible Chat
Completions endpoint:

~~~text
public visual history -> /chat/completions -> joystick JSON
                      -> parser -> validator -> controller -> simulator
~~~

The production example is 'configs/reacher_delta_eef_mllm.yaml'.

## Public input boundary

The request projection admits only task instruction text, selected public RGB
frames, the public joystick schema and control instructions, and the model's own
normalized action history. It may also include a public recovery marker when an
invalid output or API error caused a noop.

It does not serialize observation state, target coordinates, distance, reward,
success-oracle data, simulator actions, transition info, seeds, camera
randomization values, or other simulator metadata. Those values remain
available to evaluators and debug logs, but not to the model client.

The fixed prompt version is 'mllm_joystick_v1'. Static system text, task text,
control instructions, and schema are placed before episode turns. Request IDs,
timestamps, and other per-call values are not inserted into this prefix.

## Output and recovery

The model must return only:

~~~json
{"channels":{"joystick_0":0,"joystick_1":1}}
~~~

The existing parser and validator remain authoritative. Invalid JSON or an
invalid action causes one noop and does not trigger another API call. Timeouts,
network errors, HTTP 429, and HTTP 5xx are retried. HTTP 401 and other
non-retryable 4xx responses fail immediately.

A failed API call causes a noop. Three consecutive failed calls terminate the
episode with 'termination_reason=agent_error'. A local projection error,
including an image or byte safety-limit violation, terminates immediately with
the same reason.

## History policy

'action_window' and 'image_window' accept a non-negative integer or 'all'.

- 'image_window=0' sends no historical images and always sends the current frame.
- 'image_window=N' sends the previous N frames plus the current frame.
- 'image_window=all' sends every episode frame plus the current frame.
- 'action_window=all' retains every normalized action in chronological order.

Omitted historical images use '[image omitted by history policy]'; action text
is retained independently. The 'all' mode is strict: exceeding
'max_images_per_request' or 'max_request_bytes' fails instead of truncating.

Frames are deterministically normalized to uint8 RGB, resized without changing
aspect ratio, and PNG encoded once. Historical requests reuse identical PNG
bytes and SHA-256. In 'all' mode, each request appends turns without rewriting
the preceding canonical message stream.

## Artifacts and metrics

Each episode stores:

~~~text
model_calls/
  images/step_NNNN/<camera>.png
  step_NNNN/request.json
  step_NNNN/response.json
  step_NNNN/error.json
~~~

'request.json' contains canonical request metadata and image hashes, but not
Authorization or base64 image data. The actual PNG is stored separately. The
trajectory records model, prompt version, latency, attempts, tokens, request
bytes, image count, artifact directory, and adjacent-request canonical
common-prefix bytes.

When the endpoint reports cached tokens, traces record 'cached_input_tokens',
'cache_hit_ratio', and 'cache_metrics_status=reported'. Otherwise the status is
'unavailable'. 'common_prefix_bytes' is only a client-structure diagnostic; it
is not evidence of a server cache hit.

## Human review video

Automated MLLM episodes can generate an offline review video:

~~~yaml
logging:
  save_video: true
  video:
    fps: 2
    step_hold_seconds: 1
    terminal_hold_seconds: 2
    include_evaluator_debug: false
~~~

This first version supports only `agent.type=mllm`; it is not part of the human
Pygame path. Each decision frame displays the exact public PNGs referenced by
that step's request manifest, followed by the action that was produced from
them. The final frame is the public post-action terminal observation.

`review/model_view.mp4` contains only task text, public controls/mapping, model
raw and parsed actions, validation/noop status, and API latency/attempt/token
and cache fields. It never receives state, reward, distance, target, seed,
transition info, or simulator action.

Setting `include_evaluator_debug: true` additionally creates
`review/evaluator_debug.mp4`. That video is permanently labeled `EVALUATOR
ONLY` and may show only reward, distance, success, target/EEF position, and
simulator action. Do not use it as model input.

Both videos are fixed at 1280x720 and encoded with the system FFmpeg using
H.264, `yuv420p`, and no audio. Pillow is a package dependency; FFmpeg is a host
dependency. The review directory also contains the terminal PNGs,
`manifest.json`, and, on failure, `video_error.json`. A video failure produces a
warning and a failed status in `summary.json`, while preserving the episode
result and all raw artifacts.

## Endpoint cache behavior

OpenAI prompt caching requires exact prompt-prefix matches, including identical
images. Chat Completions reports cache reads under
'usage.prompt_tokens_details.cached_tokens'. See the
[OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching).

The optional 'agent.model.prompt_cache_key' is sent only when configured for an
endpoint known to support it. It is omitted by default because generic
compatible servers may reject unknown fields.

For vLLM, verify prefix caching remains enabled for the deployed version and
model. vLLM v0.19 documents block hashes over token prefixes plus multimodal
input hashes, so stable image bytes matter. See
[vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.19.1/design/prefix_caching/)
and [vLLM cache configuration](https://docs.vllm.ai/en/v0.19.1/api/vllm/config/cache/).

SGLang's RadixAttention cache must likewise remain enabled; do not launch with
'--disable-radix-cache'. HiCache is an optional larger hierarchical cache. See
the [SGLang documentation](https://docs.sglang.io/) and
[HiCache best practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices).

## Reacher smoke

Set endpoint configuration without putting credentials in YAML:

~~~bash
set -a
source .env.model-api.local
set +a
~~~

The repository includes `.env.model-api.example`. Put the real key only in
`.env.model-api.local`; that exact filename is gitignored. Do not put the key in
the YAML config, shell command arguments, or benchmark output directory.

Run the isolated five-step baseline with no historical images:

~~~bash
python3 scripts/manage_uv_env.py run gymnasium -- \
  python scripts/run_benchmark.py \
  --config configs/reacher_delta_eef_mllm.yaml \
  --set task.max_steps=5 \
  --set agent.history.image_window=0 \
  --set logging.save_video=true \
  --set logging.output_dir=outputs/reacher_api_smoke_image0
~~~

Acceptance requires successful API exchanges or earlier normal simulator
termination, valid joystick actions, normal simulator steps, complete artifacts,
an H.264 1280x720 2 FPS review video with no audio, and no credential material
in logs. Task success is not required.

After the baseline, repeat with 'agent.history.image_window=2' and 'all' in
separate output directories. Compare image counts, request bytes, latency, input
tokens, cached tokens, and cache ratio. A zero cache hit is valid. If the server
does not expose cache usage, report 'unavailable', not zero.
