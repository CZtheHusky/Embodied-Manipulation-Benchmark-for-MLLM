# API model control

The MLLM agent closes the benchmark loop through either OpenAI-compatible Chat
Completions or Responses:

~~~text
public visual context -> /chat/completions or /responses -> joystick JSON
                                                    -> parser -> validator
                                                    -> controller -> simulator
~~~

The production example is 'configs/reacher_delta_eef_mllm.yaml'.

## Public input boundary

The request projection admits only task instruction text, selected public RGB
frames, the public joystick schema and control instructions, and the model's own
action history. It may also include a public recovery marker when an invalid
model output caused a noop.

It does not serialize observation state, target coordinates, distance, reward,
success-oracle data, simulator actions, transition info, seeds, camera
randomization values, or other simulator metadata. Those values remain
available to evaluators and debug logs, but not to the model client.

Strict Chat Completions uses prompt version `mllm_joystick_v2`. Reflection
Chat Completions uses `mllm_joystick_reflection_v2`. Responses uses
`mllm_joystick_responses_v2`. Static system text, task text, control
instructions, and schema are placed before episode turns. Request IDs,
timestamps, and other per-call values are not inserted into prompt text.

## Reasoning effort

`agent.model.reasoning_effort` accepts `none`, `low`, `medium`, `high`,
`xhigh`, or `max`. Omit it only when the endpoint default is intentionally part
of the experiment. Chat Completions serializes the value as top-level
`reasoning_effort`; Responses serializes it as
`reasoning: {"effort": "<value>"}`.

The value is persisted in the sanitized request artifact, every public agent
trace, agent metadata, the experiment directory name, and
`run_manifest.json`. The endpoint-reported model remains separately available
in per-step traces and is aggregated into `run_summary.json`; a requested alias
must not be treated as proof of the model that actually served the request.

## Output modes

`agent.action_output_mode` makes the response contract explicit. The default
`action_only` mode accepts only:

~~~json
{"channels":{"joystick_0":0,"joystick_1":1}}
~~~

Any additional top-level key is invalid in this mode.

`action_with_reflection` allows one optional, case-sensitive, top-level string
key named `Reflection`:

~~~json
{
  "channels": {"joystick_0": 0, "joystick_1": 1},
  "Reflection": "joystick_0 appears to control horizontal motion; joystick_1 appears to control vertical motion"
}
~~~

The prompt asks the model to summarize its current hypothesis about each
joystick channel using only public observations and its own action history.
`Reflection` is preserved in the trajectory and subsequent assistant history;
it does not affect action validation or simulator control. Other top-level keys
remain invalid.

The reflection mode deliberately requires Chat Completions,
`response_format=json_object`, and `image_window=3`. It sends the previous three
images plus the current image while retaining all normalized action and
Reflection text history. Responses mode is restricted to `action_only`.

Switch between the two checked-in contracts by selecting the config:

~~~bash
# Strict action-only output
python scripts/run_benchmark.py --config configs/reacher_delta_eef_mllm.yaml

# Action plus optional Reflection, Chat Completions, three historical images
python scripts/run_benchmark.py --config configs/reacher_delta_eef_mllm_reflection.yaml
~~~

Reflection runs add `output-action_with_reflection` to the experiment group
name and record the mode in `run_manifest.json`. Strict runs keep their existing
directory naming for backward compatibility.

## Output recovery

The existing parser and validator remain authoritative. Invalid JSON or an
invalid action causes one noop and does not trigger another API call. Timeouts,
network errors, HTTP 429, and HTTP 5xx are retried. HTTP 401 and other
non-retryable 4xx responses fail immediately.

API failures never produce a simulator action and never advance the environment.
Set `retry_until_success=true` to keep retrying transient failures against the
same observation and exact request body. Exponential retry delay starts at
`retry_backoff_seconds` and is capped by `retry_backoff_max_seconds`. This mode
can wait indefinitely until the endpoint recovers or the process is interrupted.

With `retry_until_success=false`, transient failures stop after `max_retries`;
the episode then terminates with `termination_reason=agent_error` before
`sim.step`. Non-retryable API failures and local projection errors, including an
image or byte safety-limit violation, terminate immediately with the same
no-step behavior.

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

### Responses full-context mode

Set `agent.model.api=responses` to use `/responses`. This mode requires both
history windows to be `all`; fixed windows are rejected because they conflict
with provider-side retained context.

The first call sends the instructions and current public observation. Each
successful response ID becomes the next call's `previous_response_id`, so later
calls normally send only the current observation while the endpoint retains all
earlier public images and model outputs. `instructions` are resent on every call
because the Responses API does not inherit them through
`previous_response_id`. Requests set `store=true`, so this mode requires an
endpoint whose storage and retention policy is acceptable for the experiment.

Responses accepts both ordinary JSON and SSE responses. Set
`agent.model.stream=true` for endpoints that require SSE; the client collects
the complete stream, preserves decoded events in `response.json`, and acts only
after `response.completed`. Streaming therefore does not add intra-step actions
or change the synchronous control loop.

An API failure produces no response ID and therefore cannot enter the chain.
Transient retries reuse the exact request, current image bytes, and
`previous_response_id` while the simulator remains paused. A non-retryable or
retry-exhausted failure terminates the episode, so no synthetic API-error turn
is added. A successful response containing an invalid action is already in the
chain; the next request adds only an `invalid_action_noop` execution-feedback
record before the current observation. Private parser and endpoint error details
are never sent.

`max_images_per_request` is enforced against the complete retained
conversation, even though only incremental images appear in later HTTP bodies.
`max_request_bytes` applies to each actual incremental request body. Provider
context-window exhaustion is reported as an API error and is never handled by
silently dropping old turns.

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
trajectory records model, API type, prompt version, latency, attempts, tokens,
request bytes, image count, artifact directory, response-chain IDs, and
adjacent-request canonical common-prefix bytes.

Each MLLM benchmark invocation uses:

~~~text
<logging.output_dir>/
  model-<requested>__reasoning-<effort>__api-<api>__prompt-<version>__history-<policy>__episodes-<N>__maxsteps-<N>/
    <timestamp>-<uuid>/
      run_manifest.json
      run_summary.json
      episode_XXXX/
~~~

For Chat Completions, the group name records the prompt version plus image and
action history windows. For Responses, it records the Responses prompt version
and `history-full`, because both windows must be `all` and retained context is
chained through `previous_response_id`. The manifest
contains the full key configuration without credentials. The run summary
contains aggregate metrics, episode status, and endpoint-reported model names.

When the endpoint reports cached tokens, traces record 'cached_input_tokens',
'cache_hit_ratio', and 'cache_metrics_status=reported'. Otherwise the status is
'unavailable'. 'common_prefix_bytes' is only a client-structure diagnostic; it
is not evidence of a server cache hit.

Audit a completed run before counting it as a benchmark result:

~~~bash
python3 scripts/manage_uv_env.py run core -- \
  python scripts/audit_mllm_run.py <run-directory>
~~~

The audit checks requested versus summarized episodes, step validation and
agent trace status, retries, request/response presence, model-input PNG hashes
and black frames, prompt privacy, credential/base64 leaks, and the required
H.264 1280x720 2 FPS no-audio review videos. It prints one JSON report and exits
nonzero for `fail`, `incomplete`, or `unavailable`. Retries are reported but do
not by themselves fail an otherwise valid run.

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

## Prompt and conversation review UI

Use the read-only local review UI for exact prompt inspection:

~~~bash
python3 scripts/manage_uv_env.py run core -- \
  python scripts/review_episode.py \
  outputs/reacher_delta_eef_mllm
~~~

Open `http://127.0.0.1:8765`. Passing a log root recursively discovers
reviewable runs and exposes experiment group, run, episode, and step selectors
in one page. Normal experiment groups are listed before archive, invalid, and
interrupted groups. Catalog startup reads only lightweight run metadata; the
selected run's episodes and steps are loaded on demand. A run directory or an
individual `episode_XXXX` directory can still be passed for focused review.
Interrupted episodes that contain request artifacts but no `summary.json` are
listed as `incomplete`; completed request/response pairs remain inspectable.

The request snapshot view renders every system/user/assistant message in the
selected request, resolves image hashes to the exact persisted model-visible
PNGs, and appends the sanitized provider response. Hidden reasoning fields such
as `reasoning_content` and Responses reasoning items are replaced with an
explicit redaction marker before the artifact is written; aggregate token
counts remain available. The episode conversation view
removes repeated request prefixes and shows each observation followed by its
response over time. For Responses runs, the raw inspector shows the incremental
request body while the conversation view reconstructs the retained chain from
all step artifacts.

The JSON inspector exposes the sanitized request, sanitized response, and public
agent trace. The server does not expose arbitrary files: image routes are
restricted to PNG artifacts named in request manifests, and evaluator
state/info are not returned by its API.

`mllm_joystick_v2`, `mllm_joystick_reflection_v2`, and
`mllm_joystick_responses_v2` are prompt contract versions, not model versions.
They identify the system instruction, public-input projection, history layout,
image policy, and joystick response schema. A behavior-changing prompt update
must use a new version so results remain attributable and comparable.
Existing v1 run artifacts remain immutable and the review tooling continues to
display the prompt version recorded by each historical run.

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
  --set logging.save_video=true
~~~

Acceptance requires successful API exchanges or earlier normal simulator
termination, valid joystick actions, normal simulator steps, complete artifacts,
an H.264 1280x720 2 FPS review video with no audio, and no credential material
in logs. Task success is not required.

After the baseline, repeat with 'agent.history.image_window=2' and 'all'. Each
invocation creates a separate timestamped run directory under the configured
experiment output root, and its effective settings are preserved in each
episode's `config.yaml`. Compare image counts, request bytes, latency, input
tokens, cached tokens, and cache ratio. A zero cache hit is valid. If the server
does not expose cache usage, report 'unavailable', not zero.

To reproduce selected episodes without rerunning earlier ones, repeat
`--episode-id` as needed:

~~~bash
python scripts/run_benchmark.py \
  --config path/to/original/episode_0002/config.yaml \
  --episode-id 2 \
  --episode-id 8
~~~

The selected ID is preserved in the log path and derives the same simulator,
camera, mapping, and agent seed as a full run: `base_seed + episode_id`.
Recreating an episode does not require sequential resets.

Run a five-step full-context Responses smoke with:

~~~bash
python3 scripts/manage_uv_env.py run gymnasium -- \
  python scripts/run_benchmark.py \
  --config configs/reacher_delta_eef_mllm.yaml \
  --set task.max_steps=5 \
  --set agent.model.api=responses \
  --set agent.model.stream=true \
  --set agent.history.action_window=all \
  --set agent.history.image_window=all \
  --set logging.save_video=true
~~~

For step 1 and later, verify that each request's `previous_response_id` equals
the preceding successful `response_id`. A normal chained request contains only
one current image; the review conversation still contains every prior turn.
