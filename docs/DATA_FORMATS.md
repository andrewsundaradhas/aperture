# Data formats — what Aperture ingests, and what it exports

## Ingestion

Two ways in, depending on size.

### Large datasets — upload straight to storage

A LeRobot v3 archive of real fleet video is gigabytes. Proxying that through the API means
holding it in memory to hand to the parser and paying for the bandwidth twice, so the API stays
out of the data path:

```bash
# 1. ask for somewhere to put it
curl -X POST $API/v1/uploads/presign -H "X-API-Key: $KEY" \
     -H 'Content-Type: application/json' -d '{"filename":"fleet.zip"}'
# -> {"key": "...", "upload_url": "...", "ingest_url": "/v1/uploads/ingest"}

# 2. PUT the bytes at upload_url — straight to R2/S3/MinIO, the API never sees them
curl -X PUT --data-binary @fleet.zip "$UPLOAD_URL"

# 3. queue the parse; returns immediately
curl -X POST $API/v1/uploads/ingest -H "X-API-Key: $KEY" \
     -H 'Content-Type: application/json' -d "{\"key\":\"$KEY_FROM_STEP_1\"}"
# -> 202 {"job_id": "...", "poll": "/v1/jobs/..."}

# 4. poll until done; result carries the episode ids
curl $API/v1/jobs/$JOB_ID -H "X-API-Key: $KEY"
```

On R2/S3 step 2 is a real presigned PUT. On the local filesystem backend there is nothing to
presign, so `PUT /v1/blobs/{key}` stands in — the client flow is identical either way. Either
way the key is confined to the caller's own prefix, so one tenant cannot write over or ingest
another's objects.

### Small files — direct upload

`POST /v1/episodes/upload` takes a batch of files and parses them inline. Convenient for single
episodes and the seeded fixtures; it buffers each file in memory, so use the presigned flow for
anything large.

**One file may contain many episodes**, on either path — so the number of episodes created is
not the number of files uploaded. Both return the full list of `episode_ids`.

| Upload | Detected by | Reader | Needs |
|---|---|---|---|
| **LeRobot v3 dataset** (`.zip` / `.tar.gz`) | `meta/info.json` inside the archive | [`lerobot_v3.py`](../apps/api/aperture/ingestion/lerobot_v3.py) | `[datasets]` extra |
| **RLDS / Open X-Embodiment** (`.tfrecord`, or shards inside an archive) | `.tfrecord` extension | [`rlds_tfrecord.py`](../apps/api/aperture/ingestion/rlds_tfrecord.py) | nothing — pure stdlib |
| **Legacy portable JSON** (`*.rlds.json`, `*.lerobot.json`) | filename | [`normalize.py`](../apps/api/aperture/ingestion/normalize.py) | nothing |

Archive contents are **sniffed, not trusted from the filename** — a `.zip` named anything is
inspected for `meta/info.json` first, then for `.tfrecord` shards. Members that escape the
extraction root, and symlink members, are rejected outright.

### Neither reader needs a heavyweight dependency

`tensorflow` is not required. The TFRecord container (length + CRC framing) and
`tf.train.Example` (a three-way `oneof` over bytes/float/int64 lists) are both small, frozen
formats, implemented directly in `rlds_tfrecord.py`. Likewise the LeRobot reader does not import
the `lerobot` package — it reads the v3 layout (parquet + mp4) with `pyarrow` and `av`.

**RLDS key layout.** TFDS flattens an episode's `steps` sequence into repeated values under
slashed keys. Keys are matched by *suffix*, so `steps/observation/state` and
`observation/state` both resolve. `KEY_HINTS` in `rlds_tfrecord.py` lists every recognised
name. Also handled: actions stored as named parts (`action/world_vector`,
`action/gripper_closedness_action`, … — concatenated per step), cameras under other names
(`hand_image`, `image_wrist`; depth images are skipped), and raw TFDS shard names
(`…-train.tfrecord-00000-of-00128`). Optional per-step signal channels `action_confidence`,
`contact_force` and `subgoal` feed the failure classifier.

**Cameras (LeRobot).** Multi-camera datasets use their first camera (`observation.image` if
present), so frames stay aligned one-to-one with the parquet rows. Videos are streamed and
downscaled, not decoded into memory, so real-robot datasets of hundreds of MB ingest in flat
memory.

**Outcome — did each episode succeed?** Most trustworthy signal first:

1. An explicit success flag — LeRobot `next.success` (an episode succeeded if it ever reached
   success); RLDS `success` / `is_success` / `episode_metadata/success`.
2. Reward — RLDS / Open X-Embodiment rewards are sparse 0/1 with 1 marking the success step,
   which is not always the final step, so any step reaching 1 counts. LeRobot rewards are
   usually shaped and continuous, so the final reward is compared against `1.0`.
3. Neither — the data records no outcome (typical of teleoperated demonstrations, e.g. real
   SO-100/SO-101 or ALOHA recordings), so episodes are recorded as successes.

Logs that do not record outcomes — a batch the operator knows to be failed runs, say — can be
labelled at upload: **Results in these files** on the Upload page, or `outcome=fail|success` on
`POST /v1/episodes/upload` (form field) and `POST /v1/uploads/ingest` (JSON). The before/after
check refuses files with no recorded outcome rather than measuring a success rate from defaults.

### Legacy JSON is still supported

The `*.rlds.json` / `*.lerobot.json` encodings predate real-format support. They are **kept, not
deprecated-and-removed**: the seed script, the test fixtures, and any partner already posting
them keep working. New integrations should send the real format.

### Frame retention

Every frame's state and action is stored. `APERTURE_MAX_FRAMES_PER_EPISODE` caps frames per
episode (`0` = unlimited, the default); each truncation is logged at WARNING — a silently
shortened trajectory would teach a fine-tune that the task ends early.

Frame **images** are stored for up to `APERTURE_MAX_IMAGES_PER_EPISODE` evenly spaced frames per
episode (default 32, first and last included; `0` = every frame), downscaled so the longest side
is 256 px. Classification reads the first frame and attention maps up to 8, so this keeps uploads
light without losing anything the product uses; in exports, frames without a stored image have
`image_url: null`.

### Size limits

Direct uploads (`POST /v1/episodes/upload`) are capped at `APERTURE_MAX_UPLOAD_BYTES` per file
(50 MB). Larger datasets go through the presigned path, capped at `APERTURE_MAX_INGEST_BYTES`
(5 GB); on the local-storage backend the bytes are streamed to disk, never held in memory.

## Export

`POST /v1/clusters/{id}/dataset-export` emits one JSON document. **Schema version 2.**

```jsonc
{
  "schema_version": 2,
  "format": "rlds",              // or "lerobot"
  "episode_count": 12,
  "readiness": {
    "total_frames": 240,
    "frames_with_action": 240,
    "frames_with_image": 240,
    "trainable": true,
    "note": "Frames carry actions; this export can be fine-tuned on."
  },
  "episodes": [
    { "episode_id": "…", "data": { /* one of the two shapes below */ } }
  ]
}
```

**`format: "rlds"`** — each episode is `{embodiment_type, policy_name, outcome, steps[]}`, and
each step:

```jsonc
{
  "observation": {
    "state": [0.4, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0],   // proprioceptive vector, or null
    "image_url": "/v1/blobs/…",                       // fetchable, or null
    "action_confidence": 0.92,
    "contact_force": 1.0
  },
  "action": [0.017, -0.003, 0.01, 0.0, 0.0, 0.0, 0.0], // the learning target
  "language_instruction": "pick up the red block",
  "subgoal": "reach",
  "is_terminal": false
}
```

**`format: "lerobot"`** — each episode is `{meta, success, frames[]}`, each frame a flat row
with dotted keys: `frame_index`, `action`, `observation.state`, `observation.image_url`,
`observation.confidence`, `observation.force`, `subtask`.

### Two properties worth knowing

**Actions are the point.** A policy learns from (observation, action) pairs. Everything else on
a frame is a diagnostic signal Aperture classifies on, not a learning target. `readiness` states
plainly whether the export can be fine-tuned on — an episode ingested before actions were
captured reports `trainable: false` rather than quietly handing over an unusable dataset.

**Images are referenced, not inlined.** A 50-episode cluster at 200 frames is 10,000 images;
base64-inlining them would produce a multi-gigabyte document. Each frame carries `image_url`
instead — an API-relative path under local storage, a presigned URL under R2. Fetch them with
your API key.

### Versioning

`schema_version` increments when the document's shape changes, so a customer's loader can
refuse a version it does not understand instead of misreading fields. Version 1 (actions and
images absent entirely) is superseded and not emitted.
