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
name. A producer using different names needs a mapping added there, not a new reader.

**Outcome.** LeRobot rewards are usually shaped and continuous rather than a success flag, so
success is inferred: final reward `>= 1.0`. Same threshold as the RLDS parser, so both formats
agree. Override per dataset if yours encodes success differently.

### Legacy JSON is still supported

The `*.rlds.json` / `*.lerobot.json` encodings predate real-format support. They are **kept, not
deprecated-and-removed**: the seed script, the test fixtures, and any partner already posting
them keep working. New integrations should send the real format.

### Frame retention

Every frame of every episode is stored by default. `APERTURE_MAX_FRAMES_PER_EPISODE` caps it
(`0` = unlimited, the default); each truncation is logged at WARNING. A silently shortened
trajectory would teach a fine-tune that the task ends early, so truncation is never quiet.

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
