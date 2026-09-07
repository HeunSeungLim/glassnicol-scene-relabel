# Errata to PREREGISTRATION_SCENEVOTE_V1.md

The sealed file is unchanged; its sha256 is 4788e631... (full value in PREREGISTRATION_SCENEVOTE_V1.sha256).
The sha256 of this errata file is in PREREGISTRATION_SCENEVOTE_V1_ERRATA.sha256.

## What is wrong

The clock strings typed inside the sealed file ("Written 2026-09-02 21:20 KST", addendum "21:45 KST") are wrong:
they were written from memory of the wall clock and overstate the time. Nothing in the predictions or in the
decision rule was changed after sealing; only the human-typed times inside the text are off.

## File-system evidence (KST, 2026-09-02)

Written on this machine, so the timestamps are local and directly checkable:

    21:08:33  PREREGISTRATION_SCENEVOTE_V1.md            sealed (mtime)
    21:09:21  labels_scene.tgz                           relabelled training labels (hops 2) built
    21:10:10  seed_v1/runs/scene136_scenevote_s1337/CONTRACT.json   first scene-vote training contract

Order on disk: preregistration sealed -> labels built -> first training started.

## The H200 runs

The three H200 runs (scene_scenevote_s1337 / s3407 / s4567) were executed on a remote host and copied here
afterwards, so their local mtimes (21:51:53, 21:51:56, 21:51:59) are copy times, not start times, and the remote
mtimes are not part of this release. A locally checkable upper bound on each start time follows from the copy time
minus the recorded run length in FINAL_RESULT.json:

    s1337   copy 21:51:53 - elapsed 2373.5 s  ->  started no later than 21:12:19
    s3407   copy 21:51:56 - elapsed 2338.5 s  ->  started no later than 21:12:57
    s4567   copy 21:51:59 - elapsed 2414.9 s  ->  started no later than 21:11:44

All three bounds are after the 21:08:33 seal, so the preregistration precedes every scene-vote training run by the
local evidence alone.

## One misleading field in the contracts

Every CONTRACT.json carries "host": "h200"; that string is a default in the runner and does not identify the
machine. The accelerator is identified by "gpu_uuid" (and "gpu_name" where the runner recorded it):

    scene136_*   GPU-6a366503-1e37-3338-48b6-c2b77752c0ec   RTX PRO 6000 Blackwell   (accelerator B)
    scene_s1337  GPU-1490fe36-d37a-cc91-63b8-76a472c106bf   NVIDIA H200              (accelerator A)
    scene_s3407  GPU-cf8e1cde-198c-2542-...                 NVIDIA H200              (accelerator A)
    scene_s4567  GPU-bd719d5f-947f-7e4d-...                 NVIDIA H200              (accelerator A)

The three H200 runs used three distinct GPUs. The paper's A/B assignment follows the uuids, not the "host" field.
