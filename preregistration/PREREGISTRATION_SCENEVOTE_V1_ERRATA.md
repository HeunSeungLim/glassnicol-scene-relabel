# Errata to PREREGISTRATION_SCENEVOTE_V1.md (sealed file unchanged; sha256 4788e631...)

The clock strings typed inside the sealed file ("Written 2026-09-02 21:20 KST", addendum "21:45 KST") are wrong:
they were written from memory of the wall clock and overstate the time. File-system evidence (mtime, KST):
PREREGISTRATION_SCENEVOTE_V1.md last modified 21:08:33; labels_scene (hops 2) written 21:09; the first scene-vote
training contract (seed_v1/runs/scene136_scenevote_s1337/CONTRACT.json) 21:10:10; H200 contracts 21:10-21:11.
The order "preregistration sealed -> labels built -> training started" holds on disk; only the human-typed times
inside the text are off. Nothing in the predictions or the rule was changed after sealing.
