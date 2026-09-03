# Multi-view type-consistent relabelling — preregistration

Written before any arm trained on the relabelled set exists. The relabelled
label files are fixed (labels_relabelled/, produced by relabel_multiview_v1.py
at IoU 0.5 with background-registration-compensated linking) and their sha256
is recorded below.

## Why

GlassNICOL assigns glass type by matching a depth-derived height to the nearest
known glass. On adjacent views of the same scene, 20.3% of boxes that are the
same physical object (IoU > 0.5 after a small neck rotation) carry a different
type, and the flips are between glasses of similar height. That is per-view
measurement noise on a property that is constant for the object.

Everything measured today is consistent with training on that noise: subtype AP
that is seed-dependent (7/25, 18/25, 15/25 scenes across three seeds of one
configuration), presence AP that is stable, and a checkpoint-selection
criterion that does not predict the reported metric (r^2 = 0.003).

## Method

Every object is seen in up to 25 views. Consecutive frames are registered by
background features (same construction as the pair-eligibility test), boxes are
carried through the registration and chained by overlap, and each chain takes
its majority type. Ties are left as they were. Only the type id changes: no box
moves, no image is added or removed, the repeat roster is identical.

Relabelled: 364 of 4,336 training boxes (8.4%), 25 of 263 validation boxes.

## Design

Two arms, identical in every respect except the type ids in the training labels:

    raw          scene_transparent_s{1337,3407,4567}   already trained
    relabelled   same roster, same 70 scenes, same budget, relabelled types

Checkpoint selection stays as it was (validation generic AP), so selection is
identical across arms; type ids do not enter it.

Test: the OFFICIAL standard and OOD splits, whose labels were drawn by hand.
This is the point of the design. If relabelling merely made the training set
self-consistent, it would not move agreement with independent manual labels.

## Predictions

P1  subtype accuracy on localised boxes (IoU 0.5, conf 0.25) is higher for
    relabelled than raw, in every seed, on both official splits.
P2  six-class AP is higher for relabelled in every seed on both splits.
P3  the seed-to-seed spread of subtype accuracy is smaller for relabelled.
P4  localisation recall changes by less than subtype accuracy does.

## What refutes this

R1  any seed-split cell where relabelled subtype accuracy is not higher.
R2  the manual labels agree less with relabelled predictions than with raw ones
    on the confusable pairs (shot/whisky, whisky/water, water/beer, wine/high).
    Then the majority vote is imposing a convention the annotators do not share.

A refuted result is reported as the outcome.
