#!/usr/bin/env python3
"""View chains as repeated noisy readings of one latent type: estimate the labeller's confusion matrix
q[j,k] = P(label j | true k) and the class prior by EM over chains (Dawid-Skene with exchangeable views),
then decode each chain by MAP and compare with the majority vote.  Checks: (1) the confusion is one-sided at the
quantiser's ends (shot can only be over-read, high only under-read), (2) MAP disagrees with majority mostly in
short chains with a noisy minority class, (3) the decoded shot:whisky ratio moves toward the manual test value."""
import json, sys, numpy as np
from collections import defaultdict, Counter
from pathlib import Path
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path.insert(0, str(P / "figures/labelnoise_v1")); sys.path.insert(0, str(HERE))
import build_source as B
NAMES = ["shot", "whisky", "water", "beer", "wine", "high"]

pairs, stats = B.match_split("train", registered=True)
by, stem_of = B.load_split("train")
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for s in by:
    for f in by[s]:
        for i in range(len(by[s][f])): find((s, f, i))
for p in pairs:
    a, b = find((p["scene"], p["frame_a"], p["box_index_a"])), find((p["scene"], p["frame_b"], p["box_index_b"]))
    if a != b: parent[a] = b
chains = defaultdict(list)
for s in by:
    for f in by[s]:
        for i, bx in enumerate(by[s][f]): chains[find((s, f, i))].append((s, f, i, int(bx[0])))
chains = list(chains.values())
lens = Counter(len(c) for c in chains)
print("chains", len(chains), "boxes", sum(len(c) for c in chains), "length histogram", sorted(lens.items())[:12])

# EM (Dawid-Skene, one confusion shared by all views)
K = 6; counts = np.array([[sum(1 for *_, l in c if l == j) for j in range(K)] for c in chains], float)
q = np.full((K, K), 0.04) + np.eye(K) * (0.80 - 0.04); pi = np.full(K, 1 / K)
for it in range(500):
    logp = np.log(pi)[None, :] + counts @ np.log(q)          # items x k  (counts[i,j] * log q[j,k])
    logp -= logp.max(1, keepdims=True); post = np.exp(logp); post /= post.sum(1, keepdims=True)
    pi_new = post.mean(0); q_new = (counts.T @ post) + 1e-3   # j x k
    q_new /= q_new.sum(0, keepdims=True)
    if np.abs(q_new - q).max() < 1e-7 and np.abs(pi_new - pi).max() < 1e-7: q, pi = q_new, pi_new; break
    q, pi = q_new, pi_new
print(f"EM iterations {it+1}")
print("confusion q[label j | true k], rows = label, cols = true class (%):")
for j in range(K): print(f"  {NAMES[j]:7s}", " ".join(f"{100*q[j,k]:5.1f}" for k in range(K)))
print("prior pi (%):", np.round(100 * pi, 1), " shot:whisky true", round(float(pi[0] / pi[1]), 2))

# decode
maj, mapd = [], []
for c, po in zip(chains, post):
    cnt = Counter(l for *_, l in c); top = cnt.most_common()
    m = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else None   # None = tie, majority leaves as is
    k = int(np.argmax(po)); maj.append(m); mapd.append(k)
boxes_maj = sum(len(c) for c, m in zip(chains, maj) if m is not None)
diff_chains = [(c, m, k, po) for c, m, k, po in zip(chains, maj, mapd, post) if (m is not None and m != k)]
tie_chains = [(c, k, po) for c, m, k, po in zip(chains, maj, mapd, post) if m is None]
print(f"chains where MAP != majority: {len(diff_chains)} ({sum(len(c) for c,*_ in diff_chains)} boxes); tied chains decoded by MAP: {len(tie_chains)} ({sum(len(c) for c,*_ in tie_chains)} boxes)")
dc = Counter((NAMES[m], NAMES[k]) for c, m, k, po in diff_chains)
print("majority -> MAP (chains):", dc.most_common(8))
print("length of disagreeing chains:", sorted(Counter(len(c) for c, *_ in diff_chains).items())[:8])
conf_dis = np.array([po.max() for *_, po in diff_chains]); print("posterior of MAP class in disagreeing chains: median", round(float(np.median(conf_dis)), 3), "min", round(float(conf_dis.min()), 3))
# label shares: raw, majority-relabelled, MAP-relabelled
def shares(labels): c = Counter(labels); n = sum(c.values()); return np.array([100 * c[j] / n for j in range(K)])
raw = shares([l for c in chains for *_, l in c])
majl = shares([(m if m is not None else l) for c, m in zip(chains, maj) for *_, l in c])
mapl = shares([k for c, k in zip(chains, mapd) for _ in c])
for name, sh in (("raw labels", raw), ("majority vote", majl), ("MAP decode", mapl)):
    print(f"{name:14s} shares {np.round(sh,1)}  shot:whisky {sh[0]/sh[1]:.2f}")
print("manual test   shares [18.9 13.5 17.8 21.6 11.8 16.4]  shot:whisky 1.40")
changed_map = sum(1 for c, k in zip(chains, mapd) for *_, l in c if l != k)
changed_maj = sum(1 for c, m in zip(chains, maj) for *_, l in c if m is not None and l != m)
print(f"boxes changed: majority {changed_maj}, MAP {changed_map}")
json.dump({"confusion_label_given_true": q.tolist(), "prior": pi.tolist(), "chains": len(chains), "length_hist": dict(sorted(lens.items())),
           "map_vs_majority_chains": len(diff_chains), "directions": [[a, b, n] for (a, b), n in dc.most_common()],
           "shares": {"raw": raw.tolist(), "majority": majl.tolist(), "map": mapl.tolist()},
           "changed_boxes": {"majority": changed_maj, "map": changed_map}}, open(HERE / "CHAIN_CONFUSION_EM_V1.json", "w"), indent=1)
