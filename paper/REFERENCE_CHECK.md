# REFERENCE_CHECK — references.bib vs. main_v15.tex (2026-09-05)

Scope: all 37 entries in `paper_work_v4/references.bib`. `main_v15.tex` cites 25 of them
(`\cite{}` grep); the other 12 are in the .bib but never cited, so BibTeX/IEEEbib will not
print them. No file other than this report was modified.

Sources used: Crossref REST API (`api.crossref.org/works`, DOI records), DBLP search API +
DBLP bibtex view, arXiv export API (`export.arxiv.org/api/query`), OpenAlex (`api.openalex.org`),
PMLR page metadata, MLSys proceedings site, CVF Open Access bibtex blocks, Google Patents,
Ultralytics docs (YOLO11 citation block). Semantic Scholar was rate-limited and returned
nothing useful; DBLP went ECONNREFUSED midway, so a few DBLP look-ups were replaced by
Crossref/OpenAlex/CVF.

Overall result: no entry is fabricated or wrongly attributed. All 25 cited entries resolve to
a real publication with matching title, author list, year and venue. Page numbers all match an
official pagination (CVF Open Access or IEEE Xplore). Only cosmetic / consistency fixes remain.

---

## 1. Entries cited in main_v15.tex (25)

### gajdosech2025glassnicol — verified
- Source: Crossref DOI 10.1109/IROS60139.2025.11246715; DBLP `conf/iros/GajdosechAHMKW25` (bibtex view: pages 20516--20523, year 2025); OpenAlex.
- Record: "Shaken, Not Stirred: A Novel Dataset for Visual Understanding of Glasses in Human-Robot Bartending Tasks", Gajdošech, Ali, Habekost, Madaras, Kerzel, Wermter; 2025 IEEE/RSJ IROS (Hangzhou, 19–25 Oct 2025), pp. 20516–20523.
- Bib: title, 6 authors (order), venue, year, pages all match. Diacritics (Gajdošech) correct.
- Correction: none. Optional `doi={10.1109/IROS60139.2025.11246715}`.

### sapkota2025yolo — verified (minor title mismatch)
- Source: Crossref DOI 10.1007/s10462-025-11253-3 (issued 2025-06-11, article-number 274); DBLP `journals/air/SapkotaFQBNPZVKSYK25` (vol 58, no 9, "pages" 274).
- Record: Artif. Intell. Rev. 58(9), art. 274, 2025; 12 authors in the order Sapkota, Flores-Calero, Qureshi, Badgujar, Nepal, Poulose, Zeno, Vaddevolu, Khan, Shoman, Yan, Karkee — identical to bib.
- Mismatch: published title is "YOLO advances to its genesis: a decadal and comprehensive review of the You Only Look Once (YOLO) series" — bib drops the trailing "(YOLO)".
- Correction: add "({YOLO})" to the title (see fix list).

### lin2014coco — verified
- Source: Crossref DOI 10.1007/978-3-319-10602-1_48 (LNCS, Computer Vision – ECCV 2014, pp. 740–755, 2014); 8 authors match.
- Correction: none.

### sajjan2020cleargrasp — verified
- Source: Crossref DOI 10.1109/ICRA40945.2020.9197518 (ICRA 2020, pp. 3634–3642); DBLP CoRR abs/1910.02550 (title "ClearGrasp", author "Shreeyak S. Sajjan").
- Note: IEEE metadata spells the title "Clear Grasp"; the paper/arXiv/DBLP use "ClearGrasp". Bib is fine.
- Correction: none.

### rublee2011orb — verified
- Source: Crossref DOI 10.1109/ICCV.2011.6126544 (ICCV 2011, pp. 2564–2571), 4 authors match.
- Correction: none.

### fischler1981ransac — verified
- Source: Crossref DOI 10.1145/358669.358692; DBLP `journals/cacm/FischlerB81` — Commun. ACM 24(6):381–395, 1981.
- Correction: none.

### loshchilov2019adamw — verified
- Source: DBLP `conf/iclr/LoshchilovH19` (ICLR 2019). No pages exist (OpenReview). Correction: none.

### gupta2019lvis — verified
- Source: DBLP `conf/cvpr/GuptaDG19` pp. 5356–5364 (CVF pagination, = bib); Crossref DOI 10.1109/CVPR.2019.00550 gives IEEE pagination 5351–5359.
- Correction: none (bib follows CVF Open Access pagination, which is legitimate).

### jocher2024yolo11 — verified (software)
- Source: docs.ultralytics.com/models/yolo11 "Citations and Acknowledgements": `@software{yolo11_ultralytics, author={Glenn Jocher and Jing Qiu}, title={Ultralytics YOLO11}, version={11.0.0}, year={2024}, url={https://github.com/ultralytics/ultralytics}, license={AGPL-3.0}}`; release date 2024-09-10; DOI "pending".
- Bib matches the vendor-recommended citation (authors, title, year, URL).
- Correction: none. Optional: add "version 11.0.0" in the note.

### paul2021datadiet — verified
- Source: DBLP `conf/nips/PaulGD21` (NeurIPS 2021, pp. 20596–20607); arXiv 2107.07075. Authors match.
- Correction: none. Optional `volume={34}, pages={20596--20607}`.

### sorscher2022pruning — verified
- Source: Crossref DOI 10.52202/068431-1419 (Advances in NeurIPS 35, pp. 19523–19536, 2022); DBLP `conf/nips/SorscherGSGM22`. Authors match ("Ari S. Morcos" as on arXiv/DBLP CoRR).
- Correction: none. Optional `volume={35}, pages={19523--19536}`.

### liu2020keypose — verified
- Source: Crossref DOI 10.1109/CVPR42600.2020.01162 and DBLP `conf/cvpr/LiuJAK20` — both pp. 11599–11607, CVPR 2020. Authors match.
- Correction: none.

### bouthillier2021variance — verified
- Source: DBLP `conf/mlsys/BouthillierDBTN21`; proceedings.mlsys.org 2021 listing (16 authors, incl. "Nazanin Mohammadi Sepahvand", no Serdyuk); Proceedings of Machine Learning and Systems 3, pp. 747–769; arXiv 2103.03098.
- Bib: 16 authors in proceedings order, volume 3, pages 747–769, 2021 — all match. (arXiv v1 lists a 17th author, Dmitriy Serdyuk; the bib correctly follows the MLSys version.)
- Correction: none.

### northcutt2021errors — verified
- Source: DBLP `conf/nips/NorthcuttAM21` ("NeurIPS Datasets and Benchmarks", 2021); arXiv 2103.14749 (v4). 3 authors match.
- Correction: none.

### northcutt2021confident — verified
- Source: Crossref DOI 10.1613/jair.1.12125 — JAIR vol. 70, pp. 1373–1411, 2021. Correction: none.

### song2023noisysurvey — verified
- Source: Crossref DOI 10.1109/TNNLS.2022.3152527 — IEEE TNNLS 34(11):8135–8153, 2023. 5 authors match. Correction: none.

### liu2022nlte — verified (pagination-style note)
- Source: Crossref DOI 10.1109/CVPR52688.2022.01381 and OpenAlex: IEEE pagination 14187–14196 (= bib). CVF Open Access bibtex: pages 14207–14216.
- Both paginations are official. The bib uses IEEE pages here but CVF pages for qi2021offboard / ghiasi2021copypaste / gupta2019lvis. Not an error; optionally harmonize.

### marion2018labelfusion — verified
- Source: Crossref DOI 10.1109/ICRA.2018.8460950 — "Label Fusion: A Pipeline for Generating Ground Truth Labels for Real RGBD Data of Cluttered Scenes", ICRA 2018, pp. 3235–3242. 4 authors match. Correction: none.

### chen2016pseudotracklet — verified (patent)
- Source: Google Patents US9342759B1 (meta: DC.title "Object recognition consistency improvement using a pseudo-tracklet approach"; DC.contributor Yang Chen, Changsoo S. Jeong, Deepak Khosla, Kyungnam Kim, Shinko Y. Cheng, Lei Zhang, Alexander L. Honda, HRL Laboratories LLC; dates 2014-03-11 filed, 2016-05-17 granted; citation_patent_number US:9342759).
- Bib: title, all 7 inventors in order, "U.S. Patent 9,342,759 B1, HRL Laboratories", May 2016 — all correct.
- Correction: none.

### khan2006homography — verified
- Source: Crossref DOI 10.1007/11744085_11 (LNCS, Computer Vision – ECCV 2006, pp. 133–146, Springer, ISBN 978-3-540-33838-3); Springer book 10.1007/11744085 = "Computer Vision – ECCV 2006 … Proceedings, Part IV", LNCS vol. 3954. Authors Saad M. Khan, Mubarak Shah.
- Bib: series LNCS, volume 3954, pages 133–146, 2006 — all correct. Correction: none.

### dawid1979em — verified
- Source: Crossref DOI 10.2307/2346806 (Applied Statistics 28(1), 1979, first page 20); OUP/JSTOR/Wiley listings (via search): J. R. Stat. Soc. C (Applied Statistics) 28(1):20–28, March 1979.
- Bib: exact. Correction: none.

### kang2016tcnn — verified (venue-name consistency)
- Source: Crossref DOI 10.1109/CVPR.2016.95 — "Object Detection from Video Tubelets with Convolutional Neural Networks", 2016 IEEE CVPR, pp. 817–825; OpenAlex same. 4 authors match.
- Minor: booktitle says "Proc. IEEE/CVF Conf. …" while the bib's other 2016 CVPR entries (motiian2016privileged, shrivastava2016ohem) use "Proc. IEEE Conf. …" (IEEE Xplore title for 2016 is "2016 IEEE Conference on CVPR"). Harmonize.

### han2016seqnms — verified
- Source: arXiv export API id 1602.08465 (v3), "Seq-NMS for Video Object Detection", submitted 2016-02-26; 9 authors (Han, Khorrami, Le Paine, Ramachandran, Babaeizadeh, Shi, Li, Yan, Huang) match order.
- Correction: none.

### tripathi2016propagation — verified
- Source: Crossref DOI 10.1109/WACV.2016.7477702 — "Detecting temporally consistent objects in videos through object class label propagation", 2016 IEEE WACV, pp. 1–9; OpenAlex same. 4 authors match. Title in bib is exact.
- Correction: none required. Optional `pages={1--9}`.

### qi2021offboard — verified
- Source: CVF Open Access bibtex block: booktitle CVPR 2021, pages 6134–6144 (= bib). Crossref DOI 10.1109/CVPR46437.2021.00607 / OpenAlex give IEEE pagination 6130–6140. 7 authors match.
- Correction: none (CVF pagination is legitimate).

---

## 2. Entries in references.bib but NOT cited in main_v15.tex (12)

| key | status | source | note |
|---|---|---|---|
| lyu2022rtmdet | verified | arXiv 2212.07784 (2022-12-14), DBLP CoRR | 8 authors match |
| cao2021fakemix | verified | arXiv 2103.13279 (2021-03-24), DBLP CoRR | 7 authors match |
| liang2024randomframe | verified | Crossref 10.1007/978-3-031-80136-5_6 (ICPR 2024, LNCS, pp. 80–93, issued 2024-12-01); DBLP `conf/icpr/LiangI24` | ok |
| ghiasi2021copypaste | verified | DBLP `conf/cvpr/GhiasiCSQLCLZ21` pp. 2918–2928 (CVF); IEEE 10.1109/CVPR46437.2021.00294 = 2917–2927 | bib uses CVF pages, ok |
| wang2026seeclear | verified | arXiv 2603.19547 (2026-03-20), DBLP CoRR | 7 authors match |
| hoffer2020augment | verified | Crossref 10.1109/CVPR42600.2020.00815, DBLP — pp. 8126–8135 | ok |
| fort2021multiplicity | verified | arXiv 2105.13343 (2021-05-27), DBLP CoRR | 5 authors match |
| vapnik2009lupi | verified | Crossref 10.1016/j.neunet.2009.06.042 — Neural Netw. 22(5–6):544–557, 2009 | ok |
| motiian2016privileged | verified | Crossref 10.1109/CVPR.2016.166 — CVPR 2016, pp. 1496–1505 | ok |
| berman2019multigrain | verified | arXiv 1902.05509 (2019-02-14), DBLP CoRR | ok |
| shrivastava2016ohem | verified | Crossref 10.1109/CVPR.2016.89, DBLP — CVPR 2016, pp. 761–769 | ok |
| katharopoulos2018importance | verified | PMLR v80 page metadata: citation_firstpage 2525 / lastpage 2534 ("PMLR 80:2525-2534") | bib matches PMLR; DBLP shows 2530–2539 (DBLP is the outlier) |

Since these are uncited, IEEEbib will not print them; they can stay or be pruned at will.

---

## 3. Compact table — all 37 entries

| key | cited | status | verified against | fix needed |
|---|---|---|---|---|
| gajdosech2025glassnicol | yes | verified | Crossref 10.1109/IROS60139.2025.11246715, DBLP | none (opt. doi) |
| sapkota2025yolo | yes | minor mismatch | Crossref 10.1007/s10462-025-11253-3, DBLP | title: add "(YOLO)" |
| lin2014coco | yes | verified | Crossref 10.1007/978-3-319-10602-1_48 | none |
| sajjan2020cleargrasp | yes | verified | Crossref 10.1109/ICRA40945.2020.9197518 | none |
| rublee2011orb | yes | verified | Crossref 10.1109/ICCV.2011.6126544 | none |
| fischler1981ransac | yes | verified | Crossref 10.1145/358669.358692 | none |
| loshchilov2019adamw | yes | verified | DBLP conf/iclr/LoshchilovH19 | none |
| gupta2019lvis | yes | verified | DBLP (CVF pages), Crossref | none |
| jocher2024yolo11 | yes | verified | Ultralytics docs citation block | none (opt. version) |
| paul2021datadiet | yes | verified | DBLP conf/nips/PaulGD21 | none (opt. pages) |
| sorscher2022pruning | yes | verified | Crossref 10.52202/068431-1419, DBLP | none (opt. pages) |
| liu2020keypose | yes | verified | Crossref 10.1109/CVPR42600.2020.01162, DBLP | none |
| bouthillier2021variance | yes | verified | DBLP, proceedings.mlsys.org, arXiv 2103.03098 | none |
| northcutt2021errors | yes | verified | DBLP conf/nips/NorthcuttAM21, arXiv 2103.14749 | none |
| northcutt2021confident | yes | verified | Crossref 10.1613/jair.1.12125 | none |
| song2023noisysurvey | yes | verified | Crossref 10.1109/TNNLS.2022.3152527 | none |
| liu2022nlte | yes | verified | Crossref 10.1109/CVPR52688.2022.01381, CVF | none (opt. CVF pages 14207–14216) |
| marion2018labelfusion | yes | verified | Crossref 10.1109/ICRA.2018.8460950 | none |
| chen2016pseudotracklet | yes | verified | Google Patents US9342759B1 | none |
| khan2006homography | yes | verified | Crossref 10.1007/11744085_11, Springer LNCS 3954 | none |
| dawid1979em | yes | verified | Crossref 10.2307/2346806, OUP/JSTOR | none |
| kang2016tcnn | yes | verified (style) | Crossref 10.1109/CVPR.2016.95 | booktitle "IEEE/CVF" -> "IEEE" (consistency) |
| han2016seqnms | yes | verified | arXiv 1602.08465 | none |
| tripathi2016propagation | yes | verified | Crossref 10.1109/WACV.2016.7477702 | none (opt. pages 1–9) |
| qi2021offboard | yes | verified | CVF bibtex (6134–6144), Crossref | none |
| lyu2022rtmdet | no | verified | arXiv 2212.07784 | none |
| cao2021fakemix | no | verified | arXiv 2103.13279 | none |
| liang2024randomframe | no | verified | Crossref 10.1007/978-3-031-80136-5_6 | none |
| ghiasi2021copypaste | no | verified | DBLP (CVF pages), Crossref | none |
| wang2026seeclear | no | verified | arXiv 2603.19547 | none |
| hoffer2020augment | no | verified | Crossref 10.1109/CVPR42600.2020.00815 | none |
| fort2021multiplicity | no | verified | arXiv 2105.13343 | none |
| vapnik2009lupi | no | verified | Crossref 10.1016/j.neunet.2009.06.042 | none |
| motiian2016privileged | no | verified | Crossref 10.1109/CVPR.2016.166 | none |
| berman2019multigrain | no | verified | arXiv 1902.05509 | none |
| shrivastava2016ohem | no | verified | Crossref 10.1109/CVPR.2016.89 | none |
| katharopoulos2018importance | no | verified | PMLR v80 metadata | none |

Counts: 37 entries — 36 verified exact, 1 minor mismatch (sapkota2025yolo title), 0 not found.

---

## 4. Ready-to-apply bib fixes

Required (only one, and it is cosmetic):

```bibtex
@article{sapkota2025yolo,
  author={Ranjan Sapkota and Marco {Flores-Calero} and Rizwan Qureshi and Chetan Badgujar and Upesh Nepal and Alwin Poulose and Peter Zeno and Uday Bhanu Prakash Vaddevolu and Sheheryar Khan and Maged Shoman and Hong Yan and Manoj Karkee},
  title={{YOLO} Advances to Its Genesis: A Decadal and Comprehensive Review of the You Only Look Once ({YOLO}) Series},
  journal={Artif. Intell. Rev.},
  volume={58},
  number={9},
  note = {art. no. 274},
  year={2025}
}
```

Consistency (recommended, zero risk):

```bibtex
@inproceedings{kang2016tcnn,
  author    = {Kai Kang and Wanli Ouyang and Hongsheng Li and Xiaogang Wang},
  title     = {Object detection from video tubelets with convolutional neural networks},
  booktitle = {Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)},
  pages     = {817--825},
  year      = {2016}
}
```

Optional completeness (verified values; add only if space allows — IEEEbib prints pages/volume):

```bibtex
% tripathi2016propagation: add
  pages     = {1--9},

% paul2021datadiet: add
  volume={34},
  pages={20596--20607},

% sorscher2022pruning: add
  volume={35},
  pages={19523--19536},

% jocher2024yolo11: replace note with
  note={Software, version 11.0.0, \url{https://github.com/ultralytics/ultralytics}}

% gajdosech2025glassnicol: add
  doi={10.1109/IROS60139.2025.11246715}

% liu2022nlte: only if you want CVF pagination everywhere (qi/ghiasi/gupta already use CVF)
  pages={14207--14216},
```

No other changes are needed; do not touch dawid1979em, khan2006homography, chen2016pseudotracklet,
han2016seqnms, qi2021offboard, bouthillier2021variance, katharopoulos2018importance — all confirmed
exactly as written.

---

## 5. 한국어 요약

- references.bib 37건 전부 실존 논문/특허/소프트웨어로 확인됨. 없는 논문, 저자 오기, 연도·학회 오류 0건.
- main_v15.tex에 인용된 25건 모두 제목·저자·연도·학회·쪽수가 공식 기록(Crossref DOI, DBLP, arXiv, PMLR, Google Patents, Ultralytics 문서)과 일치.
- 중점 점검 항목 결과: dawid1979em(JRSS C 28(1):20–28) 정확 / kang2016tcnn(CVPR 2016 pp.817–825) 정확 / han2016seqnms(arXiv 1602.08465, 저자 9명 순서 일치) 정확 / tripathi2016propagation(WACV 2016 제목 정확, pp.1–9는 생략됨) / qi2021offboard(CVF 쪽수 6134–6144 정확, IEEE는 6130–6140) / khan2006homography(ECCV 2006 LNCS 3954 pp.133–146) 정확 / bouthillier2021variance(MLSys 2021 vol.3 pp.747–769, 저자 16명 proceedings판과 일치) 정확 / sapkota2025yolo(AIR 58(9) art.274, 2025, 저자 12명 일치) — 제목 끝 "(YOLO)"만 누락 / chen2016pseudotracklet(US 9,342,759 B1, HRL, 2016-05-17 등록, 발명자 7명 일치) 정확 / gajdosech2025glassnicol(IROS 2025 pp.20516–20523, DOI 10.1109/IROS60139.2025.11246715) 정확 / jocher2024yolo11(Ultralytics 공식 인용 블록과 동일) 정확.
- 수정 필요: sapkota2025yolo 제목에 "({YOLO})" 추가 1건. 권장: kang2016tcnn booktitle을 다른 2016 CVPR 항목과 같이 "Proc. IEEE Conf."로 통일. 나머지는 선택(쪽수·doi 보강).
- 인용 안 된 12건(lyu, cao, liang, ghiasi, wang, hoffer, fort, vapnik, motiian, berman, shrivastava, katharopoulos)도 전부 정확하며, IEEEbib는 미인용 항목을 출력하지 않으므로 그대로 둬도 무방.
