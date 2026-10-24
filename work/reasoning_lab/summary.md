
### correct 小样本（gold4200 有金标；real_c2/c8 为 ASR 原文，评价术语表变体召回）

| sample | effort | ok | elapsed_s | cost_cny | prompt | completion | reasoning | diff_ratio | n_items | gold_recall | gold_wrong_remaining | glossary_recall | glossary_remaining | sim_to_orig | num_equal | book_equal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gold4200 | off | True | 20.1 | 0.012789 | 3116 | 2481 | 0 | 0.006905 | 16 | 0.0806 | 57 |  |  | 0.9951 | True | True |
| real_c2 | off | True | 14.57 | 0.007796 | 2144 | 1507 | 0 | 0.000402 | 1 |  |  | 0.0 | 16 | 0.9664 | True | True |
| gold4200 | low | True | 74.58 | 0.049894 | 3141 | 11751 | 9297 | 0.029762 | 89 | 0.9194 | 5 |  |  | 0.9732 | True | True |
| gold4200 | high | True | 85.7 | 0.057638 | 3141 | 13687 | 11228 | 0.030952 | 87 | 0.9194 | 5 |  |  | 0.9723 | True | True |
| real_c2 | low | True | 84.53 | 0.058391 | 2169 | 14181 | 12698 | 0.043775 | 73 |  |  | 0.875 | 2 | 0.9279 | True | True |
| real_c8 | off | True | 13.66 | 0.007501 | 2093 | 1446 | 0 | 0.015286 | 25 |  |  | 0.2 | 4 | 0.9563 | True | True |
| real_c2 | high | True | 100.87 | 0.073787 | 2169 | 18030 | 16542 | 0.04498 | 74 |  |  | 0.875 | 2 | 0.9262 | True | True |
| real_c8 | low | True | 50.25 | 0.03186 | 2118 | 7561 | 6134 | 0.026951 | 45 |  |  | 1.0 | 0 | 0.946 | True | True |
| gold4200 | max | False | 193.93 | 0.13331 | 3141 | 32605 | 30183 | 0.057143 | 0 | 0.0 | 62 |  |  | 1.0 | True | True |
| real_c8 | high | True | 112.01 | 0.073776 | 2118 | 18040 | 16612 | 0.028962 | 51 |  |  | 1.0 | 0 | 0.9437 | True | True |
| real_c2 | max | True | 216.13 | 0.150523 | 2169 | 37214 | 35730 | 0.115261 | 121 |  |  | 0.9375 | 1 | 0.8806 | True | True |
| real_c8 | max | True | 197.14 | 0.132116 | 2118 | 32625 | 31200 | 0.032582 | 56 |  |  | 1.0 | 0 | 0.9411 | True | True |

### compose 小样本（2 块）

| sample | effort | ok | fallback | attempts | elapsed_s | cost_cny | prompt | completion | reasoning | ratio | n_paras | warnings | glossary_hit | sim_to_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m1 | off | True | False | 1 | 10.12 | 0.006498 | 3514 | 746 | 0 | 0.709 | 6 | [] | 5 | 0.7305 |
| m1 | low | True | False | 1 | 37.13 | 0.023055 | 3539 | 4879 | 4189 | 0.628 | 6 | [] | 6 | 0.7483 |
| m1 | high | True | False | 1 | 42.93 | 0.025574 | 3539 | 6324 | 5721 | 0.567 | 4 | [] | 5 | 1.0 |
| m1 | max | True | False | 1 | 65.76 | 0.047003 | 3539 | 10866 | 10133 | 0.655 | 10 | [] | 5 | 0.7539 |
| m9 | off | True | False | 1 | 9.91 | 0.004952 | 3384 | 737 | 0 | 0.661 | 8 | [] | 10 | 0.8637 |
| m9 | low | True | False | 1 | 50.48 | 0.034352 | 3409 | 8112 | 7370 | 0.665 | 6 | [] | 10 | 0.8384 |
| m9 | high | True | False | 1 | 38.06 | 0.024816 | 3409 | 5728 | 4976 | 0.658 | 10 | [] | 10 | 1.0 |
| m9 | max | True | False | 1 | 57.0 | 0.03552 | 3409 | 8404 | 7676 | 0.647 | 7 | [] | 10 | 0.7343 |

### compose 课程档案（低 vs 高；生产默认 high）

| sample | effort | ok | elapsed_s | cost_cny | prompt | completion | reasoning | n_outline | n_glossary | n_people | n_books |
|---|---|---|---|---|---|---|---|---|---|---|---|
| card | low | True | 54.26 | 0.064088 | 25976 | 9528 | 7441 | 15 | 34 | 25 | 16 |
| card | high | True | 95.68 | 0.094075 | 25976 | 17056 | 14842 | 15 | 30 | 22 | 15 |