# E-COMBINE — our v4 ⊕ collaborator DINoV2 (w on his_holdout ∩ our TEST, locked).
slice n=2265; ours=0.7214 his=0.7063 **blend(w=0.58)=0.7547** Q=0.806; paired Δ=+0.0334 CI[+0.0215,+0.0457] **KEEP v5**.
hidden-test argmax-agreement=0.765; v5 dist={1: np.float64(0.195), 2: np.float64(0.16), 3: np.float64(0.014), 4: np.float64(0.097), 5: np.float64(0.07), 6: np.float64(0.104), 7: np.float64(0.207), 8: np.float64(0.152)}

| model | slice macro | per-class F1 |
|---|---|---|
| ours(v4) | 0.7214 | St=0.97 Wa=0.84 Ru=0.00 Bi=0.88 Ca=0.93 Bu=0.54 Tr=0.82 Su=0.79 |
| his | 0.7063 | St=0.92 Wa=0.85 Ru=0.00 Bi=0.89 Ca=0.88 Bu=0.40 Tr=0.77 Su=0.93 |
| blend | 0.7547 | St=0.96 Wa=0.85 Ru=0.00 Bi=0.94 Ca=0.95 Bu=0.58 Tr=0.86 Su=0.90 |
