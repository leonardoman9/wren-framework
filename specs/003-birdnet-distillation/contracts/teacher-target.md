# Contract: teacher target and loss

Inputs per drawn training window: student logits z (K classes), folder label y, BirdNET confidences c (S = K − 1
project species, class order; S = K when there is no background class).

1. p = [c_1 … c_S, Π_i (1 − c_i)] (background at its class index, which prepare puts last); without background p = c.
2. p ← max(p, 1e-4); p ← p / Σp.
3. q = p^(1/T) / Σ p^(1/T).
4. m = 0 if y is a species (not the background) and max(c) < 0.1, else 1.
5. loss = (1 − α) · mean_batch[(1 − e^(−ce))^γ · ce] + α · T² · Σ_batch m · Σ_k q_k (log q_k − log softmax(z/T)_k) / max(1, Σ m),
   with ce = cross_entropy(z, y).

Unit checks (`tests/test_birdnet.py`):
- c = [0.9, 0, …, 0] with 8 species and background, T = 4: q_0 ≥ 3/9 (measured 3.92/9).
- c all 0 → q's largest entry is the background.
- α = 0 → loss equals the existing focal loss; α = 1 → only the KL term.
- q sums to 1 and has no zero entry for any T > 0.
- a species window with c all below 0.1 → its KL term is not counted (loss of a batch of only such windows = (1 − α) ·
  focal); a background window with c all 0 keeps its KL term.
