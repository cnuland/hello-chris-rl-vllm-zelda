# Reward Shaping

- **Env reward**: keep sparse milestone rewards; add coverage (+0.2 first-visit tile/room), small RND curiosity (≤30% of extrinsic).
- **RLAIF**: train \(R_\phi(s,a)\) from judge preferences (Bradley-Terry). Use \(r' = r + \lambda R_\phi\), λ=0.1–0.2.
- **Potential-based wrap** when possible to preserve optimality.
- **SIL/AWR**: periodically imitate top-K judged segments.
