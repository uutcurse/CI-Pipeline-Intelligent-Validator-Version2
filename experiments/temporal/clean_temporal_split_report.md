# Clean Temporal Split Report

### Objective
The previous temporal result (Macro F1 = 0.6459) was invalid due to massive data leakage. The original E06 random split had already seen 68% of the temporal test repositories. This phase attempts to construct a perfectly clean, uncontaminated temporal test set using only previously unseen repositories.

### Contamination exclusion
To guarantee no leakage, we exclude the union of all repositories present in the original E06 Train, Validation, and Test partitions.
* Original E06 Train Repositories: 2746
* Original E06 Val Repositories: 589
* Original E06 Test Repositories: 589
* **Total Excluded Repositories:** 3924

### Cutoff methodology
No chronological cutoffs could be selected because there are exactly 0 candidate repositories remaining after excluding the seen repositories.

### Final split
* Train: 0 samples, 0 repositories
* Validation: 0 samples, 0 repositories
* Test: 0 samples, 0 repositories

### Leakage verification
* Clean Train n Seen: 0
* Clean Validation n Seen: 0
* Clean Test n Seen: 0

### Class distribution
* Train: LOW (0), MEDIUM (0), HIGH (0)
* Validation: LOW (0), MEDIUM (0), HIGH (0)
* Test: LOW (0), MEDIUM (0), HIGH (0)

### Feasibility
Insufficient clean temporal data for a statistically meaningful temporal experiment. Because the entire dataset was exhaustively partitioned during the original E06 training/evaluation phase, there are no unexposed repositories left to formulate a genuinely unseen-repository temporal test set.
