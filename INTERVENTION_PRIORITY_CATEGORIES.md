# Intervention Priority Classification Categories

## Overview

The hybrid classification system combines **variance decomposition** (which factors explain recovery patterns) with **absolute conditions** (capacity/damage thresholds) to screen counties for policy interventions.

This is a **first-order screening tool** - it identifies which counties may benefit from which types of interventions, not a definitive policy prescription.

---

## Four Categories

### 1. **Critical Capacity Bottleneck** 🔴
**Color:** Dark Red (`#d7191c`)

**Definition:**
- Construction capacity < 25th percentile (P25) **AND**
- Variance share of capacity < 40%

**Interpretation:**
- Counties with very low construction capacity
- Capacity is so uniformly low it doesn't explain much variance
- But the absolute level is critically constraining
- **Policy implication:** Urgent capacity building needed regardless of variance patterns

**Key insight:** These counties may appear "damage-driven" by variance decomposition, but this is because capacity varies little (it's universally low). The low absolute capacity still represents a critical bottleneck.

---

### 2. **Capacity Building Priority** 🟠
**Color:** Orange (`#fdae61`)

**Definition:**
- Variance share of capacity > 50% **AND**
- Construction capacity < 25th percentile (P25)

**Interpretation:**
- Capacity explains most of the variance in recovery patterns
- AND capacity is objectively low
- **Policy implication:** Both variance patterns and absolute conditions point to capacity interventions

---

### 3. **Damage Mitigation Priority** 🔵
**Color:** Light Blue (`#abd9e9`)

**Definition:**
- Variance share of damage > 50% **AND**
- Total expected annual damage > 75th percentile (P75)

**Interpretation:**
- Damage explains most of the variance in recovery patterns
- AND damage exposure is objectively high
- **Policy implication:** Both variance patterns and absolute conditions point to damage reduction interventions (e.g., building codes, retrofits, coastal protection)

---

### 4. **Mixed Strategy** 🔷
**Color:** Dark Blue (`#2c7bb6`)

**Definition:**
- All counties that don't meet criteria for categories 1-3

**Interpretation:**
- No single dominant pattern in variance decomposition
- OR absolute conditions don't clearly prioritize one intervention type
- **Policy implication:** Context-dependent strategies needed; both capacity building and damage mitigation may be valuable

---

## Thresholds Used

### Annual Perspective:
- **Capacity threshold (P25):** 25th percentile of `construction_capacity` (permits/month)
- **Damage threshold (P75):** 75th percentile of `total_ead` (expected annual damage in housing units)

### Per-Event Perspective:
- **Capacity threshold (P25):** 25th percentile of `construction_capacity` (permits/month)
- **Damage threshold (P75):** 75th percentile of `median_damage_units` (median event damage)

---

## Conceptual Framework

```
┌─────────────────────────────────────────────────────────┐
│  VARIANCE DECOMPOSITION                                 │
│  → Which factors EXPLAIN observed patterns?             │
│  → Scientific understanding of drivers                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│  HYBRID CLASSIFICATION                                  │
│  → Variance patterns + Absolute conditions              │
│  → First-order screening for interventions              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│  POLICY ANALYSIS                                        │
│  → Detailed cost-benefit analysis                       │
│  → Context-specific intervention design                 │
│  → Implementation feasibility assessment                │
└─────────────────────────────────────────────────────────┘
```

---

## Important Distinctions

### Variance Share ≠ Intervention Effectiveness

**Example:** A county with very low capacity (1.5 permits/month) may appear "damage-driven" by variance decomposition because:
- Capacity is uniformly low across similar counties → doesn't vary → can't explain variance
- Damage varies more across counties → explains more variance

**But:** This doesn't mean damage reduction is the best policy! The low capacity is still a critical constraint. This is why we add the absolute condition check (capacity < P25) to catch these cases.

### Why Both Metrics Matter

1. **Variance decomposition:** Tells us which factors predict where problems will be worse
2. **Absolute conditions:** Tells us where problems are objectively severe regardless of relative patterns
3. **Hybrid approach:** Combines both to avoid missing critical constraints that don't vary much

---

## Classification Logic (Pseudocode)

```python
if capacity < P25 AND share_capacity < 0.4:
    return 'Critical Capacity Bottleneck'
    
elif share_capacity > 0.5 AND capacity < P25:
    return 'Capacity Building Priority'
    
elif share_damage > 0.5 AND damage > P75:
    return 'Damage Mitigation Priority'
    
else:
    return 'Mixed Strategy'
```

---

## Sensitivity Analysis (Separate from Classification)

Note: The sensitivity analysis (measuring marginal effects of doubling capacity vs. halving damage) is calculated but **not used** in the classification logic above. 

The sensitivity metrics provide complementary information about intervention effectiveness but are kept separate from the hybrid classification to maintain clarity:
- **Variance decomposition:** What explains patterns?
- **Hybrid classification:** First-order screening based on patterns + thresholds
- **Sensitivity analysis:** How effective would standardized interventions be?

This separation avoids conflating "explains variance" with "best intervention target."

---

## Usage Notes

- This classification is a **screening tool**, not a definitive policy recommendation
- Counties classified as "Critical Capacity Bottleneck" or "Capacity Building Priority" warrant deeper analysis of capacity constraints
- Counties classified as "Damage Mitigation Priority" warrant analysis of cost-effective damage reduction strategies
- "Mixed Strategy" counties may benefit from integrated approaches or require event-specific analysis
- Always consider local context, implementation feasibility, and cost-effectiveness in actual policy decisions
