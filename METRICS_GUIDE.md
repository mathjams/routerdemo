# Metrics Guide - Understanding Your Demo Statistics

This guide explains every metric, visualization, and statistic shown on the adaptive router demo website.

## Table of Contents

1. [Per-Image Results (Results Table)](#per-image-results)
2. [Router Graph Visualization](#router-graph-visualization)
3. [Aggregate Metrics Cards](#aggregate-metrics-cards)
4. [Routing Distribution](#routing-distribution)
5. [Savings Calculations](#savings-calculations)

---

## Per-Image Results (Results Table)

These metrics appear in the results table for **each uploaded image**.

### Predicted Class
**Example:** `42`

- **What it is:** The numeric class ID predicted by the model (0-99 for CIFAR-100)
- **How it's calculated:** `argmax(branch_logits)` - the class with highest probability
- **Why it matters:** The model's final answer for what's in the image

### Predicted Label
**Example:** `"cat"`

- **What it is:** The human-readable name for the predicted class
- **How it's calculated:** Maps predicted class ID to CIFAR-100 class names
- **Why it matters:** Easier to understand than numeric IDs

### Confidence
**Example:** `0.95` or `95%`

- **What it is:** How confident the model is in its prediction
- **How it's calculated:** `max(softmax(branch_logits))` - probability of the top prediction
- **Range:** 0.0 to 1.0 (0% to 100%)
- **Interpretation:**
  - **0.9-1.0 (90-100%):** Very confident, likely correct
  - **0.7-0.9 (70-90%):** Reasonably confident
  - **0.5-0.7 (50-70%):** Uncertain, might be wrong
  - **< 0.5 (< 50%):** Very uncertain (shouldn't happen for top prediction)
- **Why it matters:** Helps you trust or question the prediction

### Route Chosen
**Example:** `Branch 2`

- **What it is:** Which branch model handled this specific image
- **How it's calculated:** `argmax(router_logits)` - router picks the branch
- **Why it matters:** Shows which complexity level was used for this image

### Route Confidence
**Example:** `0.87` or `87%`

- **What it is:** How confident the **router** was in selecting this branch
- **How it's calculated:** `max(softmax(router_logits))` - probability of chosen branch
- **Range:** 0.0 to 1.0 (0% to 100%)
- **Interpretation:**
  - **High (>80%):** Router is sure this is the right branch
  - **Medium (60-80%):** Router has some uncertainty
  - **Low (<60%):** Router is guessing between branches
- **Why it matters:** High route confidence + high prediction confidence = very reliable result

### Ground Truth (Optional)
**Example:** `42`

- **What it is:** The actual correct class for this image (if you provided labels)
- **How it's provided:** You pass a JSON array of labels when uploading
- **Why it matters:** Lets you measure accuracy

### Is Correct (Optional)
**Example:** ✓ (checkmark) or ✗ (x)

- **What it is:** Whether the prediction matched the ground truth
- **How it's calculated:** `predicted_class == ground_truth`
- **Why it matters:** Visual indicator of correctness

### Image Preview
**Visual:** 32×32 pixel thumbnail

- **What it is:** Downscaled version of your uploaded image
- **How it's created:** Resized to 32×32 (what the model actually sees)
- **Why it matters:** Shows you exactly what input the model received

---

## Router Graph Visualization

The SVG diagram showing your model architecture.

### Router Node (Center, Purple)
**Visual:** Large purple circle in center

- **What it shows:** The router model that decides branch selection
- **Size:** Fixed (not proportional, since router is small)
- **Text shown:**
  - "Router"
  - Parameter count (e.g., "0.5M params")
- **Why it matters:** Central hub that directs all traffic

### Branch Nodes (Around Router, Green/Blue)
**Visual:** Circles arranged around router

- **What they show:** Your branch models (classifiers)
- **Size:** Proportional to number of parameters
  - Larger circle = larger model (more params)
  - Smaller circle = smaller model (fewer params)
- **Color:**
  - **Green:** Idle/default state
  - **Blue:** Currently selected (when you click a result row)
- **Text shown:**
  - "Branch X"
  - Parameter count (e.g., "2.3M")
  - FLOPs (if available, e.g., "50M FLOPs")
- **Why it matters:** Visual comparison of model sizes

### Arrows/Connections
**Visual:** Lines from router to branches

- **What they show:** Possible routing paths
- **Style:**
  - **Solid line:** Active route (when selected)
  - **Dashed line:** Inactive routes
- **Thickness:** Selected route is thicker
- **Numbers on arrows:** Count of images routed to that branch
- **Why it matters:** Shows routing decisions visually

### Legend
**Visual:** Small icons at bottom

- **Purple circle:** Router
- **Green circle:** Idle branch
- **Blue circle:** Selected branch
- **Size indicator:** "Node size ~ parameters"

---

## Aggregate Metrics Cards

Summary statistics for the **entire batch** of uploaded images.

### Total Images
**Example:** `50`

- **What it is:** Number of images you uploaded
- **How it's calculated:** Count of processed images
- **Why it matters:** Context for other metrics

### Average Confidence
**Example:** `0.89` or `89%`

- **What it is:** Mean prediction confidence across all images
- **How it's calculated:** `sum(confidences) / total_images`
- **Range:** 0.0 to 1.0 (0% to 100%)
- **Interpretation:**
  - **>85%:** Model is generally very confident
  - **70-85%:** Reasonable confidence
  - **<70%:** Model is often uncertain
- **Why it matters:** Overall measure of model certainty

### Accuracy (Optional)
**Example:** `0.92` or `92%`

- **What it is:** Percentage of correct predictions
- **How it's calculated:** `correct_predictions / total_images`
- **Only shown if:** You provided ground truth labels
- **Range:** 0.0 to 1.0 (0% to 100%)
- **Why it matters:** Direct measure of model performance

---

## Routing Distribution

Shows **how often each branch was used**.

### Per-Branch Usage
**Example for Branch 0:** `5 images (10%)`

- **What it shows:** Number and percentage of images sent to this branch
- **Visualization:** Horizontal bar (length = percentage)
- **Colors:** Primary blue
- **Why it matters:** Shows routing behavior patterns

### Interpretation

**Balanced Distribution** (all branches ~25% for 4 branches):
- Router is using all branches
- Adaptive routing is working
- Good diversity in routing decisions

**Unbalanced Distribution** (one branch >80%):
- Router prefers one branch heavily
- May indicate:
  - Training issue (router not well-trained)
  - Dataset bias (all images are similar)
  - One branch is dominant in accuracy

**Empty Branches** (some branches at 0%):
- Router never uses certain branches
- Potential issues:
  - Branches may be redundant
  - Router learned to ignore them
  - May need retraining

---

## Savings Calculations

The **key metrics** showing computational efficiency gains.

### Parameters Saved

#### Average Routed
**Example:** `2.3M parameters`

- **What it is:** Average parameters used per image with routing
- **How it's calculated:** `sum(params_of_chosen_branches) / total_images`
- **Why it matters:** Actual computational cost with routing

#### Baseline (Largest)
**Example:** `5.4M parameters`

- **What it is:** Parameters if you always used the largest branch
- **How it's calculated:** Max parameters across all branches
- **Why it matters:** What you'd use without adaptive routing

#### Savings
**Example:** `87.5%`

- **What it is:** Percentage of parameters saved vs baseline
- **How it's calculated:**
  ```
  savings = (baseline - avg_routed) / baseline * 100
  ```
- **Formula in detail:**
  ```python
  total_routed = sum(params[chosen_branch[i]] for i in images)
  total_baseline = largest_branch_params * num_images
  savings_percent = (total_baseline - total_routed) / total_baseline * 100
  ```
- **Range:** -∞ to 100%
  - **>80%:** Excellent savings, mostly using small branches
  - **50-80%:** Good savings, balanced routing
  - **<50%:** Limited savings, often using large branches
  - **Negative:** Using more than baseline (shouldn't happen)
- **Why it matters:** **Core metric** - shows computational efficiency

#### Visual Bar
**Color:** Green

- Filled portion = savings percentage
- Shows savings at a glance

### FLOPs Saved (Optional)

**Same structure as Parameters Saved, but for FLOPs**

#### What are FLOPs?
- **FLOPs:** Floating Point Operations
- **Measures:** Actual computational work (multiply-adds)
- **More accurate than params:** Better represents real inference cost

#### Average Routed
**Example:** `45M FLOPs`

- Average FLOPs per image with routing
- Includes router FLOPs + chosen branch FLOPs

#### Baseline (Largest)
**Example:** `520M FLOPs`

- FLOPs for largest branch + router
- What you'd compute without routing

#### Savings
**Example:** `91.3%`

- Percentage of FLOPs saved
- **Usually higher than parameter savings** because:
  - Small branches have fewer layers (not just narrower)
  - FLOPs scale with both width and depth

#### Availability
- **Requires:** `fvcore` library installed
- **If unavailable:** Shows `null`, only parameters shown
- **To enable:** `pip install fvcore`

#### Visual Bar
**Color:** Blue

---

## Real-World Interpretation

### Example: Good Routing System

```
Total Images: 100
Average Confidence: 88%
Accuracy: 90%

Routing Distribution:
- Branch 0 (small):  30 images (30%)
- Branch 1 (medium): 40 images (40%)
- Branch 2 (large):  20 images (20%)
- Branch 3 (largest): 10 images (10%)

Parameters Saved: 65%
FLOPs Saved: 78%
```

**Interpretation:**
- ✅ High confidence and accuracy
- ✅ Most images use smaller branches
- ✅ Significant computational savings
- ✅ Routing is working as intended

### Example: Potential Issues

```
Total Images: 100
Average Confidence: 92%
Accuracy: 91%

Routing Distribution:
- Branch 0 (small):   2 images (2%)
- Branch 1 (medium):  3 images (3%)
- Branch 2 (large):   5 images (5%)
- Branch 3 (largest): 90 images (90%)

Parameters Saved: 5%
FLOPs Saved: 8%
```

**Interpretation:**
- ⚠️ High accuracy but poor routing
- ⚠️ Almost always uses largest branch
- ⚠️ Minimal savings (defeats purpose)
- 🔧 **Action:** Router may need retraining

---

## Key Metrics Summary Table

| Metric | What It Shows | Good Range | Why It Matters |
|--------|---------------|------------|----------------|
| **Confidence** | Prediction certainty | >85% | Trust in results |
| **Route Confidence** | Router certainty | >70% | Routing reliability |
| **Accuracy** | Correctness % | >85% | Model quality |
| **Params Saved** | Size efficiency | 50-85% | Memory/storage savings |
| **FLOPs Saved** | Compute efficiency | 60-90% | Speed/cost savings |
| **Routing Balance** | Branch usage spread | ~Even | Proper adaptive behavior |

---

## Understanding the Trade-offs

### The Ideal Scenario
- **High accuracy** (model works well)
- **High savings** (efficient routing)
- **Balanced distribution** (using all branches appropriately)

### Common Trade-offs

**Accuracy vs Savings:**
- **High savings, lower accuracy:** Too aggressive (using small branches too often)
- **High accuracy, low savings:** Too conservative (using large branches too often)
- **Goal:** Find sweet spot with acceptable accuracy and good savings

**Confidence vs Correctness:**
- **High confidence, low accuracy:** Overconfident wrong predictions
- **Low confidence, high accuracy:** Model unsure but still correct
- **Best:** High confidence AND high accuracy

---

## Practical Usage Examples

### Debugging Poor Performance

**Symptom:** Low accuracy (60%)

**Check:**
1. **Average confidence:** If also low → model genuinely uncertain
2. **Routing distribution:** If unbalanced → routing may be broken
3. **Per-image results:** Look for patterns in errors

### Optimizing for Cost

**Goal:** Maximize savings while maintaining accuracy

**Strategy:**
1. Check current savings percentage
2. If <50% savings, retrain router to be more aggressive
3. Monitor accuracy - stop if it drops below threshold

### Validating Training

**After training, upload test set:**
1. **Accuracy** should match training metrics
2. **Savings** should be >50% to justify adaptive approach
3. **Distribution** should show smaller branches used often

---

## Tips for Better Insights

1. **Upload diverse images:** Test routing behavior on different image types
2. **Provide ground truth:** Enable accuracy metrics
3. **Compare batches:** Test easy vs hard images separately
4. **Watch route confidence:** Low confidence may indicate router uncertainty
5. **Check individual results:** Click rows to see which branch was used

---

## Summary

The demo provides **three levels of insight**:

1. **Per-Image:** Detailed results for each prediction
2. **Aggregate:** Overall performance statistics
3. **Visual:** Router graph showing architecture and routing

**Most Important Metrics:**
- ✅ **Accuracy:** Is the model correct?
- ✅ **Savings %:** How much computation did we save?
- ✅ **Routing Distribution:** Is adaptive routing working?

Use these metrics to understand, debug, and optimize your adaptive routing system!
