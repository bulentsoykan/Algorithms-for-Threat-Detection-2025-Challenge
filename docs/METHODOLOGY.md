# Technical Methodology

This document provides detailed technical information about the algorithms and approaches used in our 2nd place solution for the ATD 2025 Challenge.

## Table of Contents
1. [Problem Formulation](#problem-formulation)
2. [Algorithmic Approaches](#algorithmic-approaches)
3. [Implementation Details](#implementation-details)
4. [Performance Optimization](#performance-optimization)
5. [Results Analysis](#results-analysis)

## Problem Formulation

### Ship Track Construction Problem

Given a set of observations `O = {o_1, o_2, ..., o_n}` where each observation contains:
- **Position:** `(lat, lon)` - Geographic coordinates
- **Velocity:** `(v_x, v_y)` - Velocity components
- **Timestamp:** `t` - Observation time
- **No Identity:** Ship ID is not provided

**Objective:** Partition observations into tracks `T = {T_1, T_2, ..., T_k}` where each track represents a unique ship's trajectory.

### Challenges

1. **Data Association:** Determining which observations belong to the same ship
2. **Missing Data:** Ships may temporarily disappear from observation
3. **Maneuvers:** Ships can change course unpredictably
4. **Observation Density:** Varying spatial and temporal sampling rates
5. **Scale:** Real-world maritime data involves thousands of observations

## Algorithmic Approaches

### 1. Kalman Filter Framework

We implemented three Kalman filter variants using custom PyTorch implementations for GPU acceleration:

#### 1.1 Constant Velocity Model
**State Vector:** `x = [lat, lon, v_lat, v_lon]^T`

**State Transition:**
```
x_{k+1} = F_k x_k + w_k
```
where `F_k` is the state transition matrix and `w_k ~ N(0, Q)` is process noise.

**Observation Model:**
```
z_k = H x_k + v_k
```
where `H` is the observation matrix and `v_k ~ N(0, R)` is measurement noise.

**Implementation:** `kalman_const_velocity.ipynb`

#### 1.2 Constant Turn Rate Model
**Extended State Vector:** `x = [lat, lon, v_lat, v_lon, ω]^T`

Includes angular velocity `ω` to model ships making constant rate turns.

**State Transition:**
- Non-linear motion model for curved trajectories
- Extended Kalman Filter (EKF) for state estimation

**Implementation:** `kalman_const_turn.ipynb`

#### 1.3 Adaptive Half-Maneuver Model
**Adaptive Switching:**
- Combines constant velocity and maneuvering models
- Switches between models based on innovation residuals
- Improved handling of irregular ship movements

**Implementation:** `kalman_half_maneuver.ipynb`

### 2. Coverage-Based Track Refinement (CBTR)

CBTR is a distance-based clustering approach with ellipsoidal gating for improved association accuracy.

#### 2.1 Distance Metric
For each pair of observations `(o_i, o_j)`, compute distance:
```
d(o_i, o_j) = √[(Δlat)² + (Δlon)² + α(Δv_x)² + α(Δv_y)² + β(Δt)²]
```
where `α` and `β` are weighting parameters.

#### 2.2 Ellipsoidal Gating
Define an ellipsoidal gate around predicted positions:
```
(z - ẑ)^T S^{-1} (z - ẑ) ≤ γ
```
where:
- `z` is the observation
- `ẑ` is the predicted position
- `S` is the innovation covariance
- `γ` is the gate threshold

#### 2.3 Clustering
- Construct distance matrix `D` for all observation pairs
- Apply hierarchical clustering with linkage criteria
- Ellipsoidal gating filters out unlikely associations

**Implementation:** `cbtr_closest/cbtr_dist_matrix_maker.ipynb`

### 3. Deep Learning Hybrid Model

#### 3.1 Classification Framework
Formulate track construction as a 17-class classification problem using historical AIS data.

**Classes:**
1. Same ship, same track (continuation)
2. Same ship, different track (maneuver)
3-17. Different ships with varying characteristics

#### 3.2 Feature Engineering
For each observation pair `(o_i, o_j)`:
- **Spatial features:** Distance, bearing
- **Temporal features:** Time difference, velocity changes
- **Derived features:** Acceleration, heading change, closest point of approach

#### 3.3 Model Architecture
- **Input:** Feature vector for observation pair
- **Network:** Fully connected layers with dropout
- **Output:** 17-class probability distribution
- **Training:** Cross-entropy loss with class balancing

**Dataset Creation:** `create_20class_dataset.py`
**Training:** `deep_class20_folder.py`
**Evaluation:** `deep_model_eval.ipynb`

### 4. Ensemble Clustering

Multiple clustering algorithms applied to the CBTR distance matrix:

#### 4.1 DBSCAN
- Density-based clustering
- Handles noise and outliers
- No need to specify number of clusters

#### 4.2 Hierarchical Clustering
**Linkage Methods:**
- **Single linkage:** Minimum distance between clusters
- **Complete linkage:** Maximum distance between clusters
- **Average linkage:** Mean distance between clusters

#### 4.3 Comparative Analysis
Evaluate performance of different clustering methods to identify optimal approach.

**Implementation:** `dist_matrix_clustering.ipynb`

### 5. Oracle Analysis

Theoretical upper-bound analysis to understand algorithm performance limits.

#### 5.1 Screen Size Optimization
For varying screen sizes `k` (number of observations to consider):
- Compute oracle assignment using ground truth
- Analyze accuracy vs. computational cost trade-off

#### 5.2 Performance Ceiling
Determine theoretical maximum accuracy given:
- Observation density
- Data quality
- Algorithm constraints

**Implementation:** `oracle.ipynb`, `oracle_analysis.ipynb`

## Implementation Details

### GPU Acceleration

**Kalman Filters:**
- Custom PyTorch implementation
- Batch processing for parallel computation
- Significantly faster than CPU-based implementations

**Deep Learning:**
- PyTorch training pipeline
- GPU-accelerated forward/backward passes
- Mixed precision training for memory efficiency

### Memory Optimization

**Distance Matrix Computation:**
- Chunked processing for large datasets
- Pre-computation and caching of nearest neighbors
- Sparse matrix representations where applicable

**Dataset Generation:**
- Multi-process parallel generation
- NFS-based distributed processing
- Efficient parquet format for storage

### Data Pipeline

1. **Data Loading:** Read raw observations from CSV/Parquet
2. **Preprocessing:** Normalize features, handle missing data
3. **Feature Engineering:** Compute derived features for each observation pair
4. **Model Inference:** Run Kalman filters, CBTR, or deep learning model
5. **Post-processing:** Refine tracks, merge fragments, handle edge cases

## Performance Optimization

### Hyperparameter Tuning

**Optuna Framework:**
- Bayesian optimization for hyperparameter search
- Parallelized trials across multiple GPUs
- Early stopping for inefficient parameter configurations

**Tuned Parameters:**
- Kalman filter process/measurement noise covariances
- CBTR distance weights and gate thresholds
- Deep learning architecture and training hyperparameters
- Clustering algorithm parameters

### Computational Efficiency

**Runtime Performance:**
- Kalman filters: ~O(n) for n observations
- CBTR distance matrix: ~O(n²) with optimizations
- Deep learning inference: ~O(n²·m) for m features

**Memory Requirements:**
- Distance matrix: ~O(n²) storage
- Kalman filters: ~O(n) storage
- Deep learning: ~O(batch_size · model_params)

## Results Analysis

### Performance Metrics

**Primary Metric:** Track construction accuracy
- Percentage of correctly associated observations
- Evaluated against ground truth ship identities

**Secondary Metrics:**
- Precision: Fraction of predicted associations that are correct
- Recall: Fraction of true associations that are predicted
- F1 Score: Harmonic mean of precision and recall

### Error Analysis

**Common Failure Modes:**
1. **Close Encounters:** Ships passing nearby cause association errors
2. **Long Gaps:** Extended periods without observations
3. **Abrupt Maneuvers:** Sudden course changes violate motion models
4. **Observation Noise:** GPS errors and data quality issues

### Improvement Trajectory

| Dataset | Baseline | Our Team | Improvement |
|---------|----------|----------|-------------|
| Practice Set 1 | ~50% | 40% | -10% |
| Practice Set 2 | ~48% | 51% | +3% |
| Practice Set 3 | ~47% | 50% | +3% |
| **Evaluation Set** | **46.4%** | **55.1%** | **+18.8%** |

**Key Insights:**
- Initial approach underperformed due to overfitting on training data
- Iterative refinement based on practice set feedback
- Hybrid approach combining multiple algorithms yielded best results

### Algorithmic Contributions

1. **GPU-Accelerated Kalman Filters:** 10-50x speedup over CPU implementations
2. **CBTR with Ellipsoidal Gating:** Improved association accuracy by ~5%
3. **Deep Learning Hybrid:** Leveraged historical AIS data for ~8% improvement
4. **Ensemble Clustering:** Combined strengths of multiple algorithms for robustness

## Future Directions

### Potential Improvements

1. **Multi-Hypothesis Tracking:** Maintain multiple track hypotheses
2. **Graph Neural Networks:** Learn relational structures between observations
3. **Reinforcement Learning:** Learn optimal association policies
4. **Probabilistic Data Association:** JPDA/MHT algorithms for ambiguous cases
5. **Temporal Context:** Incorporate longer-term ship behavior patterns

### Scalability Enhancements

1. **Distributed Computing:** Parallelize across multiple nodes
2. **Approximate Nearest Neighbors:** Reduce distance matrix computation
3. **Incremental Processing:** Update tracks as new observations arrive
4. **Online Learning:** Adapt models based on incoming data

## References

1. Bar-Shalom, Y., et al. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
2. Blackman, S., & Popoli, R. (1999). *Design and Analysis of Modern Tracking Systems*. Artech House.
3. Goodfellow, I., et al. (2016). *Deep Learning*. MIT Press.
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

## Contact

For questions about the methodology or implementation details, please refer to the main [README](../README.md).
