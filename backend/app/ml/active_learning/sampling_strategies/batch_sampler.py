import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from ..acquisition.factory import AcquisitionFunctionFactory


class BatchActiveLearning:
    """
    Batch Active Learning for parallel experimental design.

    Enables selection of multiple points simultaneously for parallel evaluation,
    which is crucial for engineering applications where experiments can be
    run in parallel on HPC clusters or multiple test rigs.
    """

    def __init__(self, acquisition_function: str = 'expected_improvement',
                 batch_strategy: str = 'diversity', **kwargs):
        """
        Initialize batch active learning.

        Args:
            acquisition_function: Base acquisition function to use
            batch_strategy: Strategy for batch selection
                          - 'diversity': Spatial diversity with acquisition
                          - 'clustering': K-means clustering approach
                          - 'sequential': Sequential optimization
                          - 'hallucination': Fantasy point method
                          - 'local_penalization': Local penalization around selected points
        """
        self.acquisition_function_name = acquisition_function
        self.batch_strategy = batch_strategy
        self.hyperparameters = kwargs

    def select_batch(self, X_candidates: np.ndarray, model, batch_size: int,
                    current_best: float = None, **kwargs) -> Dict[str, Any]:
        """
        Select a batch of points for parallel evaluation.

        Args:
            X_candidates: Candidate points to select from
            model: Trained surrogate model
            batch_size: Number of points to select
            current_best: Current best observed value

        Returns:
            Dictionary with selected points and metadata
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        if batch_size >= len(X_candidates):
            return {
                'selected_points': X_candidates,
                'selected_indices': list(range(len(X_candidates))),
                'acquisition_values': np.ones(len(X_candidates)),
                'strategy': self.batch_strategy,
                'diversity_scores': np.ones(len(X_candidates))
            }

        # Create acquisition function
        acq_func = AcquisitionFunctionFactory.create(
            self.acquisition_function_name, **self.hyperparameters
        )

        if current_best is not None:
            acq_func.set_current_best(current_best)

        # Select strategy
        if self.batch_strategy == 'diversity':
            return self._diversity_batch_selection(X_candidates, acq_func, model, batch_size, **kwargs)
        elif self.batch_strategy == 'clustering':
            return self._clustering_batch_selection(X_candidates, acq_func, model, batch_size, **kwargs)
        elif self.batch_strategy == 'sequential':
            return self._sequential_batch_selection(X_candidates, acq_func, model, batch_size, **kwargs)
        elif self.batch_strategy == 'hallucination':
            return self._hallucination_batch_selection(X_candidates, acq_func, model, batch_size, **kwargs)
        elif self.batch_strategy == 'local_penalization':
            return self._local_penalization_batch_selection(X_candidates, acq_func, model, batch_size, **kwargs)
        else:
            raise ValueError(f"Unknown batch strategy: {self.batch_strategy}")

    def _diversity_batch_selection(self, X_candidates: np.ndarray, acq_func, model,
                                 batch_size: int, **kwargs) -> Dict[str, Any]:
        """
        Select batch using diversity-aware approach.

        Balances acquisition function values with spatial diversity.
        """
        diversity_weight = kwargs.get('diversity_weight', 0.5)
        distance_metric = kwargs.get('distance_metric', 'euclidean')

        # Get acquisition values
        acq_values = acq_func.evaluate(X_candidates, model)

        # Calculate pairwise distances
        distances = squareform(pdist(X_candidates, metric=distance_metric))

        selected_indices = []
        remaining_indices = list(range(len(X_candidates)))

        for _ in range(batch_size):
            if not remaining_indices:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining_indices:
                # Acquisition score
                acq_score = acq_values[idx]

                # Diversity score (minimum distance to selected points)
                if selected_indices:
                    min_distance = min(distances[idx, sel_idx] for sel_idx in selected_indices)
                    diversity_score = min_distance
                else:
                    diversity_score = 1.0  # First point gets maximum diversity

                # Combined score
                combined_score = (1 - diversity_weight) * acq_score + diversity_weight * diversity_score

                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        # Calculate diversity scores for selected points
        diversity_scores = np.zeros(len(selected_indices))
        for i, idx in enumerate(selected_indices):
            if i == 0:
                diversity_scores[i] = 1.0
            else:
                # Minimum distance to previously selected points
                min_dist = min(distances[idx, selected_indices[j]] for j in range(i))
                diversity_scores[i] = min_dist

        return {
            'selected_points': X_candidates[selected_indices],
            'selected_indices': selected_indices,
            'acquisition_values': acq_values[selected_indices],
            'strategy': 'diversity',
            'diversity_scores': diversity_scores,
            'diversity_weight': diversity_weight
        }

    def _clustering_batch_selection(self, X_candidates: np.ndarray, acq_func, model,
                                  batch_size: int, **kwargs) -> Dict[str, Any]:
        """
        Select batch using clustering approach.

        Clusters candidate points and selects the best point from each cluster.
        """
        # Perform K-means clustering
        n_clusters = min(batch_size, len(X_candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_candidates)

        # Get acquisition values
        acq_values = acq_func.evaluate(X_candidates, model)

        selected_indices = []

        # Select best point from each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) > 0:
                # Find best acquisition value in this cluster
                cluster_acq_values = acq_values[cluster_indices]
                best_in_cluster = cluster_indices[np.argmax(cluster_acq_values)]
                selected_indices.append(best_in_cluster)

        return {
            'selected_points': X_candidates[selected_indices],
            'selected_indices': selected_indices,
            'acquisition_values': acq_values[selected_indices],
            'strategy': 'clustering',
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_
        }

    def _sequential_batch_selection(self, X_candidates: np.ndarray, acq_func, model,
                                  batch_size: int, **kwargs) -> Dict[str, Any]:
        """
        Select batch using sequential optimization.

        Selects points one by one, removing selected points from candidates.
        """
        selected_indices = []
        remaining_candidates = X_candidates.copy()
        remaining_indices = list(range(len(X_candidates)))

        all_acq_values = []

        for _ in range(batch_size):
            if len(remaining_candidates) == 0:
                break

            # Evaluate acquisition function on remaining candidates
            acq_values = acq_func.evaluate(remaining_candidates, model)
            all_acq_values.extend(acq_values)

            # Select best point
            best_local_idx = np.argmax(acq_values)
            best_global_idx = remaining_indices[best_local_idx]

            selected_indices.append(best_global_idx)

            # Remove selected point
            remaining_candidates = np.delete(remaining_candidates, best_local_idx, axis=0)
            remaining_indices.pop(best_local_idx)

        return {
            'selected_points': X_candidates[selected_indices],
            'selected_indices': selected_indices,
            'acquisition_values': np.array(all_acq_values)[:len(selected_indices)],
            'strategy': 'sequential'
        }

    def _hallucination_batch_selection(self, X_candidates: np.ndarray, acq_func, model,
                                     batch_size: int, **kwargs) -> Dict[str, Any]:
        """
        Select batch using hallucination (fantasy points).

        Simulates what the model would look like after adding each selected point.
        """
        n_hallucinations = kwargs.get('n_hallucinations', 10)

        selected_indices = []
        selected_acq_values = []

        for batch_idx in range(batch_size):
            if len(selected_indices) >= len(X_candidates):
                break

            # Remaining candidates
            remaining_mask = np.ones(len(X_candidates), dtype=bool)
            remaining_mask[selected_indices] = False
            remaining_indices = np.where(remaining_mask)[0]

            if len(remaining_indices) == 0:
                break

            X_remaining = X_candidates[remaining_indices]

            # Evaluate each remaining candidate with hallucinations
            candidate_scores = []

            for cand_idx, x_cand in enumerate(X_remaining):
                # Generate hallucination values at this point
                x_cand_2d = x_cand.reshape(1, -1)

                # Get predicted mean and std
                if hasattr(model, 'predict'):
                    pred_result = model.predict(x_cand_2d)
                    # Extract first output
                    first_output = list(pred_result.keys())[0]
                    pred_mean = pred_result[first_output]['prediction']
                    pred_std = pred_result[first_output]['uncertainty']['standard_deviation']
                else:
                    # Fallback
                    pred_mean = 0.0
                    pred_std = 1.0

                # Generate fantasy values
                fantasy_values = np.random.normal(pred_mean, pred_std, n_hallucinations)

                # Calculate expected acquisition improvement
                expected_improvement = 0.0
                for fantasy_val in fantasy_values:
                    # Simulate adding this point with fantasy_val
                    # For simplicity, just evaluate acquisition at other points
                    # In practice, you'd update the GP posterior

                    # Evaluate acquisition on remaining candidates
                    acq_vals = acq_func.evaluate(X_remaining, model)
                    expected_improvement += np.mean(acq_vals)

                expected_improvement /= n_hallucinations
                candidate_scores.append(expected_improvement)

            # Select candidate with highest expected improvement
            best_cand_idx = np.argmax(candidate_scores)
            best_global_idx = remaining_indices[best_cand_idx]

            selected_indices.append(best_global_idx)
            selected_acq_values.append(candidate_scores[best_cand_idx])

        return {
            'selected_points': X_candidates[selected_indices],
            'selected_indices': selected_indices,
            'acquisition_values': np.array(selected_acq_values),
            'strategy': 'hallucination',
            'n_hallucinations': n_hallucinations
        }

    def _local_penalization_batch_selection(self, X_candidates: np.ndarray, acq_func, model,
                                          batch_size: int, **kwargs) -> Dict[str, Any]:
        """
        Select batch using local penalization.

        Applies penalties around selected points to encourage diversity.
        """
        penalty_radius = kwargs.get('penalty_radius', 0.1)
        penalty_strength = kwargs.get('penalty_strength', 1.0)

        # Get initial acquisition values
        acq_values = acq_func.evaluate(X_candidates, model).copy()

        # Calculate pairwise distances
        distances = squareform(pdist(X_candidates))

        selected_indices = []

        for _ in range(batch_size):
            # Select point with highest current acquisition value
            best_idx = np.argmax(acq_values)
            selected_indices.append(best_idx)

            # Apply penalty around selected point
            penalty_mask = distances[best_idx] <= penalty_radius
            penalty = penalty_strength * acq_values[best_idx] * np.exp(-distances[best_idx] / penalty_radius)

            # Apply penalty to nearby points
            acq_values[penalty_mask] -= penalty[penalty_mask]

            # Set selected point acquisition to -inf to avoid reselection
            acq_values[best_idx] = -np.inf

        # Get original acquisition values for selected points
        original_acq_values = acq_func.evaluate(X_candidates[selected_indices], model)

        return {
            'selected_points': X_candidates[selected_indices],
            'selected_indices': selected_indices,
            'acquisition_values': original_acq_values,
            'strategy': 'local_penalization',
            'penalty_radius': penalty_radius,
            'penalty_strength': penalty_strength
        }

    def optimize_batch_parameters(self, X_candidates: np.ndarray, model,
                                batch_size: int, optimization_budget: int = 50) -> Dict[str, Any]:
        """
        Optimize batch selection parameters using cross-validation.

        Args:
            X_candidates: Candidate points
            model: Trained model
            batch_size: Target batch size
            optimization_budget: Number of optimization iterations

        Returns:
            Optimized parameters
        """
        if self.batch_strategy == 'diversity':
            return self._optimize_diversity_parameters(X_candidates, model, batch_size, optimization_budget)
        elif self.batch_strategy == 'local_penalization':
            return self._optimize_penalization_parameters(X_candidates, model, batch_size, optimization_budget)
        else:
            return {'message': f'Parameter optimization not implemented for {self.batch_strategy}'}

    def _optimize_diversity_parameters(self, X_candidates: np.ndarray, model,
                                     batch_size: int, budget: int) -> Dict[str, Any]:
        """Optimize diversity weight parameter"""
        diversity_weights = np.linspace(0.1, 0.9, budget)
        best_weight = 0.5
        best_score = -np.inf

        for weight in diversity_weights:
            # Select batch with this weight
            result = self._diversity_batch_selection(
                X_candidates,
                AcquisitionFunctionFactory.create(self.acquisition_function_name),
                model, batch_size, diversity_weight=weight
            )

            # Score based on acquisition values and diversity
            acq_values = result['acquisition_values']
            diversity_scores = result['diversity_scores']

            # Combined score
            score = np.mean(acq_values) + weight * np.mean(diversity_scores)

            if score > best_score:
                best_score = score
                best_weight = weight

        return {
            'optimal_diversity_weight': best_weight,
            'optimization_score': best_score,
            'parameter_range_tested': (diversity_weights[0], diversity_weights[-1])
        }

    def _optimize_penalization_parameters(self, X_candidates: np.ndarray, model,
                                        batch_size: int, budget: int) -> Dict[str, Any]:
        """Optimize local penalization parameters"""
        # Grid search over penalty radius and strength
        n_params = int(np.sqrt(budget))

        penalty_radii = np.linspace(0.05, 0.5, n_params)
        penalty_strengths = np.linspace(0.5, 2.0, n_params)

        best_radius = 0.1
        best_strength = 1.0
        best_score = -np.inf

        for radius in penalty_radii:
            for strength in penalty_strengths:
                result = self._local_penalization_batch_selection(
                    X_candidates,
                    AcquisitionFunctionFactory.create(self.acquisition_function_name),
                    model, batch_size,
                    penalty_radius=radius, penalty_strength=strength
                )

                # Score based on acquisition values
                score = np.mean(result['acquisition_values'])

                if score > best_score:
                    best_score = score
                    best_radius = radius
                    best_strength = strength

        return {
            'optimal_penalty_radius': best_radius,
            'optimal_penalty_strength': best_strength,
            'optimization_score': best_score
        }

    def evaluate_batch_quality(self, selected_points: np.ndarray, X_candidates: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the quality of a selected batch.

        Args:
            selected_points: Selected batch points
            X_candidates: All candidate points

        Returns:
            Quality metrics
        """
        # Space-filling quality
        space_filling = self._calculate_space_filling_quality(selected_points, X_candidates)

        # Diversity within batch
        internal_diversity = self._calculate_internal_diversity(selected_points)

        # Coverage of design space
        coverage = self._calculate_design_space_coverage(selected_points, X_candidates)

        return {
            'space_filling_quality': space_filling,
            'internal_diversity': internal_diversity,
            'design_space_coverage': coverage,
            'overall_quality': (space_filling + internal_diversity + coverage) / 3
        }

    def _calculate_space_filling_quality(self, selected_points: np.ndarray, all_points: np.ndarray) -> float:
        """Calculate how well the batch fills the design space"""
        if len(selected_points) <= 1:
            return 0.0

        # Minimum distance between selected points
        distances = pdist(selected_points)
        min_distance = np.min(distances)

        # Normalize by design space size
        all_distances = pdist(all_points)
        max_possible_distance = np.max(all_distances)

        return min_distance / max_possible_distance if max_possible_distance > 0 else 0.0

    def _calculate_internal_diversity(self, selected_points: np.ndarray) -> float:
        """Calculate diversity within the selected batch"""
        if len(selected_points) <= 1:
            return 1.0

        distances = pdist(selected_points)
        return float(np.mean(distances))

    def _calculate_design_space_coverage(self, selected_points: np.ndarray, all_points: np.ndarray) -> float:
        """Calculate how well the batch covers the design space"""
        if len(selected_points) == 0:
            return 0.0

        # Calculate convex hull volume (simplified for high dimensions)
        try:
            from scipy.spatial import ConvexHull

            if selected_points.shape[1] <= 3:  # Only for low dimensions
                hull_selected = ConvexHull(selected_points)
                hull_all = ConvexHull(all_points)
                return hull_selected.volume / hull_all.volume
            else:
                # For high dimensions, use range coverage
                selected_ranges = np.ptp(selected_points, axis=0)
                all_ranges = np.ptp(all_points, axis=0)
                coverage_per_dim = selected_ranges / (all_ranges + 1e-10)
                return float(np.mean(coverage_per_dim))

        except Exception:
            # Fallback: range-based coverage
            selected_ranges = np.ptp(selected_points, axis=0)
            all_ranges = np.ptp(all_points, axis=0)
            coverage_per_dim = selected_ranges / (all_ranges + 1e-10)
            return float(np.mean(coverage_per_dim))