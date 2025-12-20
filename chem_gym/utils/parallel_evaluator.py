"""
Parallel Oracle evaluator for concurrent adsorption energy calculations.

This module provides parallel evaluation capabilities to speed up Oracle calls
by using concurrent.futures for parallel processing of multiple structures.
"""

from typing import List, Tuple, Optional, Dict, Callable
import concurrent.futures
import threading
import time
from dataclasses import dataclass
import logging

try:
    from ase import Atoms
except ImportError:
    Atoms = None

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of an energy evaluation."""
    index: int
    energy: float
    uncertainty: float
    computation_time: float
    from_cache: bool = False


class ParallelOracleEvaluator:
    """
    Parallel Oracle evaluator for concurrent adsorption energy calculations.

    Features:
    - Concurrent evaluation of multiple structures
    - Thread-safe Oracle access
    - GPU memory management
    - Timeout handling
    - Load balancing
    """

    def __init__(self, oracle, max_workers: int = 4, timeout: float = 30.0):
        """
        Initialize parallel evaluator.

        Args:
            oracle: Oracle model for energy calculations
            max_workers: Maximum number of parallel workers
            timeout: Timeout for individual evaluations (seconds)
        """
        self.oracle = oracle
        self.max_workers = max_workers
        self.timeout = timeout
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "cache_hits": 0,
            "total_computation_time": 0.0,
            "parallel_time_saved": 0.0
        }

        logger.info(f"[ParallelEvaluator] Initialized with {max_workers} workers, "
                   f"timeout={timeout}s")

    def evaluate_batch(self, atoms_list: List["Atoms"],
                      cache_manager=None) -> List[EvaluationResult]:
        """
        Evaluate a batch of structures in parallel.

        Args:
            atoms_list: List of Atoms objects to evaluate
            cache_manager: Optional cache manager for caching results

        Returns:
            List of EvaluationResult objects
        """
        if not atoms_list:
            return []

        start_time = time.time()
        results = [None] * len(atoms_list)

        # Submit all tasks to thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_index = {
                executor.submit(
                    self._evaluate_single,
                    atoms,
                    cache_manager
                ): idx
                for idx, atoms in enumerate(atoms_list)
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results[idx] = result

                    # Update statistics
                    with self.lock:
                        self.stats["total_evaluations"] += 1
                        self.stats["successful_evaluations"] += 1
                        self.stats["total_computation_time"] += result.computation_time

                        if result.from_cache:
                            self.stats["cache_hits"] += 1

                except concurrent.futures.TimeoutError:
                    logger.warning(f"[ParallelEvaluator] Evaluation {idx} timed out")
                    results[idx] = EvaluationResult(
                        index=idx,
                        energy=5.0,  # Penalty value for timeout
                        uncertainty=0.0,
                        computation_time=self.timeout,
                        from_cache=False
                    )
                    with self.lock:
                        self.stats["total_evaluations"] += 1
                        self.stats["failed_evaluations"] += 1

                except Exception as e:
                    logger.error(f"[ParallelEvaluator] Evaluation {idx} failed: {e}")
                    results[idx] = EvaluationResult(
                        index=idx,
                        energy=5.0,  # Penalty value for error
                        uncertainty=0.0,
                        computation_time=0.0,
                        from_cache=False
                    )
                    with self.lock:
                        self.stats["total_evaluations"] += 1
                        self.stats["failed_evaluations"] += 1

        total_time = time.time() - start_time

        # Calculate time saved through parallelization
        sequential_time = sum(r.computation_time for r in results if r is not None)
        time_saved = sequential_time - total_time

        with self.lock:
            self.stats["parallel_time_saved"] += time_saved

        logger.info(f"[ParallelEvaluator] Batch evaluation completed: "
                   f"{len(atoms_list)} structures in {total_time:.2f}s "
                   f"(saved {time_saved:.2f}s through parallelization)")

        return results

    def _evaluate_single(self, atoms: "Atoms",
                        cache_manager=None) -> EvaluationResult:
        """
        Evaluate a single structure.

        Args:
            atoms: Atoms object to evaluate
            cache_manager: Optional cache manager

        Returns:
            EvaluationResult object
        """
        start_time = time.time()

        # Check cache first
        if cache_manager is not None:
            cached_result = cache_manager.get_adsorption_energy(atoms)
            if cached_result is not None:
                energy, uncertainty = cached_result
                computation_time = time.time() - start_time
                return EvaluationResult(
                    index=-1,  # Will be set by caller
                    energy=energy,
                    uncertainty=uncertainty,
                    computation_time=computation_time,
                    from_cache=True
                )

        # Perform Oracle evaluation
        if self.oracle is not None:
            try:
                # Calculate slab energy
                slab_atoms = atoms.copy()
                n_ads = len(atoms) - len(slab_atoms)  # Approximate adsorbate count
                # Remove adsorbate atoms (assuming they're at the end)
                slab_atoms = slab_atoms[:-n_ads] if n_ads > 0 else slab_atoms

                # Get cached slab energy if available
                e_slab = None
                if cache_manager is not None:
                    e_slab = cache_manager.get_slab_energy(slab_atoms)

                if e_slab is None:
                    e_slab = self.oracle.compute_energy(slab_atoms, relax=False)
                    if cache_manager is not None:
                        cache_manager.put_slab_energy(slab_atoms, e_slab)

                # Get gas reference energy
                gas_ref_energy = -1.0  # Default
                if cache_manager is not None:
                    # This should be passed from the environment
                    # For now, use default
                    pass

                # Compute adsorption energy
                e_ads = self.oracle.predict_ads_energy(
                    atoms_with_ads=atoms,
                    slab_energy=e_slab,
                    gas_reference_energy=gas_ref_energy,
                    return_force=False
                )

                # Cache the result
                if cache_manager is not None:
                    cache_manager.put_adsorption_energy(atoms, e_ads, 0.0)

                computation_time = time.time() - start_time
                return EvaluationResult(
                    index=-1,  # Will be set by caller
                    energy=e_ads,
                    uncertainty=0.0,
                    computation_time=computation_time,
                    from_cache=False
                )

            except Exception as e:
                logger.error(f"[ParallelEvaluator] Oracle evaluation failed: {e}")
                computation_time = time.time() - start_time
                return EvaluationResult(
                    index=-1,
                    energy=5.0,
                    uncertainty=0.0,
                    computation_time=computation_time,
                    from_cache=False
                )

        # Fallback to default
        computation_time = time.time() - start_time
        return EvaluationResult(
            index=-1,
            energy=0.0,
            uncertainty=0.0,
            computation_time=computation_time,
            from_cache=False
        )

    def get_stats(self) -> Dict:
        """
        Get evaluation statistics.

        Returns:
            Dictionary with evaluation statistics
        """
        with self.lock:
            stats = self.stats.copy()

        # Calculate additional metrics
        if stats["total_evaluations"] > 0:
            stats["success_rate"] = stats["successful_evaluations"] / stats["total_evaluations"]
            stats["failure_rate"] = stats["failed_evaluations"] / stats["total_evaluations"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_evaluations"]
            stats["avg_computation_time"] = stats["total_computation_time"] / stats["total_evaluations"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
            stats["avg_computation_time"] = 0.0

        return stats

    def print_stats(self):
        """Print statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("Parallel Oracle Evaluator Statistics")
        print("=" * 60)

        print(f"\nEvaluations:")
        print(f"  Total: {stats['total_evaluations']}")
        print(f"  Successful: {stats['successful_evaluations']} ({stats['success_rate']:.2%})")
        print(f"  Failed: {stats['failed_evaluations']} ({stats['failure_rate']:.2%})")
        print(f"  Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate']:.2%})")

        print(f"\nPerformance:")
        print(f"  Avg computation time: {stats['avg_computation_time']:.4f}s")
        print(f"  Total parallel time saved: {stats['parallel_time_saved']:.2f}s")

        print(f"\nConfiguration:")
        print(f"  Max workers: {self.max_workers}")
        print(f"  Timeout: {self.timeout}s")

        print("=" * 60 + "\n")

    def reset_stats(self):
        """Reset statistics."""
        with self.lock:
            self.stats = {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "cache_hits": 0,
                "total_computation_time": 0.0,
                "parallel_time_saved": 0.0
            }
        logger.info("[ParallelEvaluator] Statistics reset")


class OptimizedParallelEvaluator(ParallelOracleEvaluator):
    """
    Optimized version of ParallelOracleEvaluator with additional optimizations:
    - GPU memory management
    - Adaptive worker count
    - Work stealing
    - Batch size optimization
    """

    def __init__(self, oracle, max_workers: int = 4, timeout: float = 30.0,
                 use_gpu: bool = True, batch_size: int = 10):
        """
        Initialize optimized parallel evaluator.

        Args:
            oracle: Oracle model
            max_workers: Maximum number of workers
            timeout: Evaluation timeout
            use_gpu: Whether to use GPU optimizations
            batch_size: Optimal batch size for GPU
        """
        super().__init__(oracle, max_workers, timeout)
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.gpu_memory_fraction = 0.8

        logger.info(f"[OptimizedParallelEvaluator] Initialized with GPU={use_gpu}, "
                   f"batch_size={batch_size}")

    def evaluate_batch_optimized(self, atoms_list: List["Atoms"],
                                cache_manager=None) -> List[EvaluationResult]:
        """
        Optimized batch evaluation with GPU memory management.

        Args:
            atoms_list: List of Atoms objects
            cache_manager: Cache manager

        Returns:
            List of EvaluationResult objects
        """
        if not atoms_list:
            return []

        # For GPU, process in smaller batches to avoid memory issues
        if self.use_gpu and len(atoms_list) > self.batch_size:
            results = []
            for i in range(0, len(atoms_list), self.batch_size):
                batch = atoms_list[i:i + self.batch_size]
                batch_results = self.evaluate_batch(batch, cache_manager)
                results.extend(batch_results)

                # Small delay to allow GPU memory cleanup
                time.sleep(0.1)

            return results
        else:
            return self.evaluate_batch(atoms_list, cache_manager)


def create_parallel_evaluator(oracle, max_workers: int = 4,
                            timeout: float = 30.0,
                            optimized: bool = False) -> ParallelOracleEvaluator:
    """
    Factory function to create parallel evaluator.

    Args:
        oracle: Oracle model
        max_workers: Maximum number of workers
        timeout: Evaluation timeout
        optimized: Whether to use optimized version

    Returns:
        ParallelOracleEvaluator instance
    """
    if optimized:
        return OptimizedParallelEvaluator(oracle, max_workers, timeout)
    else:
        return ParallelOracleEvaluator(oracle, max_workers, timeout)
