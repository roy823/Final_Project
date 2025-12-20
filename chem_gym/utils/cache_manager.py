"""
Intelligent caching manager for adsorption energy calculations.

This module provides caching mechanisms to optimize Oracle calls by:
1. Caching adsorption energies based on surface structure
2. Pre-computing and caching gas reference energies
3. Caching slab energies (surface without adsorbate)
4. LRU-based cache eviction for memory management
"""

from typing import Dict, Optional, Tuple
import hashlib
import time
from collections import OrderedDict
import threading
import logging

try:
    from ase import Atoms
except ImportError:
    Atoms = None

logger = logging.getLogger(__name__)


class LRUCache:
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Tuple[float, float]]:
        """Get value from cache. Returns None if not found."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None

    def put(self, key: str, value: Tuple[float, float]):
        """Put value into cache. Evicts oldest if at capacity."""
        with self.lock:
            if key in self.cache:
                # Update existing key
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Evict oldest (least recently used)
                self.cache.popitem(last=False)
            self.cache[key] = value

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)

    def stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0
            }


class AdsorptionCacheManager:
    """
    Intelligent cache manager for adsorption energy calculations.

    Features:
    - MD5 hash-based caching of surface structures
    - Pre-computation and caching of gas reference energies
    - Slab energy caching (surface without adsorbate)
    - LRU-based cache eviction
    - Thread-safe operations
    - Cache hit rate monitoring
    """

    def __init__(self, max_size: int = 1000, enable_stats: bool = True):
        """
        Initialize cache manager.

        Args:
            max_size: Maximum number of cached entries
            enable_stats: Enable cache hit rate statistics
        """
        self.adsorption_cache = LRUCache(max_size)
        self.slab_cache = LRUCache(max_size // 2)  # Smaller cache for slabs
        self.reference_cache = LRUCache(100)  # Cache for gas references

        self.enable_stats = enable_stats
        self.reset_stats()

        logger.info(f"[CacheManager] Initialized with max_size={max_size}")

    def reset_stats(self):
        """Reset cache statistics."""
        if self.enable_stats:
            self.stats = {
                "adsorption_hits": 0,
                "adsorption_misses": 0,
                "slab_hits": 0,
                "slab_misses": 0,
                "reference_hits": 0,
                "reference_misses": 0,
                "total_calls": 0
            }

    def _get_atoms_hash(self, atoms: "Atoms", include_tags: bool = False) -> str:
        """
        Generate MD5 hash for Atoms object.

        Args:
            atoms: ASE Atoms object
            include_tags: Whether to include atomic tags in hash

        Returns:
            MD5 hash string
        """
        # Get symbols and positions
        symbols = ''.join(atoms.get_chemical_symbols())
        positions = atoms.get_positions()

        # Round positions to avoid floating point precision issues
        positions_rounded = positions.round(3)

        # Create hash input
        if include_tags and hasattr(atoms, 'get_tags'):
            tags = atoms.get_tags()
            hash_input = f"{symbols}_{positions_rounded.tobytes()}_{tags.tobytes()}"
        else:
            hash_input = f"{symbols}_{positions_rounded.tobytes()}"

        # Generate MD5 hash
        return hashlib.md5(hash_input.encode()).hexdigest()

    def get_adsorption_energy(self, atoms: "Atoms") -> Optional[Tuple[float, float]]:
        """
        Get cached adsorption energy for atoms structure.

        Args:
            atoms: Atoms object with surface + adsorbate

        Returns:
            Tuple of (energy, uncertainty) or None if not cached
        """
        cache_key = self._get_atoms_hash(atoms, include_tags=True)
        result = self.adsorption_cache.get(cache_key)

        if self.enable_stats:
            self.stats["total_calls"] += 1
            if result is not None:
                self.stats["adsorption_hits"] += 1
                logger.debug(f"[CacheManager] Adsorption cache hit: {cache_key[:8]}...")
            else:
                self.stats["adsorption_misses"] += 1
                logger.debug(f"[CacheManager] Adsorption cache miss: {cache_key[:8]}...")

        return result

    def put_adsorption_energy(self, atoms: "Atoms", energy: float, uncertainty: float = 0.0):
        """
        Cache adsorption energy for atoms structure.

        Args:
            atoms: Atoms object with surface + adsorbate
            energy: Adsorption energy value
            uncertainty: Uncertainty value
        """
        cache_key = self._get_atoms_hash(atoms, include_tags=True)
        self.adsorption_cache.put(cache_key, (energy, uncertainty))
        logger.debug(f"[CacheManager] Cached adsorption energy: {energy:.4f} eV")

    def get_slab_energy(self, slab_atoms: "Atoms") -> Optional[float]:
        """
        Get cached slab energy (surface without adsorbate).

        Args:
            slab_atoms: Atoms object for surface only

        Returns:
            Energy value or None if not cached
        """
        cache_key = self._get_atoms_hash(slab_atoms)
        result = self.slab_cache.get(cache_key)

        if self.enable_stats:
            if result is not None:
                self.stats["slab_hits"] += 1
                logger.debug(f"[CacheManager] Slab cache hit: {cache_key[:8]}...")
            else:
                self.stats["slab_misses"] += 1
                logger.debug(f"[CacheManager] Slab cache miss: {cache_key[:8]}...")

        # Unpack tuple if needed
        if result is not None:
            return result[0] if isinstance(result, tuple) else result
        return None

    def put_slab_energy(self, slab_atoms: "Atoms", energy: float):
        """
        Cache slab energy.

        Args:
            slab_atoms: Atoms object for surface only
            energy: Energy value
        """
        cache_key = self._get_atoms_hash(slab_atoms)
        self.slab_cache.put(cache_key, (energy, 0.0))
        logger.debug(f"[CacheManager] Cached slab energy: {energy:.4f} eV")

    def get_gas_reference_energy(self, adsorbate: str) -> Optional[float]:
        """
        Get cached gas reference energy for adsorbate.

        Args:
            adsorbate: Adsorbate type (e.g., "CO", "O2", "H2")

        Returns:
            Reference energy or None if not cached
        """
        result = self.reference_cache.get(adsorbate)

        if self.enable_stats:
            if result is not None:
                self.stats["reference_hits"] += 1
                logger.debug(f"[CacheManager] Reference cache hit: {adsorbate}")
            else:
                self.stats["reference_misses"] += 1
                logger.debug(f"[CacheManager] Reference cache miss: {adsorbate}")

        # Unpack tuple if needed
        if result is not None:
            return result[0] if isinstance(result, tuple) else result
        return None

    def put_gas_reference_energy(self, adsorbate: str, energy: float):
        """
        Cache gas reference energy.

        Args:
            adsorbate: Adsorbate type
            energy: Reference energy value
        """
        self.reference_cache.put(adsorbate, (energy, 0.0))
        logger.debug(f"[CacheManager] Cached reference energy for {adsorbate}: {energy:.4f} eV")

    def clear_all(self):
        """Clear all caches."""
        self.adsorption_cache.clear()
        self.slab_cache.clear()
        self.reference_cache.clear()
        logger.info("[CacheManager] All caches cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "adsorption_cache": self.adsorption_cache.stats(),
            "slab_cache": self.slab_cache.stats(),
            "reference_cache": self.reference_cache.stats(),
            "detailed_stats": self.stats.copy() if self.enable_stats else {}
        }

        # Calculate hit rates
        if self.enable_stats:
            total_ads = self.stats["adsorption_hits"] + self.stats["adsorption_misses"]
            total_slab = self.stats["slab_hits"] + self.stats["slab_misses"]
            total_ref = self.stats["reference_hits"] + self.stats["reference_misses"]

            stats["hit_rates"] = {
                "adsorption_hit_rate": self.stats["adsorption_hits"] / total_ads if total_ads > 0 else 0,
                "slab_hit_rate": self.stats["slab_hits"] / total_slab if total_slab > 0 else 0,
                "reference_hit_rate": self.stats["reference_hits"] / total_ref if total_ref > 0 else 0
            }

        return stats

    def print_stats(self):
        """Print cache statistics to console."""
        stats = self.get_cache_stats()

        print("\n" + "=" * 60)
        print("Cache Statistics")
        print("=" * 60)

        print(f"\nAdsorption Cache:")
        print(f"  Size: {stats['adsorption_cache']['size']}/{stats['adsorption_cache']['max_size']}")
        print(f"  Utilization: {stats['adsorption_cache']['utilization']:.2%}")

        print(f"\nSlab Cache:")
        print(f"  Size: {stats['slab_cache']['size']}/{stats['slab_cache']['max_size']}")
        print(f"  Utilization: {stats['slab_cache']['utilization']:.2%}")

        print(f"\nReference Cache:")
        print(f"  Size: {stats['reference_cache']['size']}/{stats['reference_cache']['max_size']}")
        print(f"  Utilization: {stats['reference_cache']['utilization']:.2%}")

        if self.enable_stats and "hit_rates" in stats:
            print(f"\nHit Rates:")
            print(f"  Adsorption: {stats['hit_rates']['adsorption_hit_rate']:.2%}")
            print(f"  Slab: {stats['hit_rates']['slab_hit_rate']:.2%}")
            print(f"  Reference: {stats['hit_rates']['reference_hit_rate']:.2%}")

        print("=" * 60 + "\n")


def create_cache_manager(max_size: int = 1000, enable_stats: bool = True) -> AdsorptionCacheManager:
    """
    Factory function to create cache manager.

    Args:
        max_size: Maximum cache size
        enable_stats: Enable statistics

    Returns:
        AdsorptionCacheManager instance
    """
    return AdsorptionCacheManager(max_size=max_size, enable_stats=enable_stats)
