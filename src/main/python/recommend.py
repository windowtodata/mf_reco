"""
Complementary Product Recommender CLI.

Provides basket-based recommendations using:
1. Matrix Factorization model for item embeddings
2. LSH (Locality Sensitive Hashing) for fast similar item lookup
3. Popularity-based fallback for cold start items

Usage:
    python recommend.py --basket item1,item2,item3
    python recommend.py --basket item1,item2 --top_k 5
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import logging
import time

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LSH for Fast Similar Item Lookup
# ============================================================================

class LSHIndex:
    """
    Locality Sensitive Hashing index for fast approximate nearest neighbor search.
    Uses random hyperplane hashing for cosine similarity.
    """
    
    def __init__(
        self,
        n_hash_tables: int = 5,
        n_hash_functions: int = 10,
        seed: int = 42
    ):
        self.n_hash_tables = n_hash_tables
        self.n_hash_functions = n_hash_functions
        self.seed = seed
        
        self.hash_tables: List[Dict[str, List[int]]] = []
        self.random_vectors: List[np.ndarray] = []
        self.embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[List[str]] = None
    
    def build(self, embeddings: np.ndarray, item_ids: List[str]):
        """Build LSH index from embeddings."""
        logger.info(f"Building LSH index: {self.n_hash_tables} tables, {self.n_hash_functions} functions")
        
        self.embeddings = embeddings
        self.item_ids = item_ids
        n_items, n_factors = embeddings.shape
        
        np.random.seed(self.seed)
        
        self.hash_tables = [{} for _ in range(self.n_hash_tables)]
        self.random_vectors = [
            np.random.randn(n_factors, self.n_hash_functions)
            for _ in range(self.n_hash_tables)
        ]
        
        for item_idx in range(n_items):
            emb = embeddings[item_idx]
            for table_idx in range(self.n_hash_tables):
                hash_key = self._hash(emb, table_idx)
                if hash_key not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_key] = []
                self.hash_tables[table_idx][hash_key].append(item_idx)
        
        logger.info(f"LSH index built for {n_items} items")
    
    def _hash(self, embedding: np.ndarray, table_idx: int) -> str:
        projections = embedding @ self.random_vectors[table_idx]
        bits = (projections > 0).astype(int)
        return ''.join(map(str, bits))
    
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_items: Optional[Set[int]] = None
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors to query embedding."""
        exclude_items = exclude_items or set()
        candidates = set()
        
        for table_idx in range(self.n_hash_tables):
            hash_key = self._hash(query_embedding, table_idx)
            if hash_key in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_key])
        
        candidates = candidates - exclude_items
        
        if not candidates:
            return []
        
        candidate_list = list(candidates)
        candidate_embeddings = self.embeddings[candidate_list]
        
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_norms = candidate_embeddings / (
            np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = candidate_norms @ query_norm
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [(candidate_list[i], float(similarities[i])) for i in top_indices]
    
    def save(self, path: str):
        data = {
            'n_hash_tables': self.n_hash_tables,
            'n_hash_functions': self.n_hash_functions,
            'seed': self.seed,
            'hash_tables': self.hash_tables,
            'random_vectors': [v.tolist() for v in self.random_vectors],
            'embeddings': self.embeddings.tolist(),
            'item_ids': self.item_ids
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'LSHIndex':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls(
            n_hash_tables=data['n_hash_tables'],
            n_hash_functions=data['n_hash_functions'],
            seed=data['seed']
        )
        index.hash_tables = data['hash_tables']
        index.random_vectors = [np.array(v) for v in data['random_vectors']]
        index.embeddings = np.array(data['embeddings'])
        index.item_ids = data['item_ids']
        return index


# ============================================================================
# Popularity-based Fallback Recommender
# ============================================================================

class PopularityRecommender:
    """Fallback recommender based on item popularity for cold-start items."""
    
    def __init__(self, item_popularity: Dict[str, float]):
        self.item_popularity = item_popularity
        self.sorted_items = sorted(
            item_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def recommend(
        self,
        k: int = 5,
        exclude_items: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        exclude_items = exclude_items or set()
        recommendations = []
        for item_id, popularity in self.sorted_items:
            if item_id not in exclude_items:
                recommendations.append((item_id, popularity))
                if len(recommendations) >= k:
                    break
        return recommendations
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.item_popularity, f)
    
    @classmethod
    def load(cls, path: str) -> 'PopularityRecommender':
        with open(path, 'rb') as f:
            item_popularity = pickle.load(f)
        return cls(item_popularity)


# ============================================================================
# Main Recommender Class
# ============================================================================

class ComplementaryProductRecommender:
    """
    Complementary product recommender using Matrix Factorization.
    
    Given a basket of items, recommends complementary products based on:
    1. Item embeddings from MF model
    2. LSH for fast similarity search
    3. Popularity fallback for cold-start items
    """
    
    def __init__(self, model_dir: str, lsh_params: Optional[Dict] = None):
        self.model_dir = Path(model_dir)
        self.lsh_params = lsh_params or {
            'n_hash_tables': 5,
            'n_hash_functions': 10
        }
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
        self._build_lsh_index()
        self._build_popularity_recommender()
    
    def _load_model(self):
        """Load MF model and mappings."""
        logger.info(f"Loading model from: {self.model_dir}")
        
        with open(self.model_dir / "mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        self.user2idx = mappings['user2idx']
        self.item2idx = mappings['item2idx']
        self.idx2user = mappings['idx2user']
        self.idx2item = mappings['idx2item']
        self.n_users = mappings['n_users']
        self.n_items = mappings['n_items']
        
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
        
        from train_model import MatrixFactorization
        
        self.model = MatrixFactorization(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.config['n_factors']
        )
        
        checkpoint_path = self.model_dir / "model.pt"
        if checkpoint_path.exists():
            # Load with weights_only=True for security
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("  ✓ Loaded model weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"  Items: {self.n_items:,}")
        logger.info(f"  Factors: {self.config['n_factors']}")
    
    def _build_lsh_index(self):
        """Build or load LSH index for fast item lookup."""
        lsh_path = self.model_dir / "lsh_index.pkl"
        
        if lsh_path.exists():
            self.lsh_index = LSHIndex.load(str(lsh_path))
        else:
            logger.info("Building new LSH index...")
            
            with torch.no_grad():
                item_embeddings = self.model.item_factors.weight.cpu().numpy()
            
            item_ids = [self.idx2item[i] for i in range(self.n_items)]
            
            self.lsh_index = LSHIndex(
                n_hash_tables=self.lsh_params['n_hash_tables'],
                n_hash_functions=self.lsh_params['n_hash_functions']
            )
            self.lsh_index.build(item_embeddings, item_ids)
            self.lsh_index.save(str(lsh_path))
    
    def _build_popularity_recommender(self):
        """Build or load popularity-based fallback recommender."""
        popularity_path = self.model_dir / "popularity.pkl"
        
        if popularity_path.exists():
            self.popularity_recommender = PopularityRecommender.load(str(popularity_path))
        else:
            logger.info("Building popularity fallback...")
            
            # Use item biases as proxy for popularity
            with torch.no_grad():
                item_biases = self.model.item_bias.weight.squeeze().cpu().numpy()
            
            item_popularity = {
                self.idx2item[i]: float(item_biases[i])
                for i in range(self.n_items)
            }
            
            self.popularity_recommender = PopularityRecommender(item_popularity)
            self.popularity_recommender.save(str(popularity_path))
    
    def get_item_embedding(self, item_id: str) -> Optional[np.ndarray]:
        """Get embedding for a single item."""
        if item_id not in self.item2idx:
            return None
        
        item_idx = self.item2idx[item_id]
        
        with torch.no_grad():
            embedding = self.model.item_factors(
                torch.tensor([item_idx], device=self.device)
            )
        
        return embedding.squeeze().cpu().numpy()
    
    def get_basket_embedding(self, basket: List[str]) -> np.ndarray:
        """
        Compute aggregate embedding for a basket of items.
        Uses mean pooling of individual item embeddings.
        """
        embeddings = []
        
        for item_id in basket:
            emb = self.get_item_embedding(item_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.config['n_factors'])
        
        return np.mean(embeddings, axis=0)
    
    def recommend(
        self,
        basket: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recommend complementary products for a basket.
        
        Args:
            basket: List of item IDs in current basket
            top_k: Number of recommendations to return
            
        Returns:
            List of (item_id, score) tuples
        """
        start_time = time.time()
        
        # Compute basket embedding
        basket_embedding = self.get_basket_embedding(basket)
        
        # Items to exclude (already in basket)
        exclude_items = set()
        for item_id in basket:
            if item_id in self.item2idx:
                exclude_items.add(self.item2idx[item_id])
        
        # Query LSH index for similar items
        candidates = self.lsh_index.query(
            basket_embedding,
            k=top_k * 3,  # Get more candidates for filtering
            exclude_items=exclude_items
        )
        
        # Convert to item IDs
        recommendations = [
            (self.idx2item[idx], score)
            for idx, score in candidates
        ]
        
        # If not enough recommendations, fill with popularity-based
        if len(recommendations) < top_k:
            exclude_ids = set(basket) | {item for item, _ in recommendations}
            
            popular_items = self.popularity_recommender.recommend(
                k=top_k - len(recommendations),
                exclude_items=exclude_ids
            )
            
            recommendations.extend(popular_items)
        
        recommendations = recommendations[:top_k]
        
        elapsed_time = (time.time() - start_time) * 1000
        logger.debug(f"Recommendation time: {elapsed_time:.2f}ms")
        
        return recommendations
    
    def recommend_similar_items(
        self,
        item_id: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Item ID to find similar items for
            top_k: Number of similar items to return
            
        Returns:
            List of (item_id, similarity) tuples
        """
        embedding = self.get_item_embedding(item_id)
        
        if embedding is None:
            logger.warning(f"Item {item_id} not found in model")
            return self.popularity_recommender.recommend(k=top_k)
        
        exclude_items = {self.item2idx[item_id]}
        
        candidates = self.lsh_index.query(
            embedding,
            k=top_k,
            exclude_items=exclude_items
        )
        
        return [(self.idx2item[idx], score) for idx, score in candidates]


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complementary Product Recommender CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python recommend.py --basket item1,item2,item3
  python recommend.py --basket item1,item2 --top_k 10
  python recommend.py --similar item1 --top_k 5
        """
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/workspace/models/latest",
        help="Path to model artifacts directory"
    )
    
    parser.add_argument(
        "--basket",
        type=str,
        help="Comma-separated list of item IDs in basket"
    )
    
    parser.add_argument(
        "--similar",
        type=str,
        help="Find items similar to this item ID"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of recommendations (default: 5)"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    
    args = parser.parse_args()
    
    # Load recommender
    recommender = ComplementaryProductRecommender(args.model_dir)
    
    if args.benchmark:
        # Run performance benchmark
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Sample items for testing
        sample_items = list(recommender.item2idx.keys())[:100]
        
        times = []
        for _ in range(100):
            basket = list(np.random.choice(sample_items, size=3, replace=False))
            start = time.time()
            recommender.recommend(basket, top_k=5)
            times.append((time.time() - start) * 1000)
        
        print(f"Recommendations per second: {1000 / np.mean(times):.1f}")
        print(f"Mean latency: {np.mean(times):.2f}ms")
        print(f"P50 latency: {np.percentile(times, 50):.2f}ms")
        print(f"P95 latency: {np.percentile(times, 95):.2f}ms")
        print(f"P99 latency: {np.percentile(times, 99):.2f}ms")
        print("="*60)
        
    elif args.basket:
        # Basket recommendations
        basket = [item.strip() for item in args.basket.split(",")]
        
        print(f"\nRecommendations for basket: {basket}")
        print("-" * 60)
        
        recommendations = recommender.recommend(basket, top_k=args.top_k)
        
        if recommendations:
            for item_id, score in recommendations:
                print(f"  - {item_id} (score: {score:.4f})")
        else:
            print("  No recommendations available")
        
        print()
        
    elif args.similar:
        # Similar items
        print(f"\nItems similar to: {args.similar}")
        print("-" * 60)
        
        similar_items = recommender.recommend_similar_items(
            args.similar,
            top_k=args.top_k
        )
        
        if similar_items:
            for item_id, similarity in similar_items:
                print(f"  - {item_id} (similarity: {similarity:.4f})")
        else:
            print("  No similar items found")
        
        print()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
