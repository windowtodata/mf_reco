"""
Unit Tests for Matrix Factorization Recommender.

Run with: pytest test_recommender.py -v
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import json
import pickle
from pathlib import Path


# ============================================================================
# Test Matrix Factorization Model
# ============================================================================

class TestMatrixFactorization:
    """Tests for the MatrixFactorization model."""
    
    def test_model_initialization(self):
        """Test model initializes with correct dimensions."""
        from train_model import MatrixFactorization
        
        n_users, n_items, n_factors = 100, 50, 16
        model = MatrixFactorization(n_users, n_items, n_factors)
        
        assert model.n_users == n_users
        assert model.n_items == n_items
        assert model.n_factors == n_factors
        assert model.user_factors.weight.shape == (n_users, n_factors)
        assert model.item_factors.weight.shape == (n_items, n_factors)
        assert model.user_bias.weight.shape == (n_users, 1)
        assert model.item_bias.weight.shape == (n_items, 1)
    
    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        from train_model import MatrixFactorization
        
        n_users, n_items, n_factors = 100, 50, 16
        batch_size = 32
        
        model = MatrixFactorization(n_users, n_items, n_factors)
        
        user_ids = torch.randint(0, n_users, (batch_size,))
        item_ids = torch.randint(0, n_items, (batch_size,))
        
        predictions = model(user_ids, item_ids)
        
        assert predictions.shape == (batch_size,)
        assert not torch.isnan(predictions).any()
    
    def test_predict_all_items(self):
        """Test predicting scores for all items."""
        from train_model import MatrixFactorization
        
        n_users, n_items, n_factors = 10, 20, 8
        model = MatrixFactorization(n_users, n_items, n_factors)
        
        scores = model.predict_all_items(user_id=0)
        
        assert scores.shape == (n_items,)
        assert not torch.isnan(scores).any()
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(10, 10, 8)
        
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        targets = torch.tensor([1.0, 2.0, 3.0])
        
        predictions = model(user_ids, item_ids)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        
        # Check gradients exist
        assert model.user_factors.weight.grad is not None
        assert model.item_factors.weight.grad is not None


# ============================================================================
# Test Evaluation Metrics
# ============================================================================

class TestEvaluationMetrics:
    """Tests for NDCG and other evaluation metrics."""
    
    def test_dcg_calculation(self):
        """Test DCG calculation."""
        from eval_model import calculate_dcg
        
        # Perfect ranking: [3, 2, 1]
        relevances = [3.0, 2.0, 1.0]
        dcg = calculate_dcg(relevances, k=3)
        
        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        expected = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
        
        assert abs(dcg - expected) < 1e-6
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking equals 1.0."""
        from eval_model import calculate_ndcg
        
        recommended = ['a', 'b', 'c']
        ground_truth = {'a': 3.0, 'b': 2.0, 'c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        assert abs(ndcg - 1.0) < 1e-6
    
    def test_ndcg_worst_ranking(self):
        """Test NDCG with reversed ranking is less than 1.0."""
        from eval_model import calculate_ndcg
        
        recommended = ['c', 'b', 'a']  # Worst order
        ground_truth = {'a': 3.0, 'b': 2.0, 'c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        assert ndcg < 1.0
        assert ndcg > 0.0
    
    def test_ndcg_no_hits(self):
        """Test NDCG with no relevant items returns 0."""
        from eval_model import calculate_ndcg
        
        recommended = ['x', 'y', 'z']
        ground_truth = {'a': 3.0, 'b': 2.0, 'c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        assert ndcg == 0.0
    
    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        from eval_model import calculate_recall_at_k
        
        recommended = ['a', 'b', 'x', 'y', 'z']
        relevant = {'a', 'b', 'c'}
        
        recall = calculate_recall_at_k(recommended, relevant, k=5)
        
        # 2 hits out of 3 relevant
        assert recall == 2 / 3


# ============================================================================
# Test LSH Index
# ============================================================================

class TestLSHIndex:
    """Tests for LSH similarity search."""
    
    def test_lsh_build_and_query(self):
        """Test building LSH index and querying."""
        from recommend import LSHIndex
        
        n_items = 100
        n_factors = 16
        
        # Create random embeddings
        embeddings = np.random.randn(n_items, n_factors).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(n_items)]
        
        # Build index
        lsh = LSHIndex(n_hash_tables=3, n_hash_functions=8)
        lsh.build(embeddings, item_ids)
        
        # Query
        query = embeddings[0]  # Query for first item
        results = lsh.query(query, k=5, exclude_items={0})
        
        assert len(results) <= 5
        assert all(isinstance(idx, int) for idx, _ in results)
        assert all(isinstance(score, float) for _, score in results)
    
    def test_lsh_save_load(self):
        """Test saving and loading LSH index."""
        from recommend import LSHIndex
        
        embeddings = np.random.randn(50, 8).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(50)]
        
        lsh = LSHIndex(n_hash_tables=2, n_hash_functions=4)
        lsh.build(embeddings, item_ids)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        try:
            lsh.save(temp_path)
            loaded_lsh = LSHIndex.load(temp_path)
            
            assert loaded_lsh.n_hash_tables == lsh.n_hash_tables
            assert loaded_lsh.n_hash_functions == lsh.n_hash_functions
            assert len(loaded_lsh.item_ids) == len(item_ids)
        finally:
            os.unlink(temp_path)
    
    def test_lsh_excludes_items(self):
        """Test that LSH properly excludes specified items."""
        from recommend import LSHIndex
        
        embeddings = np.random.randn(10, 4).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(10)]
        
        lsh = LSHIndex(n_hash_tables=5, n_hash_functions=4)
        lsh.build(embeddings, item_ids)
        
        # Exclude items 0, 1, 2
        exclude = {0, 1, 2}
        results = lsh.query(embeddings[0], k=10, exclude_items=exclude)
        
        result_indices = {idx for idx, _ in results}
        assert len(result_indices & exclude) == 0


# ============================================================================
# Test Popularity Recommender
# ============================================================================

class TestPopularityRecommender:
    """Tests for popularity-based fallback."""
    
    def test_popularity_ranking(self):
        """Test items are ranked by popularity."""
        from recommend import PopularityRecommender
        
        popularity = {
            'item_a': 10.0,
            'item_b': 5.0,
            'item_c': 15.0,
            'item_d': 1.0
        }
        
        recommender = PopularityRecommender(popularity)
        recs = recommender.recommend(k=4)
        
        # Should be sorted by popularity descending
        assert recs[0][0] == 'item_c'
        assert recs[1][0] == 'item_a'
        assert recs[2][0] == 'item_b'
        assert recs[3][0] == 'item_d'
    
    def test_popularity_excludes_items(self):
        """Test exclusion of items."""
        from recommend import PopularityRecommender
        
        popularity = {'a': 3.0, 'b': 2.0, 'c': 1.0}
        recommender = PopularityRecommender(popularity)
        
        recs = recommender.recommend(k=2, exclude_items={'a'})
        
        assert len(recs) == 2
        assert 'a' not in [item for item, _ in recs]


# ============================================================================
# Test ID Mappings
# ============================================================================

class TestIDMappings:
    """Tests for user/item ID mapping utilities."""
    
    def test_mapping_consistency(self):
        """Test that mappings are consistent."""
        user2idx = {'user_a': 0, 'user_b': 1, 'user_c': 2}
        idx2user = {0: 'user_a', 1: 'user_b', 2: 'user_c'}
        
        # Verify bidirectional mapping
        for user, idx in user2idx.items():
            assert idx2user[idx] == user
        
        for idx, user in idx2user.items():
            assert user2idx[user] == idx


# ============================================================================
# Test Configuration Loading
# ============================================================================

class TestConfiguration:
    """Tests for configuration parsing."""
    
    def test_config_defaults(self):
        """Test configuration has sensible defaults."""
        import configparser
        
        config = configparser.ConfigParser()
        
        # Test fallback values
        n_factors = config.getint('model', 'n_factors', fallback=32)
        learning_rate = config.getfloat('model', 'learning_rate', fallback=0.001)
        use_gpu = config.getboolean('training', 'use_gpu', fallback=True)
        
        assert n_factors == 32
        assert learning_rate == 0.001
        assert use_gpu is True


# ============================================================================
# Integration Test Helpers
# ============================================================================

def create_test_model_artifacts(temp_dir: str, n_users: int = 10, n_items: int = 20, n_factors: int = 8):
    """Create test model artifacts for integration tests."""
    from train_model import MatrixFactorization
    
    model_dir = Path(temp_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mappings
    user2idx = {f"user_{i}": i for i in range(n_users)}
    item2idx = {f"item_{i}": i for i in range(n_items)}
    idx2user = {i: f"user_{i}" for i in range(n_users)}
    idx2item = {i: f"item_{i}" for i in range(n_items)}
    
    mappings = {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': idx2user,
        'idx2item': idx2item,
        'n_users': n_users,
        'n_items': n_items
    }
    
    with open(model_dir / "mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)
    
    # Create config
    config = {
        'n_factors': n_factors,
        'learning_rate': 0.001,
        'epochs': 10,
        'n_users': n_users,
        'n_items': n_items
    }
    
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    # Create model checkpoint
    model = MatrixFactorization(n_users, n_items, n_factors)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'epoch': 10,
        'train_rmse': 0.5,
        'test_rmse': 0.6
    }
    
    torch.save(checkpoint, model_dir / "model.pt")
    
    # Create metrics
    metrics = {
        'train_rmse': 0.5,
        'test_rmse': 0.6,
        'epoch': 10
    }
    
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    return model_dir


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
