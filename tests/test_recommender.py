"""
Unit Tests for Matrix Factorization Recommender System.

Run with: pytest tests/test_recommender.py -v
"""

import os
import sys
import pytest
import tempfile
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "main" / "python"))


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample transaction data."""
    np.random.seed(42)
    
    n_users = 100
    n_items = 50
    n_transactions = 1000
    
    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]
    
    data = {
        'network_userid': np.random.choice(users, n_transactions),
        'item_id': np.random.choice(items, n_transactions),
        'order_id': [f"order_{i//5}" for i in range(n_transactions)],
        'item_quantity': np.random.randint(1, 10, n_transactions).astype(float),
        'datestamp': pd.date_range('2025-09-01', periods=n_transactions, freq='h')
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        'n_factors': 16,
        'learning_rate': 0.01,
        'batch_size': 64,
        'epochs': 2,
        'reg_lambda': 0.01,
        'use_gpu': False
    }


@pytest.fixture
def sample_mappings():
    """Create sample ID mappings."""
    n_users = 100
    n_items = 50
    
    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]
    
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}
    
    return {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': idx2user,
        'idx2item': idx2item,
        'n_users': n_users,
        'n_items': n_items
    }


# ============================================================================
# Test Matrix Factorization Model
# ============================================================================

class TestMatrixFactorization:
    """Tests for MatrixFactorization model."""
    
    def test_model_creation(self, sample_mappings):
        """Test model can be created."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=16
        )
        
        assert model.n_users == 100
        assert model.n_items == 50
        assert model.n_factors == 16
    
    def test_model_forward(self, sample_mappings):
        """Test model forward pass."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=16
        )
        
        user_ids = torch.tensor([0, 1, 2, 3, 4])
        item_ids = torch.tensor([0, 1, 2, 3, 4])
        
        predictions = model(user_ids, item_ids)
        
        assert predictions.shape == (5,)
        assert not torch.isnan(predictions).any()
    
    def test_model_predict_all_items(self, sample_mappings):
        """Test predicting scores for all items."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=16
        )
        
        scores = model.predict_all_items(user_id=0)
        
        assert scores.shape == (sample_mappings['n_items'],)
        assert not torch.isnan(scores).any()
    
    def test_model_embeddings(self, sample_mappings):
        """Test embedding retrieval."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=16
        )
        
        user_ids = torch.tensor([0, 1, 2])
        item_ids = torch.tensor([0, 1, 2])
        
        user_emb = model.get_user_embedding(user_ids)
        item_emb = model.get_item_embedding(item_ids)
        
        assert user_emb.shape == (3, 16)
        assert item_emb.shape == (3, 16)


# ============================================================================
# Test Evaluation Metrics
# ============================================================================

class TestEvaluationMetrics:
    """Tests for evaluation metrics."""
    
    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        from eval_model import calculate_ndcg
        
        recommended = ['item_a', 'item_b', 'item_c']
        ground_truth = {'item_a': 3.0, 'item_b': 2.0, 'item_c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        # Perfect ranking should give NDCG = 1.0
        assert ndcg == pytest.approx(1.0, abs=1e-6)
    
    def test_ndcg_worst_ranking(self):
        """Test NDCG with worst ranking."""
        from eval_model import calculate_ndcg
        
        recommended = ['item_c', 'item_b', 'item_a']
        ground_truth = {'item_a': 3.0, 'item_b': 2.0, 'item_c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        # Worst ranking should give NDCG < 1.0
        assert ndcg < 1.0
        assert ndcg > 0.0
    
    def test_ndcg_no_hits(self):
        """Test NDCG with no relevant items."""
        from eval_model import calculate_ndcg
        
        recommended = ['item_x', 'item_y', 'item_z']
        ground_truth = {'item_a': 3.0, 'item_b': 2.0, 'item_c': 1.0}
        
        ndcg = calculate_ndcg(recommended, ground_truth, k=3)
        
        assert ndcg == 0.0
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        from eval_model import calculate_precision_at_k
        
        recommended = ['item_a', 'item_b', 'item_x', 'item_y', 'item_z']
        relevant = {'item_a', 'item_b', 'item_c'}
        
        precision = calculate_precision_at_k(recommended, relevant, k=5)
        
        # 2 hits in 5 recommendations
        assert precision == pytest.approx(2/5, abs=1e-6)
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        from eval_model import calculate_recall_at_k
        
        recommended = ['item_a', 'item_b', 'item_x', 'item_y', 'item_z']
        relevant = {'item_a', 'item_b', 'item_c'}
        
        recall = calculate_recall_at_k(recommended, relevant, k=5)
        
        # 2 hits out of 3 relevant items
        assert recall == pytest.approx(2/3, abs=1e-6)


# ============================================================================
# Test LSH Index
# ============================================================================

class TestLSHIndex:
    """Tests for LSH index."""
    
    def test_lsh_build(self):
        """Test LSH index building."""
        from recommend import LSHIndex
        
        n_items = 100
        n_factors = 16
        
        embeddings = np.random.randn(n_items, n_factors).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(n_items)]
        
        lsh = LSHIndex(n_hash_tables=3, n_hash_functions=5)
        lsh.build(embeddings, item_ids)
        
        assert len(lsh.hash_tables) == 3
        assert lsh.embeddings.shape == (n_items, n_factors)
    
    def test_lsh_query(self):
        """Test LSH query."""
        from recommend import LSHIndex
        
        n_items = 100
        n_factors = 16
        
        embeddings = np.random.randn(n_items, n_factors).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(n_items)]
        
        lsh = LSHIndex(n_hash_tables=5, n_hash_functions=10)
        lsh.build(embeddings, item_ids)
        
        query = embeddings[0]
        results = lsh.query(query, k=5, exclude_items={0})
        
        # Should return up to k results
        assert len(results) <= 5
        # First element should not be excluded item
        if results:
            assert results[0][0] != 0
    
    def test_lsh_save_load(self):
        """Test LSH index save and load."""
        from recommend import LSHIndex
        
        n_items = 50
        n_factors = 8
        
        embeddings = np.random.randn(n_items, n_factors).astype(np.float32)
        item_ids = [f"item_{i}" for i in range(n_items)]
        
        lsh = LSHIndex(n_hash_tables=3, n_hash_functions=5)
        lsh.build(embeddings, item_ids)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            lsh.save(f.name)
            
            loaded_lsh = LSHIndex.load(f.name)
            
            assert len(loaded_lsh.hash_tables) == 3
            assert loaded_lsh.embeddings.shape == (n_items, n_factors)
            
            os.unlink(f.name)


# ============================================================================
# Test Popularity Recommender
# ============================================================================

class TestPopularityRecommender:
    """Tests for popularity-based recommender."""
    
    def test_popularity_recommend(self):
        """Test popularity recommendations."""
        from recommend import PopularityRecommender
        
        popularity = {
            'item_a': 0.9,
            'item_b': 0.7,
            'item_c': 0.5,
            'item_d': 0.3,
            'item_e': 0.1
        }
        
        recommender = PopularityRecommender(popularity)
        
        recs = recommender.recommend(k=3)
        
        assert len(recs) == 3
        assert recs[0][0] == 'item_a'  # Most popular
        assert recs[1][0] == 'item_b'
        assert recs[2][0] == 'item_c'
    
    def test_popularity_exclude_items(self):
        """Test popularity with excluded items."""
        from recommend import PopularityRecommender
        
        popularity = {
            'item_a': 0.9,
            'item_b': 0.7,
            'item_c': 0.5
        }
        
        recommender = PopularityRecommender(popularity)
        
        recs = recommender.recommend(k=2, exclude_items={'item_a'})
        
        assert len(recs) == 2
        assert recs[0][0] == 'item_b'  # item_a excluded


# ============================================================================
# Test DCG Calculation
# ============================================================================

class TestDCG:
    """Tests for DCG calculation."""
    
    def test_dcg_basic(self):
        """Test basic DCG calculation."""
        from eval_model import calculate_dcg
        
        # DCG = rel_1/log2(2) + rel_2/log2(3) + rel_3/log2(4)
        # DCG = 3/1 + 2/1.585 + 1/2 = 3 + 1.262 + 0.5 = 4.762
        relevances = [3.0, 2.0, 1.0]
        
        dcg = calculate_dcg(relevances, k=3)
        
        expected = 3.0/np.log2(2) + 2.0/np.log2(3) + 1.0/np.log2(4)
        assert dcg == pytest.approx(expected, abs=1e-3)
    
    def test_dcg_empty(self):
        """Test DCG with empty list."""
        from eval_model import calculate_dcg
        
        dcg = calculate_dcg([], k=3)
        
        assert dcg == 0.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_model_save_load(self, sample_mappings, model_config):
        """Test model checkpoint save and load."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=model_config['n_factors']
        )
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': {},
                'epoch': 5,
                'train_rmse': 0.5,
                'test_rmse': 0.6
            }
            torch.save(checkpoint, checkpoint_path)
            
            # Load
            loaded_checkpoint = torch.load(checkpoint_path)
            
            new_model = MatrixFactorization(
                n_users=sample_mappings['n_users'],
                n_items=sample_mappings['n_items'],
                n_factors=model_config['n_factors']
            )
            new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
            
            # Compare predictions
            user_ids = torch.tensor([0, 1, 2])
            item_ids = torch.tensor([0, 1, 2])
            
            pred1 = model(user_ids, item_ids)
            pred2 = new_model(user_ids, item_ids)
            
            assert torch.allclose(pred1, pred2)
    
    def test_full_artifacts_save_load(self, sample_mappings, model_config):
        """Test saving and loading full model artifacts."""
        from train_model import MatrixFactorization
        
        model = MatrixFactorization(
            n_users=sample_mappings['n_users'],
            n_items=sample_mappings['n_items'],
            n_factors=model_config['n_factors']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save mappings
            with open(model_dir / "mappings.pkl", "wb") as f:
                pickle.dump(sample_mappings, f)
            
            # Save config
            with open(model_dir / "config.json", "w") as f:
                json.dump(model_config, f)
            
            # Save model
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'train_rmse': 0.4,
                'test_rmse': 0.5
            }
            torch.save(checkpoint, model_dir / "model.pt")
            
            # Verify all files exist
            assert (model_dir / "mappings.pkl").exists()
            assert (model_dir / "config.json").exists()
            assert (model_dir / "model.pt").exists()
            
            # Load and verify
            with open(model_dir / "mappings.pkl", "rb") as f:
                loaded_mappings = pickle.load(f)
            
            assert loaded_mappings['n_users'] == sample_mappings['n_users']
            assert loaded_mappings['n_items'] == sample_mappings['n_items']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
