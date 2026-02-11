"""
Model Evaluation Module for Matrix Factorization Recommender.

Implements evaluation metrics:
- RMSE: Root Mean Square Error for rating prediction
- NDCG@K: Normalized Discounted Cumulative Gain using quantities as relevance
- Precision@K: Precision of top-K recommendations
- Recall@K: Recall of top-K recommendations

NDCG with quantity-based relevance is chosen because:
1. It captures ranking quality (order of recommendations matters)
2. Uses average item quantity as relevance score (higher quantity = more relevant)
3. Normalized to [0,1] range for easy interpretation
4. Industry standard for recommendation evaluation
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Evaluation Metrics
# ============================================================================

def calculate_dcg(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain.
    
    DCG = Σ (rel_i / log2(i + 1)) for i = 1 to k
    
    Args:
        relevances: List of relevance scores
        k: Cutoff position
        
    Returns:
        DCG score
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        dcg += rel / np.log2(i + 1)
    return dcg


def calculate_ndcg(
    recommended_items: List[str],
    ground_truth: Dict[str, float],
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Uses item quantities as relevance scores:
    - Higher quantity items are more relevant
    - NDCG rewards ranking high-relevance items at the top
    
    Args:
        recommended_items: Ordered list of recommended item IDs
        ground_truth: Dict mapping item_id → average quantity (relevance)
        k: Cutoff for NDCG@K
        
    Returns:
        NDCG@K score in [0, 1]
    """
    if not ground_truth:
        return 0.0
    
    # Get relevances for recommended items
    relevances = [ground_truth.get(item, 0.0) for item in recommended_items[:k]]
    
    # DCG for actual ranking
    dcg = calculate_dcg(relevances, k)
    
    # IDCG: ideal DCG (perfect ranking by relevance)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[:k]
    idcg = calculate_dcg(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_precision_at_k(
    recommended_items: List[str],
    relevant_items: set,
    k: int = 10
) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = |relevant ∩ recommended@K| / K
    
    Args:
        recommended_items: Ordered list of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        Precision@K score
    """
    recommended_at_k = set(recommended_items[:k])
    hits = len(recommended_at_k & relevant_items)
    return hits / k if k > 0 else 0.0


def calculate_recall_at_k(
    recommended_items: List[str],
    relevant_items: set,
    k: int = 10
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = |relevant ∩ recommended@K| / |relevant|
    
    Args:
        recommended_items: Ordered list of recommended item IDs
        relevant_items: Set of relevant item IDs
        k: Cutoff position
        
    Returns:
        Recall@K score
    """
    if not relevant_items:
        return 0.0
    
    recommended_at_k = set(recommended_items[:k])
    hits = len(recommended_at_k & relevant_items)
    return hits / len(relevant_items)


# ============================================================================
# Model Evaluator
# ============================================================================

class ModelEvaluator:
    """
    Evaluates Matrix Factorization model on test data.
    
    Computes ranking metrics using quantity as relevance.
    """
    
    def __init__(
        self,
        model_dir: str,
        k: int = 10,
        device: str = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model_dir: Path to model artifacts directory
            k: Cutoff for top-K metrics
            device: Device for inference (auto-detected if None)
        """
        self.model_dir = Path(model_dir)
        self.k = k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model artifacts from directory."""
        logger.info(f"Loading model from: {self.model_dir}")
        
        # Load mappings
        with open(self.model_dir / "mappings.pkl", "rb") as f:
            self.mappings = pickle.load(f)
        
        self.user2idx = self.mappings['user2idx']
        self.item2idx = self.mappings['item2idx']
        self.idx2user = self.mappings['idx2user']
        self.idx2item = self.mappings['idx2item']
        self.n_users = self.mappings['n_users']
        self.n_items = self.mappings['n_items']
        
        # Load config
        with open(self.model_dir / "config.json", "r") as f:
            self.config = json.load(f)
        
        # Load model
        self._load_model()
        
        logger.info(f"  Users: {self.n_users:,}")
        logger.info(f"  Items: {self.n_items:,}")
    
    def _load_model(self):
        """Load trained PyTorch model."""
        from train_model import MatrixFactorization
        
        self.model = MatrixFactorization(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.config['n_factors']
        )
        
        # Load checkpoint
        checkpoint_path = self.model_dir / "model.pt"
        if checkpoint_path.exists():
            # PyTorch 2.6+ defaults to weights_only=True, but our checkpoint
            # contains numpy scalars, so we need weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("  ✓ Loaded model weights")
        else:
            logger.warning("  ! No checkpoint found")
        
        self.model.to(self.device)
        self.model.eval()
    
    def calculate_rmse(
        self,
        df: pd.DataFrame,
        user_col: str,
        item_col: str,
        rating_col: str,
        batch_size: int = 4096
    ) -> float:
        """
        Calculate RMSE for the provided data.
        """
        # Filter known users/items
        valid_mask = (df[user_col].isin(self.user2idx)) & (df[item_col].isin(self.item2idx))
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            logger.warning("No valid user-item pairs found for RMSE calculation")
            return 0.0
            
        # Map to indices
        user_indices = torch.tensor([self.user2idx[u] for u in valid_df[user_col]], dtype=torch.long)
        item_indices = torch.tensor([self.item2idx[i] for i in valid_df[item_col]], dtype=torch.long)
        ratings = torch.tensor(valid_df[rating_col].values, dtype=torch.float32)
        
        self.model.eval()
        total_sq_error = 0.0
        count = 0
        
        with torch.no_grad():
            for i in range(0, len(user_indices), batch_size):
                u_batch = user_indices[i:i+batch_size].to(self.device)
                i_batch = item_indices[i:i+batch_size].to(self.device)
                r_batch = ratings[i:i+batch_size].to(self.device)
                
                preds = self.model(u_batch, i_batch)
                total_sq_error += torch.sum((preds - r_batch) ** 2).item()
                count += len(r_batch)
                
        return np.sqrt(total_sq_error / count) if count > 0 else 0.0

    def get_user_recommendations(
        self,
        user_id: str,
        exclude_items: Optional[set] = None,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get top-K recommendations for a user.
        
        Args:
            user_id: User identifier
            exclude_items: Items to exclude (e.g., already purchased)
            top_k: Number of recommendations (default: self.k)
            
        Returns:
            List of (item_id, score) tuples
        """
        top_k = top_k or self.k
        
        if user_id not in self.user2idx:
            return []
        
        user_idx = self.user2idx[user_id]
        
        # Get scores for all items
        with torch.no_grad():
            scores = self.model.predict_all_items(user_idx)
            scores = scores.cpu().numpy()
        
        # Create item-score pairs
        item_scores = [
            (self.idx2item[idx], float(scores[idx]))
            for idx in range(self.n_items)
        ]
        
        # Exclude items if specified
        if exclude_items:
            item_scores = [
                (item, score) for item, score in item_scores
                if item not in exclude_items
            ]
        
        # Sort by score and return top-K
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:top_k]
    
    def evaluate_on_test_data(
        self,
        test_dir: str,
        user_col: str = "network_userid",
        item_col: str = "item_id",
        rating_col: str = "item_quantity",
        sample_users: Optional[int] = None
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            test_dir: Path to test data (partitioned Parquet)
            user_col: User column name
            item_col: Item column name
            rating_col: Rating column name
            sample_users: Number of users to sample (None = all)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        # Load test data
        test_df = self._load_test_data(test_dir)
        
        logger.info(f"Test records: {len(test_df):,}")
        
        # Aggregate to user-item level
        user_item_ratings = test_df.groupby(
            [user_col, item_col],
            as_index=False
        )[rating_col].mean()
        
        logger.info(f"User-item pairs: {len(user_item_ratings):,}")
        
        # Calculate RMSE on test data
        logger.info("Calculating RMSE on test data...")
        rmse = self.calculate_rmse(user_item_ratings, user_col, item_col, rating_col)
        logger.info(f"Test RMSE: {rmse:.4f}")
        
        # Get unique users
        unique_users = user_item_ratings[user_col].unique()
        
        if sample_users and sample_users < len(unique_users):
            unique_users = np.random.choice(
                unique_users, size=sample_users, replace=False
            )
            logger.info(f"Sampling {sample_users} users for evaluation")
        
        # Evaluate per user
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        
        evaluated_users = 0
        
        for user_id in unique_users:
            if user_id not in self.user2idx:
                continue
            
            # Get ground truth for this user
            user_data = user_item_ratings[user_item_ratings[user_col] == user_id]
            ground_truth = dict(zip(user_data[item_col], user_data[rating_col]))
            
            if not ground_truth:
                continue
            
            relevant_items = set(ground_truth.keys())
            
            # Get recommendations (exclude items from training if available)
            recommendations = self.get_user_recommendations(user_id)
            recommended_items = [item for item, _ in recommendations]
            
            # Calculate metrics
            ndcg = calculate_ndcg(recommended_items, ground_truth, self.k)
            precision = calculate_precision_at_k(recommended_items, relevant_items, self.k)
            recall = calculate_recall_at_k(recommended_items, relevant_items, self.k)
            
            ndcg_scores.append(ndcg)
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            evaluated_users += 1
        
        # Aggregate metrics
        metrics = {
            'rmse': rmse,
            f'ndcg@{self.k}': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            f'precision@{self.k}': float(np.mean(precision_scores)) if precision_scores else 0.0,
            f'recall@{self.k}': float(np.mean(recall_scores)) if recall_scores else 0.0,
            'num_users_evaluated': evaluated_users,
            'k': self.k
        }
        
        logger.info("="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        logger.info(f"Users evaluated: {evaluated_users:,}")
        logger.info(f"NDCG@{self.k}: {metrics[f'ndcg@{self.k}']:.4f}")
        logger.info(f"Precision@{self.k}: {metrics[f'precision@{self.k}']:.4f}")
        logger.info(f"Recall@{self.k}: {metrics[f'recall@{self.k}']:.4f}")
        
        return metrics
    
    def _load_test_data(self, test_dir: str) -> pd.DataFrame:
        """Load test data from partitioned Parquet."""
        test_path = Path(test_dir)
        
        partition_dirs = list(test_path.glob("date_partition=*"))
        
        if not partition_dirs:
            raise ValueError(f"No partitions found in {test_dir}")
        
        dfs = []
        for partition_dir in partition_dirs:
            parquet_files = list(partition_dir.glob("*.parquet"))
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)


def save_evaluation_metrics(
    metrics: Dict,
    rmse: float,
    metrics_dir: str,
    end_date: str
) -> str:
    """
    Save evaluation metrics to files.
    
    Args:
        metrics: Evaluation metrics dictionary
        rmse: Test RMSE from training
        metrics_dir: Base directory for metrics
        end_date: Training data end date
        
    Returns:
        Path to saved metrics directory
    """
    # Create dated eval folder
    eval_dir = Path(metrics_dir) / f"eval_{datetime.now().strftime('%Y_%m_%d')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV (summary)
    csv_data = {
        'eval_date': [end_date],
        'rmse': [rmse],
        'ndcg': [metrics.get('ndcg@10', 0.0)],
        'precision': [metrics.get('precision@10', 0.0)],
        'recall': [metrics.get('recall@10', 0.0)],
        'users_evaluated': [metrics.get('num_users_evaluated', 0)]
    }
    
    df = pd.DataFrame(csv_data)
    csv_path = eval_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics CSV: {csv_path}")
    
    # Save full metrics as JSON
    json_path = eval_dir / "metrics_full.json"
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics JSON: {json_path}")
    
    return str(eval_dir)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run model evaluation standalone."""
    import argparse
    import configparser
    
    parser = argparse.ArgumentParser(description="Evaluate MF Recommender Model")
    parser.add_argument(
        '-c', '--config_path',
        default="/workspace/src/main/resources/train_params.ini",
        help="Path to configuration file"
    )
    parser.add_argument(
        '--model_dir',
        default=None,
        help="Path to model directory"
    )
    parser.add_argument(
        '--test_dir',
        default=None,
        help="Path to test data"
    )
    parser.add_argument(
        '--sample_users',
        type=int,
        default=None,
        help="Number of users to sample"
    )
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config_path)
    
    model_dir = args.model_dir or config.get("training", "latest_model_dir")
    test_dir = args.test_dir or os.path.join(config.get("data", "train_base"), "test_data")
    k = config.getint("evaluation", "ndcg_k", fallback=10)
    
    evaluator = ModelEvaluator(model_dir=model_dir, k=k)
    
    metrics = evaluator.evaluate_on_test_data(
        test_dir=test_dir,
        user_col=config.get("model", "user_col"),
        item_col=config.get("model", "item_col"),
        rating_col=config.get("model", "rating_col"),
        sample_users=args.sample_users
    )
    
    # Use computed RMSE
    rmse = metrics.get('rmse', 0.0)
    
    # Save metrics
    save_evaluation_metrics(
        metrics=metrics,
        rmse=rmse,
        metrics_dir=config.get("training", "metrics_dir"),
        end_date=config.get("data", "end_date")
    )
    
    print(f"\n✓ Evaluation complete")
    print(f"  NDCG@{k}: {metrics.get(f'ndcg@{k}', 0):.4f}")
    print(f"  RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
