"""
Matrix Factorization Model Training with PyTorch and Ray Train.

Implements bias-aware Matrix Factorization model based on:
- Koren et al., "Matrix Factorization Techniques for Recommender Systems"

Uses Ray Train for distributed training and Ray Data for efficient data loading.
Compatible with Ray 2.51.0 checkpoint API.
"""

import os
import pickle
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ray
from ray import train as ray_train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig, Checkpoint
from ray.train.torch import TorchTrainer
import ray.data
from monitoring import MLflowLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PyTorch Matrix Factorization Model
# ============================================================================

class MatrixFactorization(nn.Module):
    """
    Bias-aware Matrix Factorization model.
    
    Prediction: r_ui = μ + b_u + b_i + p_u · q_i
    
    Where:
        - μ: global bias (average rating)
        - b_u: user bias
        - b_i: item bias  
        - p_u: user latent factors
        - q_i: item latent factors
    """
    
    def __init__(
        self, 
        n_users: int, 
        n_items: int, 
        n_factors: int = 32,
        sparse: bool = False
    ):
        """
        Initialize Matrix Factorization model.
        
        Args:
            n_users: Number of unique users
            n_items: Number of unique items
            n_factors: Dimension of latent factors
            sparse: Whether to use sparse gradients (for large vocabs)
        """
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # User embeddings and bias
        self.user_factors = nn.Embedding(n_users, n_factors, sparse=sparse)
        self.user_bias = nn.Embedding(n_users, 1, sparse=sparse)
        
        # Item embeddings and bias
        self.item_factors = nn.Embedding(n_items, n_factors, sparse=sparse)
        self.item_bias = nn.Embedding(n_items, 1, sparse=sparse)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small random values."""
        # Xavier/Glorot initialization for factors
        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.item_factors.weight)
        
        # Zero initialization for biases
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: predict ratings.
        
        Args:
            user_ids: Tensor of user indices [batch_size]
            item_ids: Tensor of item indices [batch_size]
            
        Returns:
            Predicted ratings [batch_size]
        """
        # Get embeddings
        user_emb = self.user_factors(user_ids)  # [batch, n_factors]
        item_emb = self.item_factors(item_ids)  # [batch, n_factors]
        
        # Dot product of latent factors
        dot_product = (user_emb * item_emb).sum(dim=1, keepdim=True)
        
        # Add biases
        user_b = self.user_bias(user_ids)  # [batch, 1]
        item_b = self.item_bias(item_ids)  # [batch, 1]
        
        # Final prediction: μ + b_u + b_i + p_u · q_i
        prediction = self.global_bias + user_b + item_b + dot_product
        
        return prediction.squeeze()
    
    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Get user latent factors."""
        return self.user_factors(user_ids)
    
    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get item latent factors."""
        return self.item_factors(item_ids)
    
    def predict_all_items(self, user_id: int) -> torch.Tensor:
        """
        Predict ratings for all items for a given user.
        
        Args:
            user_id: Single user index
            
        Returns:
            Predicted ratings for all items [n_items]
        """
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.user_factors.weight.device)
            user_emb = self.user_factors(user_tensor)  # [1, n_factors]
            user_b = self.user_bias(user_tensor)  # [1, 1]
            
            # Compute scores for all items
            all_item_embs = self.item_factors.weight  # [n_items, n_factors]
            all_item_biases = self.item_bias.weight  # [n_items, 1]
            
            # Dot product with all items
            scores = torch.matmul(all_item_embs, user_emb.T).squeeze()  # [n_items]
            
            # Add biases
            scores = self.global_bias + user_b.squeeze() + all_item_biases.squeeze() + scores
            
            return scores


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_partitioned_parquet(data_dir: str) -> ray.data.Dataset:
    """
    Load data from date-partitioned Parquet directories.
    
    Args:
        data_dir: Path to directory containing date_partition=* folders
        
    Returns:
        Ray Dataset
    """
    logger.info(f"Loading partitioned Parquet from: {data_dir}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    partition_dirs = list(data_path.glob("date_partition=*"))
    
    if not partition_dirs:
        raise ValueError(f"No date_partition= folders found in {data_dir}")
    
    logger.info(f"Found {len(partition_dirs)} date partitions")
    
    # Collect all parquet files
    parquet_files = []
    for partition_dir in sorted(partition_dirs):
        files = list(partition_dir.glob("*.parquet"))
        parquet_files.extend([str(f) for f in files])
    
    if not parquet_files:
        raise ValueError(f"No .parquet files found in {data_dir}")
    
    logger.info(f"Total parquet files: {len(parquet_files)}")
    
    # Read with Ray Data
    ds = ray.data.read_parquet(parquet_files)
    
    count = ds.count()
    logger.info(f"Loaded dataset with {count:,} records")
    
    return ds


def create_id_mappings(
    train_ds: ray.data.Dataset,
    test_ds: ray.data.Dataset,
    user_col: str = "network_userid",
    item_col: str = "item_id"
) -> Dict:
    """
    Create user and item ID mappings from datasets.
    
    Args:
        train_ds: Training dataset
        test_ds: Test dataset
        user_col: User column name
        item_col: Item column name
        
    Returns:
        Dictionary with mappings and counts
    """
    logger.info("Creating ID mappings...")
    
    # Collect unique users and items from both datasets
    train_users = set(
        row[user_col] for row in 
        train_ds.select_columns([user_col]).take_all()
    )
    test_users = set(
        row[user_col] for row in 
        test_ds.select_columns([user_col]).take_all()
    )
    
    train_items = set(
        row[item_col] for row in 
        train_ds.select_columns([item_col]).take_all()
    )
    test_items = set(
        row[item_col] for row in 
        test_ds.select_columns([item_col]).take_all()
    )
    
    # Combine to get all unique IDs
    all_users = sorted(train_users | test_users)
    all_items = sorted(train_items | test_items)
    
    # Create bidirectional mappings
    user2idx = {user: idx for idx, user in enumerate(all_users)}
    item2idx = {item: idx for idx, item in enumerate(all_items)}
    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}
    
    logger.info(f"Unique users: {len(user2idx):,}")
    logger.info(f"Unique items: {len(item2idx):,}")
    
    return {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'idx2user': idx2user,
        'idx2item': idx2item,
        'n_users': len(user2idx),
        'n_items': len(item2idx)
    }


def prepare_training_data(
    ds: ray.data.Dataset,
    mappings: Dict,
    user_col: str = "network_userid",
    item_col: str = "item_id",
    rating_col: str = "item_quantity"
) -> ray.data.Dataset:
    """
    Prepare dataset for training by mapping IDs to indices.
    
    Args:
        ds: Ray Dataset
        mappings: ID mappings dictionary
        user_col: User column name
        item_col: Item column name
        rating_col: Rating column name
        
    Returns:
        Processed Ray Dataset with user_idx, item_idx, rating columns
    """
    user2idx = mappings['user2idx']
    item2idx = mappings['item2idx']
    
    def map_ids(batch):
        """Map user/item IDs to indices and aggregate."""
        import pandas as pd
        
        df = pd.DataFrame(batch)
        
        # Map IDs to indices
        df['user_idx'] = df[user_col].map(user2idx)
        df['item_idx'] = df[item_col].map(item2idx)
        df['rating'] = df[rating_col].astype(float)
        
        # Aggregate to user-item level (average rating)
        df_agg = df.groupby(['user_idx', 'item_idx'], as_index=False)['rating'].mean()
        
        # Drop rows with unmapped IDs
        df_agg = df_agg.dropna(subset=['user_idx', 'item_idx'])
        
        # Convert to proper types
        df_agg['user_idx'] = df_agg['user_idx'].astype(int)
        df_agg['item_idx'] = df_agg['item_idx'].astype(int)
        
        return df_agg[['user_idx', 'item_idx', 'rating']].to_dict('list')
    
    return ds.map_batches(map_ids, batch_format="pandas", batch_size=None)


# ============================================================================
# Training Function for Ray Train
# ============================================================================

def train_loop(config: Dict):
    """
    Training loop executed by Ray Train workers.
    
    Uses Checkpoint.from_directory() for Ray 2.51.0 compatibility.
    
    Args:
        config: Training configuration dictionary
    """
    from ray.train import get_dataset_shard
    
    # Get data shards
    train_ds = get_dataset_shard("train")
    test_ds = get_dataset_shard("test")
    
    # Device setup
    device = "cuda" if config.get('use_gpu', True) and torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")
    
    # Create model
    model = MatrixFactorization(
        n_users=config['n_users'],
        n_items=config['n_items'],
        n_factors=config['n_factors']
    )
    
    # Prepare model for distributed training
    model = ray_train.torch.prepare_model(model)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['reg_lambda']
    )
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    best_test_rmse = float('inf')
    
    all_epoch_metrics = []
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_count = 0
        
        # Create data loader for this epoch
        train_loader = train_ds.iter_torch_batches(
            batch_size=config['batch_size'],
            dtypes={
                "user_idx": torch.long,
                "item_idx": torch.long,
                "rating": torch.float32
            },
            device=device
        )
        
        for batch in train_loader:
            user_ids = batch["user_idx"]
            item_ids = batch["item_idx"]
            ratings = batch["rating"]
            
            optimizer.zero_grad()
            
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * len(user_ids)
            train_count += len(user_ids)
        
        train_loss /= max(train_count, 1)
        train_rmse = np.sqrt(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_count = 0
        
        test_loader = test_ds.iter_torch_batches(
            batch_size=config['batch_size'],
            dtypes={
                "user_idx": torch.long,
                "item_idx": torch.long,
                "rating": torch.float32
            },
            device=device
        )
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch["user_idx"]
                item_ids = batch["item_idx"]
                ratings = batch["rating"]
                
                predictions = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                test_loss += loss.item() * len(user_ids)
                test_count += len(user_ids)
        
        test_loss /= max(test_count, 1)
        test_rmse = np.sqrt(test_loss)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Track best model
        is_best = test_rmse < best_test_rmse
        if is_best:
            best_test_rmse = test_rmse
        
        # Prepare metrics - ensure all values are JSON serializable (native Python types)
        metrics = {
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "test_loss": float(test_loss),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "learning_rate": float(optimizer.param_groups[0]['lr']),
            "is_best": bool(is_best)  # Convert numpy.bool_ to Python bool
        }

        # Add metrics to all_epoch_metrics for potential later use (e.g., logging, MLflow)
        all_epoch_metrics.append(metrics)
                
        # Save checkpoint using Checkpoint.from_directory() for Ray 2.51.0
        # Create a temporary directory to save checkpoint data
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_data = {
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse)
            }
            
            # Save checkpoint to file
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            torch.save(checkpoint_data, checkpoint_path)
            
            # Create Checkpoint from directory
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            
            # Report metrics to Ray Train
            ray_train.report(metrics, checkpoint=checkpoint)
        
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']}: "
            f"Train RMSE={train_rmse:.4f}, "
            f"Test RMSE={test_rmse:.4f}"
            f"{' (best)' if is_best else ''}"
        )
    
    rank = ray_train.get_context().get_world_rank()
    if rank == 0:
        with open(os.path.join(config['checkpoint_base'], "all_epoch_metrics.json"), "w") as f:
            json.dump(all_epoch_metrics, f, indent=2)


# ============================================================================
# Model Trainer Class
# ============================================================================

class ModelTrainer:
    """
    Orchestrates Matrix Factorization model training.
    
    Uses Ray Train for distributed training and handles:
    - Data loading and preprocessing
    - Model training with checkpointing
    - MLflow integration (optional)
    - Model artifact saving
    """
    
    def __init__(self, config: Dict, monitoring_client=None, app_config=None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            monitoring_client: Optional MonitoringClient for metrics
            app_config: Optional ConfigParser for app-wide settings (e.g. MLflow)
        """
        self.config = config
        self.monitoring = monitoring_client
        self.app_config = app_config
        self.checkpoint_base = Path(config.get('checkpoint_base', '/workspace/model_checkpoints'))
    
    def train(
        self,
        train_dir: str,
        test_dir: str,
        user_col: str = "network_userid",
        item_col: str = "item_id",
        rating_col: str = "item_quantity"
    ) -> Dict:
        """
        Train Matrix Factorization model.
        
        Args:
            train_dir: Path to training data (partitioned Parquet)
            test_dir: Path to test data (partitioned Parquet)
            user_col: User column name
            item_col: Item column name
            rating_col: Rating column name
            
        Returns:
            Training metrics and checkpoint information
        """
        logger.info("="*80)
        logger.info("MATRIX FACTORIZATION TRAINING")
        logger.info("="*80)
        
        # Load data
        logger.info("Loading training data...")
        train_ds_raw = load_partitioned_parquet(train_dir)
        
        logger.info("Loading test data...")
        test_ds_raw = load_partitioned_parquet(test_dir)
        
        # Create ID mappings
        mappings = create_id_mappings(
            train_ds_raw, test_ds_raw,
            user_col=user_col, item_col=item_col
        )
        
        # Prepare training data
        logger.info("Preparing training data...")
        train_ds = prepare_training_data(
            train_ds_raw, mappings,
            user_col=user_col, item_col=item_col, rating_col=rating_col
        )
        train_ds = train_ds.materialize()
        
        logger.info("Preparing test data...")
        test_ds = prepare_training_data(
            test_ds_raw, mappings,
            user_col=user_col, item_col=item_col, rating_col=rating_col
        )
        test_ds = test_ds.materialize()
        
        train_count = train_ds.count()
        test_count = test_ds.count()
        
        logger.info(f"Train interactions: {train_count:,}")
        logger.info(f"Test interactions: {test_count:,}")
        
        # Training configuration
        train_loop_config = {
            'n_users': mappings['n_users'],
            'n_items': mappings['n_items'],
            'n_factors': self.config['n_factors'],
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs'],
            'reg_lambda': self.config['reg_lambda'],
            'use_gpu': self.config.get('use_gpu', True),
            'checkpoint_base': str(self.checkpoint_base),
        }
        
        logger.info(f"Training config: {train_loop_config}")
        
        # Create dated checkpoint directory
        today = datetime.now().strftime("%Y_%m_%d")
        checkpoint_dir = self.checkpoint_base / f"checkpoint_{today}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        
        # Configure Ray Trainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop,
            train_loop_config=train_loop_config,
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=self.config.get('use_gpu', True),
                resources_per_worker={
                    "GPU": 1 if self.config.get('use_gpu', True) else 0,
                    "CPU": 4
                }
            ),
            run_config=RunConfig(
                name="mf_training",
                storage_path=str(checkpoint_dir),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="test_rmse",
                    checkpoint_score_order="min"
                )
            ),
            datasets={"train": train_ds, "test": test_ds}
        )
        
        # Run training
        logger.info("Starting Ray Train...")
        start_time = datetime.now()
        
        result = trainer.fit()
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract final metrics
        metrics = self._extract_metrics(result)
        metrics['training_time_seconds'] = training_time
        
        logger.info("="*80)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        logger.info(f"Final Train RMSE: {metrics.get('train_rmse', 'N/A')}")
        logger.info(f"Final Test RMSE: {metrics.get('test_rmse', 'N/A')}")
        
        # Save model artifacts
        self._save_model_artifacts(
            mappings=mappings,
            checkpoint_dir=checkpoint_dir,
            metrics=metrics,
            result=result,
            config=train_loop_config
        )

        with open(os.path.join(self.checkpoint_base, "all_epoch_metrics.json"), "r") as f:
            all_epoch_metrics = json.load(f)
        
        mlflow_logger = None
        if self.app_config:
            mlflow_logger = MLflowLogger(self.app_config)

        logger.info("="*80)
        logger.info("EPOCH SUMMARY")
        logger.info("="*80)
        for epoch_metrics in all_epoch_metrics:
            logger.info(
                f"Epoch {epoch_metrics['epoch']}: "
                f"Train RMSE={epoch_metrics['train_rmse']:.4f}, "
                f"Test RMSE={epoch_metrics['test_rmse']:.4f}, "
                f"LR={epoch_metrics['learning_rate']:.6f}"
            )
            if mlflow_logger:
                mlflow_logger.log_metrics(epoch_metrics, step=epoch_metrics['epoch'])
        
        metrics['checkpoint_dir'] = str(checkpoint_dir)
        metrics['mappings'] = mappings
        
        return metrics
    
    def _extract_metrics(self, result) -> Dict:
        """Extract metrics from Ray Train result."""
        metrics = {}
        
        if hasattr(result, 'metrics') and result.metrics:
            logger.info("Found metrics in result")
            metrics = dict(result.metrics)
        elif hasattr(result, 'metrics_dataframe') and result.metrics_dataframe is not None:
            logger.info("Found metrics dataframe in result")
            df = result.metrics_dataframe
            if len(df) > 0:
                metrics = df.iloc[-1].to_dict()
        
        if not metrics:
            logger.warning("No metrics found in result")
            metrics = {
                'train_rmse': float('inf'),
                'test_rmse': float('inf'),
                'epoch': 0
            }
        
        return metrics
    
    def _save_model_artifacts(
        self,
        mappings: Dict,
        checkpoint_dir: Path,
        metrics: Dict,
        result,
        config: Dict
    ):
        """Save model artifacts to checkpoint directory."""
        model_dir = checkpoint_dir / "model_artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model artifacts to: {model_dir}")
        
        # Save ID mappings
        with open(model_dir / "mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        logger.info("  ✓ Saved mappings.pkl")
        
        # Save metrics
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'mappings'}
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics_to_save, f, indent=2, default=str)
        logger.info("  ✓ Saved metrics.json")
        
        # Save config
        with open(model_dir / "config.json", "w") as f:
            config_copy = config.copy()
            config_copy.pop('checkpoint_base', None)
            json.dump(config_copy, f, indent=2)
        logger.info("  ✓ Saved config.json")
        
        # Extract and save PyTorch model checkpoint using as_directory()
        if result.checkpoint:
            try:
                # Use as_directory() to extract checkpoint files
                with result.checkpoint.as_directory() as ckpt_dir:

                    src_checkpoint = os.path.join(ckpt_dir, "checkpoint.pt")
                    if os.path.exists(src_checkpoint):
                        shutil.copy(src_checkpoint, model_dir / "model.pt")
                        logger.info("  ✓ Saved model.pt")
                    else:
                        # Try to find any .pt file
                        for f in os.listdir(ckpt_dir):
                            if f.endswith('.pt'):
                                shutil.copy(os.path.join(ckpt_dir, f), model_dir / "model.pt")
                                logger.info(f"  ✓ Saved model.pt (from {f})")
                                break
            except Exception as e:
                logger.warning(f"Failed to save model checkpoint: {e}")
        
        logger.info(f"✓ Model artifacts saved to: {model_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run model training standalone."""
    import argparse
    import configparser
    
    parser = argparse.ArgumentParser(description="Train MF Recommender Model")
    parser.add_argument(
        '-c', '--config_path',
        default="/workspace/src/main/resources/train_params.ini",
        help="Path to configuration file"
    )
    parser.add_argument(
        '--train_dir',
        default=None,
        help="Path to training data"
    )
    parser.add_argument(
        '--test_dir',
        default=None,
        help="Path to test data"
    )
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config_path)
    
    trainer_config = {
        'n_factors': config.getint("model", "n_factors"),
        'learning_rate': config.getfloat("model", "learning_rate"),
        'batch_size': config.getint("model", "batch_size"),
        'epochs': config.getint("model", "epochs"),
        'reg_lambda': config.getfloat("model", "reg_lambda"),
        'use_gpu': config.getboolean("training", "use_gpu"),
        'checkpoint_base': config.get("training", "checkpoint_base")
    }
    
    trainer = ModelTrainer(trainer_config, app_config=config)
    
    train_dir = args.train_dir or os.path.join(config.get("data", "train_base"), "train_data")
    test_dir = args.test_dir or os.path.join(config.get("data", "train_base"), "test_data")
    
    metrics = trainer.train(
        train_dir=train_dir,
        test_dir=test_dir,
        user_col=config.get("model", "user_col"),
        item_col=config.get("model", "item_col"),
        rating_col=config.get("model", "rating_col")
    )
    
    print("="*80)
    print(f"MATRIX FACTORIZATION TRAINING:{metrics.get('epoch', 'N/A')} epochs completed")
    print("="*80)
    
    print(f"\n✓ Training complete")
    print(f"  Test RMSE: {metrics.get('test_rmse', 'N/A')}")
    print(f"  Checkpoint: {metrics.get('checkpoint_dir')}")


if __name__ == "__main__":
    main()
