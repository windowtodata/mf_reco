"""
Training Pipeline Orchestrator.

Coordinates the full training workflow:
1. Data Pipeline: Extract and prepare data using RayDP
2. Train/Test Split: Split by date partitions
3. Model Training: Train Matrix Factorization with Ray Train
4. Evaluation: Calculate NDCG and other metrics
5. Model Selection: Compare with previous best model

Usage:
    python ray_submit.py -j train_pipeline.py -s "data"   # Run data pipeline only
    python ray_submit.py -j train_pipeline.py -s "train"  # Run training pipeline
"""

import os
import sys
import shutil
import json
import pickle
import torch
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging
import configparser
import argparse

# Import pipeline components
from data_pipeline import DataPipeline
from train_model import ModelTrainer, MatrixFactorization
from eval_model import ModelEvaluator, save_evaluation_metrics
from monitoring import PipelineStage, create_influxdb_logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline for Matrix Factorization recommender.
    
    Supports two run stages:
    - "data": Run data extraction only
    - "train": Run full training pipeline (assumes data already extracted)
    """
    
    def __init__(
        self, 
        config: configparser.ConfigParser, 
        run_stage: str = "data"
    ):
        """
        Initialize training pipeline.
        
        Args:
            config: ConfigParser with pipeline settings
            run_stage: Pipeline stage to run ("data" or "train")
        """
        self.config = config
        self.run_stage = run_stage
        
        # Initialize monitoring client
        self.monitoring = create_influxdb_logger(config)
        
        # Column names from config
        self.user_col = config.get('model', 'user_col', fallback='network_userid')
        self.item_col = config.get('model', 'item_col', fallback='item_id')
        self.rating_col = config.get('model', 'rating_col', fallback='item_quantity')
    
    def run_data_pipeline(self) -> str:
        """
        Run data extraction pipeline.
        
        Returns:
            Path to output data directory
        """
        logger.info("\n" + "="*80)
        logger.info("[STEP 1/4] DATA PIPELINE")
        logger.info("="*80)
        
        self.monitoring.send_event(PipelineStage.DATA_PIPELINE_START)
        
        try:
            pipeline = DataPipeline(self.config)
            
            data_dir = pipeline.run(
                data_path=self.config.get("data", "data_path"),
                end_date=self.config.get("data", "end_date"),
                num_days=self.config.getint("data", "num_days"),
                output_base=self.config.get("data", "output_base")
            )
            
            self.monitoring.send_event(
                PipelineStage.DATA_PIPELINE_COMPLETE,
                {"output_dir": data_dir}
            )
            
            return data_dir
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            self.monitoring.send_event(
                PipelineStage.DATA_PIPELINE_FAILED,
                {"error": str(e)}
            )
            raise
    
    def split_train_test_partitions(self, data_dir: str) -> Tuple[str, str]:
        """
        Split date partitions into train and test sets.
        
        Uses test_fraction from config to determine split point.
        Latest partitions go to test set.
        
        Args:
            data_dir: Path to data directory with date partitions
            
        Returns:
            Tuple of (train_dir, test_dir) paths
        """
        logger.info("\n" + "="*80)
        logger.info("[STEP 2/4] TRAIN/TEST SPLIT BY DATE PARTITION")
        logger.info("="*80)
        
        data_path = Path(data_dir)
        partition_dirs = sorted([
            d for d in data_path.glob("date_partition=*") 
            if d.is_dir()
        ])
        
        if not partition_dirs:
            raise ValueError(f"No date partitions found in {data_dir}")
        
        logger.info(f"Found {len(partition_dirs)} date partitions")
        logger.info(f"Date range: {partition_dirs[0].name} to {partition_dirs[-1].name}")
        
        # Split by test fraction (chronological split)
        test_fraction = self.config.getfloat("data", "test_fraction")
        split_idx = int(len(partition_dirs) * (1 - test_fraction))
        
        train_partitions = partition_dirs[:split_idx]
        test_partitions = partition_dirs[split_idx:]
        
        logger.info(f"Train partitions: {len(train_partitions)} ({(1-test_fraction)*100:.0f}%)")
        logger.info(f"Test partitions: {len(test_partitions)} ({test_fraction*100:.0f}%)")
        
        # Create output directories
        train_base = self.config.get("data", "train_base")
        train_dir = Path(train_base) / "train_data"
        test_dir = Path(train_base) / "test_data"
        
        # Clear existing directories
        for dir_path in [train_dir, test_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy partitions
        logger.info("Copying train partitions...")
        for partition in train_partitions:
            dest = train_dir / partition.name
            shutil.copytree(partition, dest)
        
        logger.info("Copying test partitions...")
        for partition in test_partitions:
            dest = test_dir / partition.name
            shutil.copytree(partition, dest)
        
        logger.info(f"✓ Train data: {train_dir}/ ({len(train_partitions)} partitions)")
        logger.info(f"✓ Test data: {test_dir}/ ({len(test_partitions)} partitions)")
        
        return str(train_dir), str(test_dir)
    
    def train_model(self, train_dir: str, test_dir: str) -> Dict:
        """
        Train Matrix Factorization model.
        
        Args:
            train_dir: Path to training data
            test_dir: Path to test data
            
        Returns:
            Training metrics dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("[STEP 3/4] MODEL TRAINING")
        logger.info("="*80)
        
        self.monitoring.send_event(PipelineStage.TRAIN_START)
        
        try:
            # Training config
            trainer_config = {
                'n_factors': self.config.getint("model", "n_factors"),
                'learning_rate': self.config.getfloat("model", "learning_rate"),
                'batch_size': self.config.getint("model", "batch_size"),
                'epochs': self.config.getint("model", "epochs"),
                'reg_lambda': self.config.getfloat("model", "reg_lambda"),
                'use_gpu': self.config.getboolean("training", "use_gpu"),
                'checkpoint_base': self.config.get("training", "checkpoint_base")
            }
            
            trainer = ModelTrainer(
                trainer_config, 
                monitoring_client=self.monitoring,
                app_config=self.config
            )
            
            # Train model
            metrics = trainer.train(
                train_dir=train_dir,
                test_dir=test_dir,
                user_col=self.user_col,
                item_col=self.item_col,
                rating_col=self.rating_col
            )
            
            # Log final metrics
            self.monitoring.log_metrics({
                'train_rmse': metrics.get('train_rmse', 0),
                'test_rmse': metrics.get('test_rmse', 0)
            })
            
            self.monitoring.send_event(
                PipelineStage.TRAIN_COMPLETE,
                {"test_rmse": metrics.get('test_rmse', 0)}
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.monitoring.send_event(
                PipelineStage.TRAIN_FAILED,
                {"error": str(e)}
            )
            raise
    
    def evaluate_model(self, test_dir: str, model_dir: str) -> Dict:
        """
        Evaluate trained model on test data.
        
        Args:
            test_dir: Path to test data
            model_dir: Path to model artifacts
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("[STEP 4a] MODEL EVALUATION")
        logger.info("="*80)
        
        self.monitoring.send_event(PipelineStage.EVAL_START)
        
        try:
            k = self.config.getint("evaluation", "ndcg_k", fallback=10)
            
            evaluator = ModelEvaluator(model_dir=model_dir, k=k)
            
            metrics = evaluator.evaluate_on_test_data(
                test_dir=test_dir,
                user_col=self.user_col,
                item_col=self.item_col,
                rating_col=self.rating_col,
                sample_users=500  # Sample for faster evaluation
            )
            
            # Log evaluation metrics
            self.monitoring.log_metrics(metrics)
            
            self.monitoring.send_event(
                PipelineStage.EVAL_COMPLETE,
                metrics
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.monitoring.send_event(
                PipelineStage.EVAL_FAILED,
                {"error": str(e)}
            )
            raise
    
    def select_best_model(
        self, 
        new_checkpoint_dir: str, 
        new_rmse: float,
        test_dir: str
    ) -> str:
        """
        Compare new model with previous best and save the better one.
        
        Args:
            new_checkpoint_dir: Path to new model checkpoint
            new_rmse: Test RMSE of new model
            test_dir: Path to test data (for potential re-evaluation)
            
        Returns:
            Path to best model directory
        """
        logger.info("\n" + "="*80)
        logger.info("[STEP 4b] MODEL SELECTION")
        logger.info("="*80)
        
        latest_model_path = self.config.get("training", "latest_model_dir")
        latest_dir = Path(latest_model_path)
        
        new_model_artifacts = Path(new_checkpoint_dir) / "model_artifacts"
        
        # Check if previous model exists
        if latest_dir.exists() and (latest_dir / "metrics.json").exists():
            with open(latest_dir / "metrics.json", 'r') as f:
                prev_metrics = json.load(f)
            prev_rmse = prev_metrics.get('test_rmse', float('inf'))
            
            logger.info(f"Previous best RMSE: {prev_rmse:.4f}")
            logger.info(f"New model RMSE: {new_rmse:.4f}")
            
            if new_rmse < prev_rmse:
                logger.info("✓ New model is better! Updating latest model")
                if latest_dir.exists():
                    shutil.rmtree(latest_dir)
                shutil.copytree(new_model_artifacts, latest_dir)
                logger.info(f"   Copied to: {latest_dir}")
                
                if self.config.getboolean("mlflow", "enabled", fallback=False):
                    self._log_best_model_to_mlflow(latest_dir)
            else:
                logger.info("✗ Previous model is better, keeping it")
        else:
            logger.info("✓ No previous model found, saving new model as best")
            latest_dir.parent.mkdir(parents=True, exist_ok=True)
            if latest_dir.exists():
                shutil.rmtree(latest_dir)
            shutil.copytree(new_model_artifacts, latest_dir)
            logger.info(f"   Saved to: {latest_dir}")
            
            if self.config.getboolean("mlflow", "enabled", fallback=False):
                self._log_best_model_to_mlflow(latest_dir)
        
        return str(latest_dir)
    
    def _log_best_model_to_mlflow(self, model_dir: Path):
        """Log the best model to MLflow with registration."""
        try:
            logger.info("Logging best model to MLflow...")
            
            # MLflow config
            tracking_uri = self.config.get("mlflow", "tracking_uri")
            experiment_name = self.config.get("mlflow", "experiment_name")
            registered_name = self.config.get("training", "registered_model_name")
            
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            
            # Load metadata
            with open(model_dir / "config.json", "r") as f:
                model_config = json.load(f)
            
            with open(model_dir / "mappings.pkl", "rb") as f:
                mappings = pickle.load(f)
                
            with open(model_dir / "metrics.json", "r") as f:
                metrics = json.load(f)
            
            # Reconstruct model
            model = MatrixFactorization(
                n_users=mappings['n_users'],
                n_items=mappings['n_items'],
                n_factors=model_config['n_factors']
            )
            
            # Load weights
            checkpoint_path = model_dir / "model.pt"
            # weights_only=False needed for numpy scalars in checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Start MLflow run
            run_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                # Log params and metrics
                mlflow.log_params(model_config)
                mlflow.log_metrics(metrics)
                
                # Log artifacts
                mlflow.log_artifact(str(model_dir / "mappings.pkl"), artifact_path="artifacts")
                mlflow.log_artifact(str(model_dir / "config.json"), artifact_path="artifacts")
                
                # Log model and register
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=registered_name
                )
                
                logger.info(f"✓ Model logged to MLflow (Run ID: {run.info.run_id})")
                logger.info(f"✓ Registered as: {registered_name}")
                
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {e}")

    def run(self):
        """
        Run the pipeline based on configured stage.
        
        Stages:
        - "data": Run data extraction only
        - "train": Run full training pipeline
        """
        start_time = datetime.now()
        logger.info(f"\nPipeline start time: {start_time}")
        logger.info(f"Run stage: {self.run_stage}")
        logger.info("="*80)
        
        try:
            if self.run_stage == "data":
                # Data extraction only
                data_dir = self.run_data_pipeline()
                logger.info(f"\n✓ Data extraction complete: {data_dir}")
                
            elif self.run_stage == "train":
                # Full training pipeline
                data_dir = self.config.get("data", "output_base")
                
                # Step 1: Split train/test
                train_dir, test_dir = self.split_train_test_partitions(data_dir)
                logger.info(f"\n✓ Train/test split complete: {train_dir}, {test_dir}") 
                
                # Step 2: Train model
                train_metrics = self.train_model(train_dir, test_dir)
                logger.info("\n✓ Model training complete")
                
                # Step 3: Evaluate model
                checkpoint_dir = train_metrics.get('checkpoint_dir')
                model_dir = Path(checkpoint_dir) / "model_artifacts"
                
                if model_dir.exists():
                    eval_metrics = self.evaluate_model(test_dir, str(model_dir))
                    
                    # Save evaluation metrics
                    save_evaluation_metrics(
                        metrics=eval_metrics,
                        rmse=train_metrics.get('test_rmse', 0),
                        metrics_dir=self.config.get("training", "metrics_dir"),
                        end_date=self.config.get("data", "end_date")
                    )
                
                # Step 4: Select best model
                best_model_dir = self.select_best_model(
                    new_checkpoint_dir=checkpoint_dir,
                    new_rmse=train_metrics.get('test_rmse', float('inf')),
                    test_dir=test_dir
                )
                
                # Summary
                duration = (datetime.now() - start_time).total_seconds()
                
                logger.info("\n" + "="*80)
                logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
                logger.info("="*80)
                logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
                logger.info(f"\nTraining Metrics:")
                logger.info(f"  Train RMSE: {train_metrics.get('train_rmse', 0):.4f}")
                logger.info(f"  Test RMSE: {train_metrics.get('test_rmse', 0):.4f}")
                
                if 'eval_metrics' in locals():
                    k = self.config.getint("evaluation", "ndcg_k", fallback=10)
                    logger.info(f"\nEvaluation Metrics:")
                    logger.info(f"  NDCG@{k}: {eval_metrics.get(f'ndcg@{k}', 0):.4f}")
                    logger.info(f"  Precision@{k}: {eval_metrics.get(f'precision@{k}', 0):.4f}")
                
                logger.info(f"\nArtifacts:")
                logger.info(f"  Checkpoint: {checkpoint_dir}")
                logger.info(f"  Latest model: {best_model_dir}")
                logger.info("="*80 + "\n")
                
            else:
                raise ValueError(f"Unknown run stage: {self.run_stage}")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
        finally:
            # Clean up
            self.monitoring.close()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Training Pipeline for MF Recommender"
    )
    parser.add_argument(
        '-c', '--config_path',
        default="/workspace/src/main/resources/train_params.ini",
        help="Path to configuration file"
    )
    parser.add_argument(
        '-s', '--run_stage',
        choices=['data', 'train'],
        default="data",
        help="Pipeline stage to run: 'data' for extraction, 'train' for full training"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(args.config_path)
    
    # Run pipeline
    pipeline = TrainingPipeline(config=config, run_stage=args.run_stage)
    pipeline.run()


if __name__ == "__main__":
    main()
