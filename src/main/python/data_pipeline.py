"""
Data Pipeline using RayDP (PySpark on Ray)
Extracts and prepares data for Matrix Factorization training.
Saves partitioned data as Parquet for efficient processing.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import configparser

import ray
from raydp import init_spark, stop_spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Data extraction pipeline using RayDP.
    
    Reads transaction data, filters by date range, calculates average
    item quantity per order as the rating, and saves as partitioned Parquet.
    """
    
    def __init__(self, config: configparser.ConfigParser):
        """
        Initialize data pipeline.
        
        Args:
            config: ConfigParser with pipeline settings
        """
        self.config = config
        self.spark: Optional[SparkSession] = None
        
        # Load Spark configuration
        self.spark_config = {
            'app_name': config.get('spark', 'app_name', fallback='MFDataPipeline'),
            'num_executors': config.getint('spark', 'num_executors', fallback=2),
            'executor_cores': config.getint('spark', 'executor_cores', fallback=2),
            'executor_memory': config.get('spark', 'executor_memory', fallback='2g'),
            'shuffle_partitions': config.getint('spark', 'shuffle_partitions', fallback=8)
        }
        
        # Column names from config
        self.user_col = config.get('model', 'user_col', fallback='network_userid')
        self.item_col = config.get('model', 'item_col', fallback='item_id')
        self.rating_col = config.get('model', 'rating_col', fallback='item_quantity')
        self.order_col = config.get('model', 'order_col', fallback='order_id')
        self.date_col = config.get('model', 'date_col', fallback='datestamp')
    
    def _init_spark(self):
        """Initialize Spark on Ray cluster."""
        logger.info("Initializing Spark on Ray...")
        logger.info(f"Spark config: {self.spark_config}")
        
        self.spark = init_spark(
            app_name=self.spark_config['app_name'],
            num_executors=self.spark_config['num_executors'],
            executor_cores=self.spark_config['executor_cores'],
            executor_memory=self.spark_config['executor_memory'],
            configs={
                "spark.sql.shuffle.partitions": str(self.spark_config['shuffle_partitions']),
                "spark.sql.adaptive.enabled": "true",
                "spark.sql.adaptive.coalescePartitions.enabled": "true"
            }
        )
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("✓ Spark initialized successfully")
    
    def _stop_spark(self):
        """Stop Spark cluster."""
        if self.spark:
            logger.info("Shutting down Spark...")
            stop_spark()
            self.spark = None
            logger.info("✓ Spark stopped")
    
    def run(
        self,
        data_path: str,
        end_date: str,
        num_days: int,
        output_base: str = "model_data"
    ) -> str:
        """
        Run data extraction pipeline.
        
        Args:
            data_path: Path to source CSV data
            end_date: End date for data extraction (YYYY-MM-DD)
            num_days: Number of days to include
            output_base: Base directory for output
            
        Returns:
            Path to output directory
        """
        logger.info("="*80)
        logger.info("DATA PIPELINE - RayDP → Parquet")
        logger.info("="*80)
        
        try:
            self._init_spark()
            
            # Calculate date range
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=num_days - 1)
            
            logger.info(f"Date range: {start_dt.date()} to {end_dt.date()} ({num_days} days)")
            
            # Load data
            logger.info(f"Loading data from: {data_path}")
            df = self.spark.read.csv(data_path, header=True, inferSchema=True)
            
            initial_count = df.count()
            logger.info(f"Initial records: {initial_count:,}")
            
            # Show schema
            logger.info("Schema:")
            df.printSchema()
            
            # Parse and filter by date
            logger.info("Filtering by date range...")
            df = df.withColumn(
                self.date_col, 
                F.to_date(F.col(self.date_col))
            )
            
            df_filtered = df.filter(
                (F.col(self.date_col) >= start_dt.date()) &
                (F.col(self.date_col) <= end_dt.date())
            )
            
            # Add date partition column for efficient storage
            df_filtered = df_filtered.withColumn(
                "date_partition",
                F.date_format(F.col(self.date_col), "yyyy_MM_dd")
            )
            
            # Ensure rating column is numeric
            df_filtered = df_filtered.withColumn(
                self.rating_col,
                F.col(self.rating_col).cast(FloatType())
            )
            
            # Remove nulls in key columns
            df_filtered = df_filtered.filter(
                F.col(self.user_col).isNotNull() &
                F.col(self.item_col).isNotNull() &
                F.col(self.rating_col).isNotNull()
            )
            
            filtered_count = df_filtered.count()
            pct = (filtered_count / initial_count * 100) if initial_count > 0 else 0
            logger.info(f"Filtered records: {filtered_count:,} ({pct:.1f}%)")
            
            # Calculate average quantity per order per item (for MF rating)
            # This aggregates at (user, order, item) level first
            logger.info("Calculating average item quantities...")
            
            df_aggregated = df_filtered.groupBy(
                self.user_col, 
                self.order_col, 
                self.item_col,
                "date_partition"
            ).agg(
                F.mean(self.rating_col).alias("avg_quantity"),
                F.sum(self.rating_col).alias("total_quantity"),
                F.count("*").alias("transaction_count")
            )
            
            agg_count = df_aggregated.count()
            logger.info(f"Aggregated records: {agg_count:,}")
            
            # Save as Parquet partitioned by date
            logger.info(f"Saving as Parquet to: {output_base}")
            
            # Select only necessary columns for training
            df_final = df_aggregated.select(
                F.col(self.user_col),
                F.col(self.order_col),
                F.col(self.item_col),
                F.col("avg_quantity").alias(self.rating_col),
                F.col("date_partition")
            )
            
            # Write partitioned Parquet
            df_final.write \
                .mode("overwrite") \
                .partitionBy("date_partition") \
                .parquet(output_base)
            
            # Log partition details
            output_path = Path(output_base)
            partition_dirs = sorted(output_path.glob("date_partition=*"))
            logger.info(f"Created {len(partition_dirs)} date partitions")
            
            total_size = 0
            for partition_dir in partition_dirs[:5]:  # Show first 5
                parquet_files = list(partition_dir.glob("*.parquet"))
                partition_size = sum(f.stat().st_size for f in parquet_files)
                total_size += partition_size
                logger.info(
                    f"  {partition_dir.name}: "
                    f"{len(parquet_files)} files, "
                    f"{partition_size / 1024 / 1024:.2f} MB"
                )
            
            if len(partition_dirs) > 5:
                logger.info(f"  ... and {len(partition_dirs) - 5} more partitions")
            
            logger.info("="*80)
            logger.info("✓ DATA PIPELINE COMPLETE")
            logger.info("="*80)
            logger.info(f"Output: {output_base}/ (Parquet format)")
            logger.info(f"Total records: {agg_count:,}")
            logger.info(f"Date partitions: {len(partition_dirs)}")
            
            return output_base
            
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        finally:
            self._stop_spark()


def main():
    """Run data pipeline standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Pipeline for MF Recommender")
    parser.add_argument(
        '-c', '--config_path',
        default="/workspace/src/main/resources/train_params.ini",
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config_path)
    
    pipeline = DataPipeline(config)
    
    output_dir = pipeline.run(
        data_path=config.get("data", "data_path"),
        end_date=config.get("data", "end_date"),
        num_days=config.getint("data", "num_days"),
        output_base=config.get("data", "output_base")
    )
    
    print(f"\n✓ Data saved to: {output_dir}")


if __name__ == "__main__":
    main()
