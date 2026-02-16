"""
Monitoring utilities for InfluxDB and MLflow integration.
Provides separate APIs for InfluxDB and MLflow.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from enum import Enum
import mlflow

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stage identifiers for monitoring."""
    DATA_PIPELINE_START = "data_pipeline_start"
    DATA_PIPELINE_COMPLETE = "data_pipeline_complete"
    DATA_PIPELINE_FAILED = "data_pipeline_failed"
    TRAIN_START = "train_start"
    TRAIN_COMPLETE = "train_complete"
    TRAIN_FAILED = "train_failed"
    EVAL_START = "eval_start"
    EVAL_COMPLETE = "eval_complete"
    EVAL_FAILED = "eval_failed"


class InfluxDBLogger:
    """InfluxDB monitoring client."""

    def __init__(self, config):
        self.config = config
        self.influxdb_enabled = config.getboolean("influxdb", "enabled", fallback=False)
        self._influx_client = None
        self._influx_write_api = None
        self._setup_influxdb()

    def _setup_influxdb(self):
        """Setup InfluxDB client if enabled."""
        if not self.influxdb_enabled:
            logger.info("InfluxDB monitoring disabled")
            return

        try:
            from influxdb_client import InfluxDBClient
            from influxdb_client.client.write_api import SYNCHRONOUS

            self._influx_client = InfluxDBClient(
                url=self.config.get("influxdb", "url"),
                token=self.config.get("influxdb", "token"),
                org=self.config.get("influxdb", "org")
            )
            self._influx_write_api = self._influx_client.write_api(write_options=SYNCHRONOUS)
            self._influx_bucket = self.config.get("influxdb", "bucket")
            self._influx_org = self.config.get("influxdb", "org")
            logger.info("InfluxDB client initialized")
        except ImportError:
            logger.warning("influxdb-client not installed, disabling InfluxDB")
            self.influxdb_enabled = False
        except Exception as e:
            logger.warning(f"InfluxDB setup failed: {e}")
            self.influxdb_enabled = False

    def send_event(self, stage: PipelineStage, metadata: Optional[Dict] = None):
        """Send pipeline event to InfluxDB."""
        logger.info(f"Event: {stage.value}")
        if not self.influxdb_enabled:
            return
        try:
            from influxdb_client import Point

            point = Point("pipeline_events") \
                .tag("stage", stage.value) \
                .field("status", 1) \
                .time(datetime.now())

            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (int, float)):
                        point = point.field(key, value)
                    else:
                        point = point.tag(key, str(value))

            self._influx_write_api.write(
                bucket=self._influx_bucket,
                org=self._influx_org,
                record=point
            )
        except Exception as e:
            logger.warning(f"Failed to send event to InfluxDB: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to InfluxDB."""
        if not self.influxdb_enabled:
            return
        try:
            from influxdb_client import Point

            point = Point("training_metrics")
            if step is not None:
                point = point.field("step", step)
            for name, value in metrics.items():
                point = point.field(name, float(value))

            self._influx_write_api.write(
                bucket=self._influx_bucket,
                org=self._influx_org,
                record=point
            )
        except Exception as e:
            logger.warning(f"Failed to send metrics to InfluxDB: {e}")

    def close(self):
        """Close InfluxDB client."""
        if self._influx_client:
            try:
                self._influx_client.close()
            except:
                pass


class MLflowLogger:
    """MLflow monitoring client."""

    def __init__(self, config):
        self.config = config
        self.mlflow_enabled = config.getboolean("mlflow", "enabled", fallback=False)
        self._mlflow = None
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow client if enabled."""
        if not self.mlflow_enabled:
            logger.info("MLflow monitoring disabled")
            return
        try:
            mlflow.set_tracking_uri(self.config.get("mlflow", "tracking_uri"))
            mlflow.set_experiment(self.config.get("mlflow", "experiment_name"))
            self._mlflow = mlflow
            logger.info("MLflow client initialized")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.mlflow_enabled = False

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if not self.mlflow_enabled:
            return
        try:
            for name, value in metrics.items():
                self._mlflow.log_metric(name, float(value), step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
