"""
Ray Job Submission Script.

Submits pipeline jobs to Ray cluster for execution.
Supports both data extraction and training stages.

Usage:
    python ray_submit.py -j train_pipeline.py -s "data"   # Run data pipeline
    python ray_submit.py -j train_pipeline.py -s "train"  # Run training
"""

import logging
import time
import configparser
import argparse
from ray.job_submission import JobSubmissionClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_config(config_path: str) -> configparser.ConfigParser:
    """Read configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def submit_job(
    client: JobSubmissionClient,
    job_file: str,
    run_stage: str,
    config_path: str
) -> str:
    """
    Submit job to Ray cluster.
    
    Args:
        client: Ray job submission client
        job_file: Python file to execute
        run_stage: Pipeline stage ("data" or "train")
        config_path: Path to config file
        
    Returns:
        Job ID
    """
    # Runtime environment
    runtime_env = {
        "working_dir": ".",
        "env_vars": {
            "PYTHONPATH": "/workspace/src/main/resources:/workspace/src/main/python"
        }
    }
    
    # Entrypoint command
    entrypoint = f"python {job_file} --run_stage {run_stage} --config_path {config_path}"
    
    logger.info("Submitting job...")
    logger.info(f"  Entrypoint: {entrypoint}")
    logger.info(f"  Working dir: {runtime_env['working_dir']}")
    
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
    )
    
    return job_id


def wait_for_job(client: JobSubmissionClient, job_id: str, poll_interval: int = 5):
    """
    Wait for job to complete and stream logs.
    
    Args:
        client: Ray job submission client
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks
    """
    logger.info(f"Job submitted with ID: {job_id}")
    logger.info("")
    logger.info("NOTE: First run may take 2-3 minutes (installing packages)")
    logger.info("      Subsequent runs will be faster (cached)")
    logger.info("")
    logger.info("Waiting for job to complete...")
    logger.info("="*80)
    
    last_status = None
    last_log_position = 0
    
    while True:
        status = client.get_job_status(job_id)
        
        # Log status changes
        if status != last_status:
            logger.info(f"Status: {status}")
            last_status = status
        
        # Stream logs (if supported)
        try:
            logs = client.get_job_logs(job_id)
            if logs and len(logs) > last_log_position:
                new_logs = logs[last_log_position:]
                print(new_logs, end='', flush=True)
                last_log_position = len(logs)
        except:
            pass
        
        # Check if terminal
        if status.is_terminal():
            logger.info("="*80)
            logger.info(f"Job finished with status: {status}")
            logger.info("="*80)
            
            # Get final logs
            try:
                logs = client.get_job_logs(job_id)
                if logs:
                    print("\n--- FINAL OUTPUT ---")
                    print(logs)
                    print("--- END OUTPUT ---\n")
            except Exception as e:
                logger.warning(f"Could not retrieve final logs: {e}")
            
            if str(status) == "SUCCEEDED":
                logger.info("✓ Job completed successfully!")
            else:
                logger.error(f"✗ Job failed with status: {status}")
            
            break
        
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(
        description="Submit jobs to Ray cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run data extraction
    python ray_submit.py -j train_pipeline.py -s data
    
    # Run training pipeline
    python ray_submit.py -j train_pipeline.py -s train
    
    # Use custom config
    python ray_submit.py -j train_pipeline.py -s train -c /path/to/config.ini
        """
    )
    
    parser.add_argument(
        '-j', '--job_file',
        required=True,
        help='Python file to execute (e.g., train_pipeline.py)'
    )
    
    parser.add_argument(
        '-s', '--run_stage',
        choices=['data', 'train'],
        default='data',
        help='Pipeline stage: "data" for extraction, "train" for training'
    )
    
    parser.add_argument(
        '-c', '--config_path',
        default='/workspace/src/main/resources/train_params.ini',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Submit job and exit without waiting'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RAY JOB SUBMISSION")
    logger.info("="*80)
    
    # Read config
    config = read_config(args.config_path)
    
    # Connect to Ray cluster
    ray_dashboard_url = config.get('ray', 'ray_cluster', fallback='http://localhost:8265')
    logger.info(f"Connecting to Ray cluster at: {ray_dashboard_url}")
    
    try:
        client = JobSubmissionClient(ray_dashboard_url)
        
        # Submit job
        job_id = submit_job(
            client=client,
            job_file=args.job_file,
            run_stage=args.run_stage,
            config_path=args.config_path
        )
        
        logger.info(f"✓ Job submitted: {job_id}")
        
        # Wait for completion (unless --no-wait)
        if not args.no_wait:
            wait_for_job(client, job_id)
        else:
            logger.info("Job submitted. Use --no-wait to skip waiting.")
            logger.info(f"Monitor at: {ray_dashboard_url}")
            
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
