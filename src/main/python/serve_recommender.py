"""
Ray Serve Deployment for Complementary Product Recommender.

Usage:
    serve run serve_recommender:deployment
"""

import os
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import ray
from ray import serve

from recommend import ComplementaryProductRecommender

# Configure logging
logger = logging.getLogger("ray.serve")

app = FastAPI()

class BasketRequest(BaseModel):
    basket: List[str]
    top_k: Optional[int] = 5

@serve.deployment(
    name="recommender",
    # Share GPU among replicas (e.g., 5 replicas on 1 GPU)
    ray_actor_options={"num_gpus": 0.25},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_num_ongoing_requests_per_replica": 1000,
    }
)
@serve.ingress(app)
class RecommenderDeployment:
    def __init__(self, model_dir: str):
        """
        Initialize the recommender.
        
        This loads the model and LSH index into memory (RAM).
        They remain cached for the lifetime of this replica, ensuring fast response times.
        """
        logger.info(f"Initializing RecommenderDeployment with model_dir: {model_dir}")
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory {model_dir} does not exist! Recommendations may fail.")
        
        self.recommender = ComplementaryProductRecommender(model_dir)
        logger.info("Recommender initialized successfully")

    @app.post("/recommend")
    def recommend(self, request: BasketRequest) -> Dict[str, Any]:
        try:
            recommendations = self.recommender.recommend(
                request.basket, 
                top_k=request.top_k
            )
            return {
                "basket": request.basket,
                "recommendations": [
                    {"item_id": item, "score": score} 
                    for item, score in recommendations
                ]
            }
        except Exception as e:
            logger.error(f"Error during recommendation: {e}")
            return {"error": str(e)}

    @app.get("/health")
    def health(self):
        return {"status": "healthy"}

# Entrypoint
model_dir = os.environ.get("MODEL_DIR", "/workspace/models/latest")
deployment = RecommenderDeployment.bind(model_dir=model_dir)