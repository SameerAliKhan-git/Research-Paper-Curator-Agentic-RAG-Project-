import os
import sys
# Fix sys.path for importing src package
sys.path.append(os.getcwd())

import asyncio
import logging
from redis.asyncio import Redis
from src.config import get_settings
from src.services.auth.api_key_service import APIKeyService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    settings = get_settings()
    
    # Connect to local Redis (mapped to port 6379)
    redis_client = Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )
    
    service = APIKeyService(redis_client)
    
    # Register our developer test API key
    raw_key = "dev-test-key-999"
    user_id = "dev_user"
    
    logger.info(f"Registering API key: {raw_key} for user: {user_id}")
    
    result = await service.create_key(
        raw_key=raw_key,
        user_id=user_id,
        tier="admin",
        rate_limit=1000,
        daily_quota=10000,
        tenants=["default", "tenant-1"]
    )
    
    logger.info(f"Successfully registered key. Key hash: {result['key_hash']}")
    
    # Verify we can validate it
    validated = await service.validate_key(raw_key)
    if validated:
        logger.info(f"Verification successful: Validated key for user: {validated.user_id}, tier: {validated.tier}, tenants: {validated.tenants}")
    else:
        logger.error("Verification failed! Key not found in Redis.")
        
    await redis_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
