import os
import datetime
import random
import string
from typing import Dict, Any
from s3_manager import S3Manager, extract_s3_key_from_url
from scripts.inference import run_inference
from dotenv import load_dotenv
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
import shutil

# Comment out the restoration import until it exists
# from restoration import *
from schemas.input import INPUT_SCHEMA

logger = RunPodLogger()
# Load environment variables
load_dotenv()

# Add detailed logging for debugging
logger.info("=== DEBUGGING RUNPOD SERVERLESS ENVIRONMENT ===")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"__file__ path: {__file__}")
logger.info(f"Absolute __file__ path: {os.path.abspath(__file__)}")
logger.info(f"Directory listing of current directory: {os.listdir('.')}")

# Log environment variables (without exposing sensitive values)
logger.info("=== ENVIRONMENT VARIABLES CHECK ===")
logger.info(f"RUNPOD_SECRET_S3_BUCKET_NAME exists: {os.getenv('RUNPOD_SECRET_S3_BUCKET_NAME') is not None}")
logger.info(f"RUNPOD_SECRET_S3_ACCESS_KEY_ID exists: {os.getenv('RUNPOD_SECRET_S3_ACCESS_KEY_ID') is not None}")
logger.info(f"RUNPOD_SECRET_S3_SECRET_ACCESS_KEY exists: {os.getenv('RUNPOD_SECRET_S3_SECRET_ACCESS_KEY') is not None}")
logger.info(f"RUNPOD_SECRET_S3_REGION exists: {os.getenv('RUNPOD_SECRET_S3_REGION') is not None}")
logger.info(f"RUNPOD_SECRET_S3_ENDPOINT_URL exists: {os.getenv('RUNPOD_SECRET_S3_ENDPOINT_URL') is not None}")

def setup_s3():
    """Initialize and return S3Manager instance"""
    return S3Manager(
        bucket_name=os.getenv('RUNPOD_SECRET_S3_BUCKET_NAME'),
        access_key_id=os.getenv('RUNPOD_SECRET_S3_ACCESS_KEY_ID'),
        secret_access_key=os.getenv('RUNPOD_SECRET_S3_SECRET_ACCESS_KEY'),
        region=os.getenv('RUNPOD_SECRET_S3_REGION', 'us-east-1'),
        endpoint_url=os.getenv('RUNPOD_SECRET_S3_ENDPOINT_URL')
    )

def generate_unique_filename(extension: str = "") -> str:
    """Generate unique filename with datetime timestamp and random string"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    
    return f"{timestamp}_{random_string}{extension}"

def handler(event):
    """
    Process the input payload and return the result file URL
    
    Args:
        payload (Dict[str, Any]): Input payload according to input schema
        
    Returns:
        Dict[str, str]: Dictionary containing the result file URL
    """
    # DEBUG: Log the complete event structure received
    logger.info("=== RUNPOD HANDLER DEBUG ===")
    logger.info(f"Complete event received: {event}")
    logger.info(f"Event type: {type(event)}")
    logger.info(f"Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    # Check if event has 'input' field
    if isinstance(event, dict):
        logger.info(f"Event has 'input' field: {'input' in event}")
        logger.info(f"Event has 'id' field: {'id' in event}")
        if 'input' in event:
            logger.info(f"Event['input'] content: {event['input']}")
            logger.info(f"Event['input'] type: {type(event['input'])}")
        else:
            logger.info("Available event fields:")
            for key, value in event.items():
                logger.info(f"  {key}: {type(value)} = {value}")
    
    # Try to extract payload with error handling
    try:
        if 'input' in event:
            payload = validate(event["input"], INPUT_SCHEMA)
            payload = event['input']
            logger.info(f"Successfully extracted payload from event['input']: {payload}")
        else:
            # Maybe the entire event IS the input payload
            logger.info("No 'input' field found, trying to use entire event as payload")
            payload = validate(event, INPUT_SCHEMA)
            logger.info(f"Successfully validated entire event as payload: {payload}")
    except Exception as e:
        logger.error(f"Failed to extract/validate payload: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        raise Exception(f"Failed to get job. | Error Type: {type(e).__name__} | Error Message: {str(e)}")
    #payload=payload['validated_input']
    if not payload['source_video']:
        logger.error(f"Failed to get source video, payload is {payload}, event is {event}")
        raise Exception(f"Failed to get source video, payload is {payload}, event is {event}")
    # Define temp_dir outside try block so it's available in except block
    # Use a directory relative to the current file with unique naming
    current_dir = os.path.dirname(os.path.abspath(__file__))
    unique_session_id = generate_unique_filename()
    temp_dir = os.path.join(current_dir, "tmp", f"lipsync_{unique_session_id}")
    
    # Log path information
    logger.info(f"Current directory resolved to: {current_dir}")
    logger.info(f"Temporary directory path: {temp_dir}")
    
    try:
        # Initialize S3 manager
        s3 = setup_s3()
        
        # Create a unique temporary directory
        logger.info(f"Attempting to create temporary directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Temporary directory created successfully: {os.path.exists(temp_dir)}")
        
        # Extract S3 keys from URLs
        logger.info(f"Extracting S3 keys from URLs")
        video_s3_key = extract_s3_key_from_url(payload['source_video'])
        audio_s3_key = extract_s3_key_from_url(payload['source_audio'])
        
        logger.info(f"Video S3 key: {video_s3_key}")
        logger.info(f"Audio S3 key: {audio_s3_key}")
        
        # Generate unique local filenames
        video_filename = generate_unique_filename("mp4")
        audio_filename = generate_unique_filename("mp3")
        
        # Download files from S3
        video_path = os.path.join(temp_dir, video_filename)
        audio_path = os.path.join(temp_dir, audio_filename)
        
        logger.info(f"Attempting to download video from S3: {video_s3_key}")
        if not s3.download_file(video_s3_key, video_path):
            logger.error(f"Failed to download video file from S3: {video_s3_key}")
            raise Exception("Failed to download video file")
        logger.info(f"Video download successful, file exists: {os.path.exists(video_path)}")
            
        logger.info(f"Attempting to download audio from S3: {audio_s3_key}")
        if not s3.download_file(audio_s3_key, audio_path):
            logger.error(f"Failed to download audio file from S3: {audio_s3_key}")
            raise Exception("Failed to download audio file")
        logger.info(f"Audio download successful, file exists: {os.path.exists(audio_path)}")
        
        # Prepare inference arguments
        unet_config_path = "configs/unet/second_stage.yaml"
        inference_ckpt_path = "checkpoints/latentsync_unet.pt"
        
        # Log file existence checks
        logger.info(f"Checking if config file exists: {os.path.exists(unet_config_path)}")
        logger.info(f"Checking if checkpoint file exists: {os.path.exists(inference_ckpt_path)}")
        
        # Generate unique result filename
        result_filename = generate_unique_filename("mp4")
        result_local_path = os.path.join(temp_dir, result_filename)
        
        inference_args = {
            'unet_config_path': unet_config_path,
            'inference_ckpt_path': inference_ckpt_path,
            'video_path': video_path,
            'audio_path': audio_path,
            'video_out_path': result_local_path,
            'inference_steps': 20,
            'guidance_scale': 1.0,
            'seed': 1247,
            'start_frame': payload.get('start_frame', 0)
        }
        
        # Run inference
        logger.info(f"Starting inference with arguments: {inference_args}")
        try:
            result_path = run_inference({'input': inference_args})
            logger.info(f"Inference completed successfully, result path: {result_path}")
            logger.info(f"Result file exists: {os.path.exists(result_path)}")
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
        
        # Upload result back to S3 with unique filename
        logger.info(f"Uploading result to S3")
        upload_success, result_s3_key = s3.upload_file(result_path, video_s3_key.replace('.mp4', "_result.mp4"))
        
        if not upload_success:
            raise Exception("Failed to upload result file to S3")
        
        logger.info(f"Result uploaded successfully to S3 key: {result_s3_key}")
        
        # Generate result URL
        result_url = s3.get_file_url(result_s3_key, expiration=86400*31)  # month-long expiration
        
        # Clean up temporary directory
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        
        # Return the result URL
        return {
            "result_url": result_url,
            "s3_key": result_s3_key
        }
        
    except Exception as e:
        # Clean up on error
        logger.error(f"Error in handler: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        if os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        raise Exception(f"Error processing request: {str(e)}")
    
if __name__ == '__main__':
    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler,
            "return_aggregate_stream": True
        }
    )