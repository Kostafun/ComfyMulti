import os
import logging
import datetime
import random
import string
from typing import Optional
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('s3_manager')

class S3Manager:
    def __init__(self, bucket_name: str, access_key_id: str, secret_access_key: str, region: str = 'us-east-1', endpoint_url: str = None):
        """
        Initialize S3Manager with AWS S3 credentials
        
        Args:
            bucket_name (str): Name of the S3 bucket
            access_key_id (str): AWS Access Key ID
            secret_access_key (str): AWS Secret Access Key
            region (str): AWS region (default: us-east-1)
            endpoint_url (str, optional): Custom S3 endpoint URL for S3-compatible services
        """
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint_url = endpoint_url
        
        # Setup S3 client
        try:
            session_config = {
                'aws_access_key_id': access_key_id,
                'aws_secret_access_key': secret_access_key,
                'region_name': region
            }
            
            if endpoint_url:
                session_config['endpoint_url'] = endpoint_url
            
            self.s3_client = boto3.client('s3', **session_config)
            
            # Test connection by checking if bucket exists
            # logger.info(f"Testing connection to S3 bucket: {bucket_name}")
            # self.s3_client.head_bucket(Bucket=bucket_name)
            # logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket not found: {bucket_name}")
            else:
                logger.error(f"S3 connection failed: {e}")
            raise

    def generate_unique_filename(self, original_filename: str, extension: str = None) -> str:
        """
        Generate a unique filename using datetime timestamp and random string
        
        Args:
            original_filename (str): Original filename for reference
            extension (str, optional): File extension to use
            
        Returns:
            str: Unique filename with timestamp and random string
        """
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate random string (8 characters)
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Extract extension from original filename if not provided
        if not extension:
            _, extension = os.path.splitext(original_filename)
        
        # Ensure extension starts with dot
        if extension and not extension.startswith('.'):
            extension = '.' + extension
        
        # Create unique filename
        unique_filename = f"{timestamp}_{random_string}{extension}"
        logger.info(f"Generated unique filename: {unique_filename}")
        
        return unique_filename

    def upload_file(self, local_path: str, s3_key: str = None) -> tuple[bool, str]:
        """
        Upload a file to S3 bucket
        
        Args:
            local_path (str): Local path of the file to upload
            s3_key (str, optional): S3 key (path) for the file. If None, generates unique filename
            
        Returns:
            tuple[bool, str]: (Success status, S3 key used)
        """
        logger.info(f"Uploading file from {local_path} to S3")
        
        # Check if local file exists
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            return False, ""
            
        try:
            # Generate unique S3 key if not provided
            if not s3_key:
                filename = os.path.basename(local_path)
                s3_key = self.generate_unique_filename(filename)
            
            file_size = os.path.getsize(local_path)
            logger.info(f"File size: {file_size} bytes")
            
            # Upload file to S3
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"File upload successful: {s3_key}")
            return True, s3_key
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            return False, ""

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a file from S3 bucket
        
        Args:
            s3_key (str): S3 key (path) of the file to download
            local_path (str): Local path where to save the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Downloading file from S3 key {s3_key} to {local_path}")
        
        try:
            # Create directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir:
                logger.info(f"Creating local directory: {local_dir}")
                os.makedirs(local_dir, exist_ok=True)
            
            # Download the file
            logger.info(f"Starting file download from S3")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Verify download
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                logger.info(f"File download successful. Size: {file_size} bytes")
                return True
            else:
                logger.error(f"File download failed. File does not exist at {local_path}")
                return False
                
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {e}")
            return False

    def get_file_url(self, s3_key: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for accessing a file in S3
        
        Args:
            s3_key (str): S3 key (path) of the file
            expiration (int): URL expiration time in seconds (default: 1 hour)
            
        Returns:
            str: Presigned URL for the file
        """
        logger.info(f"Generating presigned URL for S3 key: {s3_key}")
        
        try:
            # Generate presigned URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            logger.info(f"Generated presigned URL successfully")
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            
            # Fallback to public URL format if presigned URL fails
            if self.endpoint_url:
                url = f"{self.endpoint_url}/{self.bucket_name}/{s3_key}"
            else:
                url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            logger.info(f"Using fallback public URL format")
            return url

    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3 bucket
        
        Args:
            s3_key (str): S3 key (path) of the file to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Deleting file from S3: {s3_key}")
        
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"File deleted successfully: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """
        List files in S3 bucket with optional prefix filter
        
        Args:
            prefix (str): Prefix to filter files (optional)
            
        Returns:
            list: List of S3 keys matching the prefix
        """
        logger.info(f"Listing files in S3 bucket with prefix: {prefix}")
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
            
            logger.info(f"Found {len(files)} files")
            return files
            
        except ClientError as e:
            logger.error(f"Error listing files in S3: {e}")
            return []

def extract_s3_key_from_url(s3_url: str) -> str:
    """
    Extract the S3 key (object path) from an S3 URL
    
    Args:
        s3_url (str): S3 URL in various formats
        
    Returns:
        str: S3 key (object path)
        
    Raises:
        ValueError: If URL format is invalid
    """
    logger.info(f"Extracting S3 key from URL")
    
    try:
        parsed_url = urlparse(s3_url)
        
        # Handle different S3 URL formats
        if parsed_url.netloc.endswith('.amazonaws.com'):
            # Format: https://bucket-name.s3.region.amazonaws.com/path/to/file
            # or: https://s3.region.amazonaws.com/bucket-name/path/to/file
            if parsed_url.netloc.startswith('s3.'):
                # Path-style URL: https://s3.region.amazonaws.com/bucket-name/path/to/file
                path_parts = parsed_url.path.lstrip('/').split('/', 1)
                if len(path_parts) < 2:
                    raise ValueError(f"Invalid S3 path-style URL format: {s3_url}")
                return path_parts[1]  # Return everything after bucket name
            else:
                # Virtual-hosted-style URL: https://bucket-name.s3.region.amazonaws.com/path/to/file
                return parsed_url.path.lstrip('/')
        else:
            # Custom endpoint or other S3-compatible service
            # Assume format: https://endpoint/bucket-name/path/to/file
            path_parts = parsed_url.path.lstrip('/').split('/', 1)
            if len(path_parts) < 2:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            return path_parts[1]  # Return everything after bucket name
            
    except Exception as e:
        logger.error(f"Error parsing S3 URL: {e}")
        raise ValueError(f"Invalid S3 URL format: {s3_url}")