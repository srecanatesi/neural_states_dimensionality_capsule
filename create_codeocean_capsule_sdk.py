#!/usr/bin/env python3
"""
Script to create a Code Ocean capsule from a GitHub repository using the Code Ocean Python SDK.

This script requires Python 3.9+ and the codeocean package.

Usage:
    python create_codeocean_capsule_sdk.py

Environment variables required:
    CODEOCEAN_API_TOKEN - Your Code Ocean API token
    CODEOCEAN_API_URL - Code Ocean API base URL (optional, defaults to https://api.codeocean.com)

Installation:
    pip install codeocean
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Check Python version
if sys.version_info < (3, 9):
    print("ERROR: This script requires Python 3.9 or higher", file=sys.stderr)
    print(f"Current Python version: {sys.version}", file=sys.stderr)
    sys.exit(1)

# Try to import Code Ocean SDK
try:
    import codeocean
    from codeocean import CodeOceanClient
except ImportError:
    print("ERROR: Code Ocean SDK not installed", file=sys.stderr)
    print("Please install it with: pip install codeocean", file=sys.stderr)
    sys.exit(1)

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
GITHUB_REPO_URL = "https://github.com/srecanatesi/pipeline-capsule"
CAPSULE_NAME = "neural_states_dimensionality_capsule"
CAPSULE_DESCRIPTION = "Code Ocean capsule for processing single session: download NWB, preprocess, and run HMM cross-validation"

def get_api_token() -> Optional[str]:
    """Get Code Ocean API token from environment variable."""
    token = os.environ.get('CODEOCEAN_API_TOKEN')
    if not token:
        print("ERROR: CODEOCEAN_API_TOKEN environment variable not set", file=sys.stderr)
        print("Please set it with: export CODEOCEAN_API_TOKEN=your_token_here", file=sys.stderr)
        print("\nTo get your API token:", file=sys.stderr)
        print("1. Log in to Code Ocean", file=sys.stderr)
        print("2. Go to Account Settings > API Tokens", file=sys.stderr)
        print("3. Generate a new token", file=sys.stderr)
    return token

def get_api_url() -> str:
    """Get Code Ocean API base URL."""
    return os.environ.get('CODEOCEAN_API_URL', 'https://api.codeocean.com')

def create_capsule_from_github_sdk(
    api_token: str,
    api_url: str,
    github_repo_url: str,
    capsule_name: str,
    capsule_description: str
) -> Dict[str, Any]:
    """
    Create a Code Ocean capsule from a GitHub repository using the SDK.
    
    Args:
        api_token: Code Ocean API token
        api_url: Code Ocean API base URL
        github_repo_url: URL of the GitHub repository
        capsule_name: Name for the capsule
        capsule_description: Description for the capsule
    
    Returns:
        Response data from the API
    """
    # Initialize the Code Ocean client
    print(f"Initializing Code Ocean client...")
    print(f"API URL: {api_url}")
    
    try:
        # Create client - try different initialization methods
        client = None
        
        # Method 1: Direct initialization with token and URL
        try:
            client = CodeOceanClient(api_token=api_token, api_url=api_url)
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Initialize with environment variables
        if client is None:
            try:
                os.environ['CODEOCEAN_API_TOKEN'] = api_token
                os.environ['CODEOCEAN_API_URL'] = api_url
                client = CodeOceanClient()
            except Exception as e:
                print(f"Method 2 failed: {e}")
        
        # Method 3: Try default initialization
        if client is None:
            try:
                client = codeocean.CodeOceanClient(api_token=api_token)
            except Exception as e:
                print(f"Method 3 failed: {e}")
        
        if client is None:
            raise Exception("Could not initialize Code Ocean client")
        
        print("✓ Code Ocean client initialized successfully")
        
        # Read metadata if available
        metadata_path = Path(__file__).parent / 'metadata' / 'metadata.yml'
        metadata = {}
        if metadata_path.exists():
            try:
                import yaml
                with open(metadata_path, 'r') as f:
                    metadata = yaml.safe_load(f) or {}
            except ImportError:
                print("Warning: PyYAML not installed, skipping metadata import")
            except Exception as e:
                print(f"Warning: Could not read metadata: {e}")
        
        # Prepare capsule creation parameters
        capsule_params = {
            'name': capsule_name,
            'description': capsule_description,
        }
        
        # Try different ways to specify the GitHub repository
        git_params = [
            {'git': {'url': github_repo_url, 'branch': 'master'}},
            {'git_repository_url': github_repo_url, 'git_branch': 'master'},
            {'repository_url': github_repo_url, 'branch': 'master'},
        ]
        
        # Add metadata if available
        if metadata:
            if 'authors' in metadata:
                capsule_params['authors'] = metadata['authors']
            if 'keywords' in metadata:
                capsule_params['tags'] = metadata['keywords']
            if 'version' in metadata:
                capsule_params['version'] = metadata['version']
        
        # Try to create capsule with different parameter structures
        for i, git_param in enumerate(git_params, 1):
            try:
                params = {**capsule_params, **git_param}
                print(f"\nAttempting to create capsule (method {i})...")
                print(f"Parameters: {json.dumps(params, indent=2)}")
                
                # Try different SDK methods
                result = None
                
                # Method 1: create_capsule
                try:
                    if hasattr(client, 'create_capsule'):
                        result = client.create_capsule(**params)
                except Exception as e:
                    print(f"  create_capsule failed: {e}")
                
                # Method 2: capsules.create
                if result is None:
                    try:
                        if hasattr(client, 'capsules') and hasattr(client.capsules, 'create'):
                            result = client.capsules.create(**params)
                    except Exception as e:
                        print(f"  capsules.create failed: {e}")
                
                # Method 3: post to capsules endpoint
                if result is None:
                    try:
                        if hasattr(client, 'post'):
                            result = client.post('/v1/capsules', json=params)
                    except Exception as e:
                        print(f"  post method failed: {e}")
                
                if result:
                    print(f"✓ Capsule created successfully!")
                    return result if isinstance(result, dict) else result.json() if hasattr(result, 'json') else {'result': str(result)}
                    
            except Exception as e:
                print(f"✗ Method {i} failed: {e}")
                continue
        
        raise Exception("All methods to create capsule failed")
        
    except Exception as e:
        print(f"\n✗ Error creating capsule: {e}", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    print("Code Ocean Capsule Creator (SDK Version)")
    print("=" * 60)
    print(f"Repository: {GITHUB_REPO_URL}")
    print(f"Capsule Name: {CAPSULE_NAME}")
    print("=" * 60)
    
    # Get API token
    api_token = get_api_token()
    if not api_token:
        sys.exit(1)
    
    # Get API URL
    api_url = get_api_url()
    print(f"API URL: {api_url}")
    
    # Create capsule
    result = create_capsule_from_github_sdk(
        api_token=api_token,
        api_url=api_url,
        github_repo_url=GITHUB_REPO_URL,
        capsule_name=CAPSULE_NAME,
        capsule_description=CAPSULE_DESCRIPTION
    )
    
    if result:
        print("\n" + "="*60)
        print("Capsule created successfully!")
        print("="*60)
        print(json.dumps(result, indent=2, default=str))
        
        # Try to extract capsule ID from different response formats
        capsule_id = None
        if isinstance(result, dict):
            capsule_id = result.get('id') or result.get('capsule_id') or result.get('data', {}).get('id')
        elif hasattr(result, 'id'):
            capsule_id = result.id
        
        if capsule_id:
            print(f"\nCapsule ID: {capsule_id}")
            print(f"You can view it at: https://codeocean.com/capsule/{capsule_id}")
    else:
        print("\n" + "="*60, file=sys.stderr)
        print("Capsule creation failed. You can create it manually:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print(f"1. Go to https://codeocean.com", file=sys.stderr)
        print(f"2. Click 'Add Capsule' > 'Clone from Git'", file=sys.stderr)
        print(f"3. Enter repository URL: {GITHUB_REPO_URL}", file=sys.stderr)
        print(f"4. Configure the capsule settings", file=sys.stderr)
        print("="*60, file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

