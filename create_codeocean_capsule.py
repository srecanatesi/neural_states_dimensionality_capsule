#!/usr/bin/env python3
"""
Script to create a Code Ocean capsule from a GitHub repository using the Code Ocean API.

Usage:
    python create_codeocean_capsule.py

Environment variables required:
    CODEOCEAN_API_TOKEN - Your Code Ocean API token
    CODEOCEAN_API_URL - Code Ocean API base URL (default: https://api.codeocean.com)

Or set them in a .env file:
    CODEOCEAN_API_TOKEN=your_token_here
    CODEOCEAN_API_URL=https://api.codeocean.com
"""

import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
GITHUB_REPO_URL = "https://github.com/srecanatesi/pipeline-capsule"
CAPSULE_NAME = "Neural States Dimensionality Pipeline"
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

def create_capsule_from_github(
    api_token: str,
    api_url: str,
    github_repo_url: str,
    capsule_name: str,
    capsule_description: str
) -> Dict[str, Any]:
    """
    Create a Code Ocean capsule from a GitHub repository.
    
    Args:
        api_token: Code Ocean API token
        api_url: Code Ocean API base URL
        github_repo_url: URL of the GitHub repository
        capsule_name: Name for the capsule
        capsule_description: Description for the capsule
    
    Returns:
        Response data from the API
    """
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
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
    
    # Prepare capsule creation payload
    payload = {
        'name': capsule_name,
        'description': capsule_description,
        'git_repository_url': github_repo_url,
        'git_branch': 'master',  # or 'main' depending on your default branch
        'is_public': False,  # Set to True if you want it public
    }
    
    # Add metadata if available
    if metadata:
        if 'authors' in metadata:
            payload['authors'] = metadata['authors']
        if 'keywords' in metadata:
            payload['tags'] = metadata['keywords']
        if 'version' in metadata:
            payload['version'] = metadata['version']
    
    # Try different possible API endpoints
    endpoints = [
        f'{api_url}/v1/capsules',
        f'{api_url}/capsules',
        f'{api_url}/api/v1/capsules',
    ]
    
    for endpoint in endpoints:
        try:
            print(f"Attempting to create capsule via {endpoint}...")
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 201:
                print(f"✓ Capsule created successfully!")
                return response.json()
            elif response.status_code == 401:
                print(f"✗ Authentication failed. Please check your API token.", file=sys.stderr)
                print(f"Response: {response.text}", file=sys.stderr)
                break
            elif response.status_code == 404:
                print(f"✗ Endpoint not found: {endpoint}")
                continue
            else:
                print(f"✗ Request failed with status {response.status_code}")
                print(f"Response: {response.text}")
                if response.status_code < 500:  # Don't retry on client errors
                    break
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            continue
    
    # If all endpoints failed, provide manual instructions
    print("\n" + "="*60, file=sys.stderr)
    print("API creation failed. You can create the capsule manually:", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"1. Go to https://codeocean.com", file=sys.stderr)
    print(f"2. Click 'Add Capsule' > 'Clone from Git'", file=sys.stderr)
    print(f"3. Enter repository URL: {github_repo_url}", file=sys.stderr)
    print(f"4. Configure the capsule settings", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    return None

def main():
    """Main function."""
    print("Code Ocean Capsule Creator")
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
    result = create_capsule_from_github(
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
        print(json.dumps(result, indent=2))
        
        if 'id' in result or 'capsule_id' in result:
            capsule_id = result.get('id') or result.get('capsule_id')
            print(f"\nCapsule ID: {capsule_id}")
            print(f"You can view it at: https://codeocean.com/capsule/{capsule_id}")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()

