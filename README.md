Code Ocean pipeline capsule
===========================

This folder contains a minimal capsule to run one session end-to-end:
download the NWB, preprocess, and run HMM cross-validation. It is intended
for batch/array execution by providing `SESSION_ID` per run or a
`parameters.json` sweep in Code Ocean.

Creating the Capsule in Code Ocean
-----------------------------------

### Option 1: Using the Code Ocean API (Automated)

1. Get your Code Ocean API token:
   - Log in to Code Ocean
   - Go to Account Settings > API Tokens
   - Generate a new token

2. Set environment variables:
   ```bash
   export CODEOCEAN_API_TOKEN=your_token_here
   export CODEOCEAN_API_URL=https://api.codeocean.com  # Optional, has default
   ```

3. Run the creation script:
   ```bash
   pip install -r requirements_api.txt
   python create_codeocean_capsule.py
   ```

### Option 2: Manual Creation via Web UI

1. Go to https://codeocean.com
2. Click "Add Capsule" > "Clone from Git"
3. Enter repository URL: `https://github.com/srecanatesi/pipeline-capsule`
4. Configure the capsule:
   - Set the entry point to `run.sh`
   - Configure the environment using `environment.yml`
   - Set up parameter sweeps using `parameters.json`

Entry point
 - run.sh (expects `SESSION_ID` env var)

Outputs
 - /results/sessions_preprocessed/df_<SESSION_ID>.pkl
 - /results/sessions_hmm_crossval/hmm_<SESSION_ID>.pkl
 - /results/logs/session_<SESSION_ID>.log

How it works
 - download_single_session.py: downloads a single session NWB to /data/sessions_nwb/
 - process_session.py: orchestrates download → preprocess → HMM (fast)
 - It invokes top-level scripts: ../preprocessing_nwb.py and ../hmm_crossvalidation_fast.py

Batch runs
 - Provide parameters.json with objects like {"SESSION_ID": "766640955"}
 - Use Code Ocean parameter sweep to run multiple sessions in parallel

Environment
 - See environment.yml for dependencies similar to ssm_env

Notes
 - Internet access is required (public S3).
 - If you prefer the full (non-fast) HMM, update process_session.py to
   invoke ../hmm_crossvalidation.py instead.




