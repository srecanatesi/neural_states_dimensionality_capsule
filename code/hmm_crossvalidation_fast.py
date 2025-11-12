# Code Ocean capsule version - import from parent code directory if available
from pathlib import Path
import sys
import importlib.util

# Try to import from parent code directory
# In Code Ocean, the original script might be at different locations
possible_paths = [
    Path('/code') / 'code' / 'hmm_crossvalidation_fast.py',  # If repo root is mounted
    Path('/code') / 'hmm_crossvalidation_fast.py',  # If code/ is mounted directly
]

hmm_module = None
for parent_hmm_path in possible_paths:
    if parent_hmm_path.exists():
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("hmm_crossvalidation_fast", str(parent_hmm_path))
        hmm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hmm_module)
        break

if hmm_module is None:
    print("ERROR: Could not find hmm_crossvalidation_fast.py at expected locations", file=sys.stderr)
    print(f"Tried: {possible_paths}", file=sys.stderr)
    sys.exit(1)

# Import main function
main = hmm_module.main

if __name__ == '__main__':
    main()




