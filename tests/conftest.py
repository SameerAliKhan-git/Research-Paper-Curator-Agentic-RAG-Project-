# Test configuration and shared fixtures
import sys
from pathlib import Path

# Ensure src is on the path for all tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
