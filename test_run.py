# check_current_code.py
import os
import sys
sys.path.append('backend')

from backend.app.provider_client import get_best_provider
import inspect

print("🔍 CURRENT get_best_provider() CODE:")
print("=" * 60)
print(inspect.getsource(get_best_provider))
print("=" * 60)