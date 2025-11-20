"""
Test czy wszystkie biblioteki działają
"""

print("🔄 Testuję import bibliotek...")

try:
    import numpy as np
    print("✅ NumPy:", np.__version__)
except ImportError as e:
    print("❌ NumPy:", e)

try:
    import pandas as pd
    print("✅ Pandas:", pd.__version__)
except ImportError as e:
    print("❌ Pandas:", e)

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
except ImportError as e:
    print("❌ PyTorch:", e)

try:
    import sklearn
    print("✅ Scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("❌ Scikit-learn:", e)

try:
    import wfdb
    print("✅ WFDB:", wfdb.__version__)
except ImportError as e:
    print("❌ WFDB:", e)

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib: OK")
except ImportError as e:
    print("❌ Matplotlib:", e)

print("\n🎉 Test zakończony!")
