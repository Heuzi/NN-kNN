#!/usr/bin/env python3
"""Verify that all required packages are installed."""

packages_to_check = [
    'matplotlib',
    'numpy',
    'torch',
    'pandas',
    'sklearn',
    'scipy',
    'ipython',
    'jupyter',
    'requests',
    'gymnasium',
    'pydantic',
    'tqdm',
    'wandb',
]

print("Checking installed packages...\n")
failed = []

for package in packages_to_check:
    try:
        if package == 'sklearn':
            import sklearn
            print(f"✓ {package}: {sklearn.__version__}")
        elif package == 'jupyter':
            import jupyter
            print(f"✓ {package}: installed")
        else:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'installed')
            print(f"✓ {package}: {version}")
    except ImportError as e:
        print(f"✗ {package}: NOT INSTALLED - {e}")
        failed.append(package)

if failed:
    print(f"\n{len(failed)} packages missing: {', '.join(failed)}")
else:
    print("\n✓ All packages installed successfully!")
