#!/usr/bin/env bash
# build.sh for Render deployment
set -e  # Exit immediately if a command exits with a non-zero status

echo "=== Starting build process ==="

# Upgrade pip, setuptools, and wheel to avoid build issues
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Optional: run Django migrations (uncomment if needed)
# echo "Applying Django migrations..."
# python manage.py migrate

# Optional: collect static files if your project uses them
# echo "Collecting static files..."
# python manage.py collectstatic --noinput

echo "=== Build completed successfully ==="
