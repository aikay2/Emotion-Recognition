# #!/usr/bin/env bash
# # build.sh
# set -e  # exit on error

# # Install a specific Python version via pyenv (Render should have pyenv available)
# echo "Installing Python 3.10.14..."
# pyenv install -s 3.10.14
# pyenv global 3.10.14

# # Upgrade pip and setuptools first
# pip install --upgrade pip setuptools wheel

# # Install dependencies
# pip install -r requirements.txt

# # Any other build steps you need, like migrations
# # python manage.py migrate

set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate