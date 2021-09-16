#!/bin/sh
# assumption:
# * virtualenv is avaliable and working

export KATANA_PATH=$(pwd)
export PIP_REQUIRE_VIRTUALENV=true
export PIP_RESPECT_VIRTUALENV=true
virtualenv -p /usr/bin/python3 --clear --no-site-packages $KATANA_PATH/.venv/katana_env
source $KATANA_PATH/.venv/katana_env/bin/activate
pip install -r $KATANA_PATH/requirements.txt --force-reinstall
