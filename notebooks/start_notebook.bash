#!/bin/bash

# Activate the correct anaconda environment
# conda activate sample_project

# Lab or notebook
read -p 'jupyter type [*lab/notebook] ' JUPYTER_TYPE
JUPYTER_TYPE=${JUPYTER_TYPE:-lab} # Default is lab
read -p 'launch directory [../] ' RUN_DIR
RUN_DIR=${RUN_DIR:-../} # Default is root project directory
echo 'Opening a jupyter '$JUPYTER_TYPE' session in '$RUN_DIR

# Open a jupyter notebook
jupyter $JUPYTER_TYPE $RUN_DIR
