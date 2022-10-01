echo """
check existing stuffs..
    echo \$PYTHONPATH
    conda env list

run the following commands (dont copy):

    export PYTHONPATH=\"\$PYTHONPATH:\$PWD\"
    conda activate thesis-code
    python ./apps/deep/multi_model_generator.py --config_path "./config/multi_model_generator/linux_test.yml"
"""
