
# how to setup FNFT
1. clone the [FNFT](https://github.com/FastNFT/FNFT.git) under `project/lib/<FNFT_name>`
2. build by the [fNFT steps](https://github.com/FastNFT/FNFT/blob/master/INSTALL.md)
3. open `project/FNFTpy/auxilary.py` and set the path to the `../lib/<FNFT_name>/.../libfnft.dylib`

# how to install scipy (on M1):
1. on terminal run `brew install scipy`
2. then copy this folder: <pre>`/opt/homebrew/lib/python3.9/site-packages/scipy`</pre>
to: `<project_dir>/venv/lib/python**/site_packages/___HERE__`

# how to install conda 
###(but no need for that, actually)
1. download anaconda for mac
2. create in pycharm new interpreter, choose system
3. apply the path: `/Users/yarden/opt/anaconda3/bin/python`
4. enjoy all packages preinstalled, 
   1. if something still missing use `conda install <package_name>`
   2. or via pycharm (requirements.txt)
   3. some packages might not work via `conda install` (like `ModulationPy`) so just use `pip install`


# Notes
* Tensor flow only works on version 2.3, not 2.4 the latest