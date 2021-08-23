# how to install conda
1. download anaconda for mac
2. create in pycharm new interpreter, choose system
3. apply the path: `/Users/yarden/opt/anaconda3/bin/python`
4. enjoy all packages preinstalled, 
   1. if something still missing use `conda install <package_name>`
   2. or via pycharm (requirements.txt)

# how to setup FNFT
1. clone the [FNFT](https://github.com/FastNFT/FNFT.git) under `project/lib/<FNFT_name>`
2. build by the [fNFT steps](https://github.com/FastNFT/FNFT/blob/master/INSTALL.md)
3. open `project/FNFTpy/auxilary.py` and set the path to the `../lib/<FNFT_name>/.../libfnft.dylib`