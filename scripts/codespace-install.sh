# 
# #########################################
# # GitHub CodeSpace Container Installation 
# #########################################
# 
# To be run in the codespace, the first time 
# Could this be part of the container build? Probably some day sure! 
# 

# Conda will at times complain that "your environment is not set up for `conda activate`". 
# If so, this `conda init` bit will set it up. 
conda init bash
source ~/.bashrc 

conda activate base
conda install -y -c litex-hub open_pdks.sky130a
conda upgrade -y pip

# There *should* be compatible versions of VLSIR and Hdl21 on PyPi. 
# If not, install them from source with something like: 
##git clone git@github.com:dan-fritchman/Hdl21.git
##cd Hdl21
##git checkout 8eef2239920e4bf61e3b378f028e985d1d41a106
##cd ..
##git clone git@github.com:Vlsir/Vlsir.git
##cd Vlsir
##git checkout 2079c44fd7b34023c737a7c01ea624b50289524f
##cd ..
##python scripts/manage.py install 

# Pip-install all the other, "normal" dependencies
pip install -e ".[dev]"
