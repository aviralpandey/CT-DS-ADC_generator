# CT-DS-ADC_generator

A Continuous Time (CT) Delta Sigma (DS) ADC Generator 

Installation in the [Docker](./Dockerfile)-configured GitHub dev-container environment: 

```
# The packages containing the PDK model-files are conda-only, 
# and hence not covered by the `pip install` below.
conda activate base
conda install -y -c litex-hub open_pdks.sky130a

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
conda upgrade -y pip
pip install -e ".[dev]"
```
