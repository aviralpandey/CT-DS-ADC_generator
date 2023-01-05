# CT-DS-ADC_generator

A Continuous Time (CT) Delta Sigma (DS) ADC Generator 

Installation: 

```
conda install -y -c litex-hub open_pdks.sky130a

git clone git@github.com:dan-fritchman/Hdl21.git
cd Hdl21
git checkout 8eef2239920e4bf61e3b378f028e985d1d41a106
cd ..

git clone git@github.com:Vlsir/Vlsir.git
cd Vlsir
git checkout 2079c44fd7b34023c737a7c01ea624b50289524f
cd ..
python scripts/manage.py install 

conda upgrade -y pip
pip install -e ".[dev]"
```
