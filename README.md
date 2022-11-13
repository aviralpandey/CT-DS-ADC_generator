# CT-DS-ADC_generator
A Continuous Time (CT) Delta Sigma (DS) ADC Generator 

The (I think?) important things to run in a GitHub codespace:

```
conda activate base
conda install -y -c litex-hub open_pdks.sky130a

git clone git@github.com:dan-fritchman/Hdl21.git
cd Hdl21
git checkout 8eef2239920e4bf61e3b378f028e985d1d41a106
cd ..

git clone git@github.com:Vlsir/Vlsir.git
cd Vlsir
git checkout 2079c44fd7b34023c737a7c01ea624b50289524f
cd ..

conda install -y poetry
conda upgrade pip
pip install -e ".[dev]"
```
