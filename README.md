# parafuzz
code for paper: ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP.

You can set up an environment as follows:

```
conda create --name parafuzz python=3.8 -y (help)

conda activate parafuzz

conda install pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0 -c pytorch
pip install --upgrade trojai #if use TrojAI dataset
conda install jsonpickle == 2.2.0
conda install colorama == 0.4.5
asyncio == 3.4.3
numpy == 1.23.4
openai == 0.27.2
pandas == 1.5.2
```

For typical dataset such as Twitter Hate soeech/SST-2, simply run ```python main.py```. No other directories are needed.

If using TrojAI dataset, run ```cd trojai; python parafuzz.py```. 

The RAP directory is constructed on https://github.com/lancopku/RAP. Refer to the original repo for environment and code structure. Only include the necessary code for evaluation against attacks (EP, StyleBKD, HiddenKiller) here. 
