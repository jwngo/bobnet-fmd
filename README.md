# Material Recognition 
This repository uses the 
[Flickr Material Database(FMD)](https://people.csail.mit.edu/lavanya/fmd.html)
and 
[Materials in Context Database (MINC-2500)](http://opensurfaces.cs.cornell.edu/publications/minc/) 
dataset.

- generate\_txt.py is used to generate the txt files for 
train and test data. 

## Usage 

```python
python train.py <expname> --alexnet --minc # Uses pre-trained AleNet on Minc
python train.py <expname> --efficientnet --aug # Uses pre-trained EfficientNet and RandAugment
```

Experiments will be saved to results/<expname> with 
run.pth (for resuming) and best\_acc.pth (best model) saved. 

You should add the datasets into FMD/ for FMD
and minc-2500/ for MINC.
