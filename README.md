# ribsup
Rib Suppression
![](rick-roll.gif)


### Useful code fragments:
1. Clone repository: 
```
git clone https://github.com/rB080/ribsup.git
```
2. Setup virtual environment:
```
python -m venv path/to/env/
```
3. Activate environment:
```
source path/to/env/bin/activate
```
4. Setup requirements:
```
pip install -r reqs.txt
```

### Run Console:
```
python console.py --workspace exp1 --log_name operation_logs --base path/to/workspace/directory --segdata_root path/to/segmentation/dataset --data_root path/to/JSRT/dataset --seg_epochs 10 --seg_lr 1e-3 --trans_epochs 100 --trans_lr 1e-5 --train_segmentor True --make_segmentation True --train_translators True --make_translations True
```