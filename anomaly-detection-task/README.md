This is the code repository for the anomaly detection task.

Prerequisites:
1. Install anaconda
2. Download BPIC2012 and BPIC2013, as well as the synthetic dataset large and huge into the respective folders.

To run Code:
1. Create conda environment form environment.yml:
conda env create --name $yourname --file=./environment.yml

2. Activate conda env
conda activate $yourname

3. Gointo folder "Code"
cd ./Code/python-scripts

4. Create grpah representation
python gen_graphset.py

5. Recreate hyperparameter search or the run with the optimized hyperparameters:
python runner_hyperparam.py

python runner_final.py