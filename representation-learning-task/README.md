This is the Code repository accompanying my master thesis "Clustering Process Data using Graph Neural Networks for Representation Learning".
The Code in this foilder implements ProcessGraph2vec runs it on datasets BPIC2015 and BPIC2019 and clusters the results.

Prerequisites:
1. Install anaconda
2. Download BPIC2015 and BPIC2019 into the respective folders. You can find in /Data/EventLogs/BPICX/link-todata.txt links to download the files. (These are the folders to put the data in)

To run Code:
1. Create conda environment form environment.yml:
conda env create --name $yourname --file=./environment.yml

2. Activate conda env
conda activate $yourname

3. Gointo folder "Code"
cd ./Code

4. Run orchestrator.py for the hyperparamaeter search or orchestrator_final.py for the optimized hyperparameter combination.
python orchestrator.py

python orchestrator_final.py

5. You can find under /Data/Database/ the embeddings for each dataset in separate sqlite databases and a graph2vec_info.db databse containing the results.

