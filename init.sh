conda create -n gasrl -y python=3.9
conda activate gasrl
conda install -y numpy=1.25
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y ipykernel
python -m ipykernel install --user --name gasrl --display-name "gasrl"
pip install gymnasium
conda install -y matplotlib
conda install -y pandas
conda install -y seaborn
conda install -y statsmodels
conda install -y mpl_toolkits
conda install -y os
pip install stable-baselines3
