conda create -n gymnasium -y python=3.9
conda activate gymnasium
conda install -y numpy=1.25
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y ipykernel
python -m ipykernel install --user --name gymnasium --display-name "gymnasium"
sudo apt update
sudo apt-get install -y swig
sudo apt install -y build-essential
pip install gymnasium
pip install gymnasium[box2d]
pip install gymnasium[mujoco]
conda install -y matplotlib
conda install -y pandas
conda install -y seaborn
conda install -y statsmodels.api
conda install -y mpl_toolkits.axes_grid1
conda install -y os
pip install itables
pip install lightgbm
pip install scikit-learn
pip install stable-baselines3
