# Intall Miniconda

First we install miniconda from https://docs.anaconda.com/free/miniconda/

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```
~/miniconda3/bin/conda init bash
```

If you enconunter permission denied problems when creating and environment run 

```
 sudo env "PATH=$PATH" conda  create -n  fingerprint python=3.11

```

Activvate conda environment

```
conda activate fingerprint
```

Install python requirements in conda environment

```
pip install -r requirements.txt
```