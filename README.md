# Multivariate Open-set Time-series Anomaly Detection
Official implementation for the manuscript "Open-Set Multivariate Time-Series Anomaly Detection."

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->


## Abstract
Numerous methods for time-series anomaly detection (TSAD) have emerged in recent years, most of which are unsupervised and assume that only normal samples are available during the training phase, due to the challenge of obtaining abnormal data in real-world scenarios. Still, limited samples of abnormal data are often available, albeit they are far from representative of all possible unknown anomaly distributions. Supervised methods can be utilized to classify normal and seen anomalies, but they tend to overfit to the seen anomalies present during training, hence, they fail to generalize to unseen anomalies. This paper is the first to address the open-set TSAD problem, in which a small number of labeled anomalies are introduced in the training phase in order to achieve superior anomaly detection performance compared to both supervised and unsupervised TSAD algorithms. The proposed method, called (M)ultivariate (O)pen-(S)et time-series (A)nomaly (D)etector (MOSAD), is a novel multi-head framework that incorporates a Generative, a Discriminative, and a novel Anomaly-Aware Contrastive head. The latter produces a superior representation space for anomaly detection compared to conventional supervised contrastive learning. Extensive experiments on three real-world datasets establish MOSAD as a new state-of-the-art in the TSAD field.
## Running the code
Download the [SMD](https://github.com/NetManAIOps/OmniAnomaly), [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), and [TUSZ](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) datasets and place them in the directory like so:
  ```
  data/
  │
  ├── PTBXL/
  │   ├── records100
  │   |    ├── 00000
  │   |    ├── 01000
  │   |    └── ...
  │   ├── ptbxl_database.csv
  │   ├── scp_statements.csv  
  │   └── ...
  │
  ├── ServerMachineDataset/ 
  │   ├── interpretation_label
  │   ├── test
  │   ├── test_label
  │   ├── train
  │   └── LICENSE
  │
  └── TUSZ/
      └── eval
           ├── aaaaaaaq
           ├── aaaaaamc
           └── ...
  ```

Run the data preprocessing `.ipynb` files in the `data_preprocessing` folder to generate `.h5` data files.

To train the model, run the command:

  ```
  python train.py -c config.json
  ```

The training script automatically calls the testing script after training. To call the training script manually, run the command:

  ```
  python test.py -r \path\to\checkpoint
  ```

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements
This project is built on the template [pytorch-template](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque)