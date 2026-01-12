This file details how to reproduce the results of the project.
This codebase builds on https://github.com/mazurowski-lab/finetune-SAM.


To set up an environment for running the code:
conda env create -f environment.yml

To prepare data:
Berkeley DeepDrive: Download "10K Images" and "Labels" from http://bdd-data.berkeley.edu/download.html.
Place the contents of the 10K directory inside the datasets/BDD directory

Cityscapes: Download gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip (11GB) from https://www.cityscapes-dataset.com/downloads/
Place gtFine_trainvaltest and leftImg8bit_trainvaltest inside datasets/city directory

India Driving Dataset: Download Dataset Name: IDD Segmentation (IDD 20k Part II) (5.5 GB) from https://idd.insaan.iiit.ac.in/dataset/download/
Place gt_Fine and leftImg8bit in the dataset/IDD_Segmentation directory

To reproduce preprocessing:
python preprocess_all.py to preprocess all datasets
python preprocess_bdd.py to only preprocess Berkeley DeepDrive
python preprocess_city.py to only preprocess Cityscapes
python preprocess_idd.py to only preprocess India Driving Dataset

For the rest, you should enter the finetune-SAM directory:
cd finetune-SAM

To train models:
python repeat_training.py (This can be modified to train more models or to train with other random seeds)

Once the models are trained you can reproduce figures from the report (all figures can be found in finetune-SAM/curves):
python_display_img.py to reproduce figure 1
python display_results.py to reproduce figure 2
python display_time.py to reproduce figure 3
python train_curves.py to reproduce figure 4
python display_res_report.py to reproduce figure 5