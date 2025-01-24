Deleted for anonymity
<!---
# NN-kNN
Update: Now we separate the functionality of the original backup_of_nnknn.ipynb into multiple files, some are in .py files. The main entry point is nnknn.ipynb.

This is for better modularity and organization. It is also easier to extend and build upon. You can still refer to old backup_of_nnknn.ipynb files to learn about the model (which is probably easier since everything is in one file).

# old NN-kNN
The notebook backup_of_nnknn.ipynb provides an example of how to use our code. Once you run the code once, you should understand the workflow. The code can be run on google colab or your own computer (if you copy everything to your own colab, notice there are dependencies so you need to copy the whole project, not just the ipynb).

To run the code, you need to run the file in this order:

Run the section "Setup"
Run "NCA and LMNN setup", if you don't want to use NCA or LMNN, you can skip this step.
Run "Data Sets", choose dataset_name, this will determine the dataset to experiment on. The code for data set preprocessing is in the folder "dataset"
If your data set is a classification data set, run "Classification with NNKNN"
If your data set is a regression data set, run "Regression with NNKNN"
This should be the essentials.

You may run "Results Interpretation" to see how to interpret the model or results. 

The section "Sanity Check" provides a standard neural network so you can compare that with NN-kNN.

Some notes:

Each data set requires more or less a different configuration of parameters. This is currently stored in a config file and handled by the line
```
cfg = conf_file['dataset'][dataset_name]
```
The current config should be relatively good, feel free to tweak it

nnknn.ipynb provides an example of the workflow. The actual code for nn-knn is in the folder "model". If you intend to build and expand your own nnknn model, you can copy the folder "model" and use the nnknn.ipynb as a guide only.
-->

