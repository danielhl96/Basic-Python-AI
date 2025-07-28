# Basic-Python-AI
A fundamental Python repository includes Keras and other AI technologies.

In the file example.py, you can see how to use the rcnn. By the way, rcnn.py is the script that runs the model.

To train your own model, you can use rcnn_script_model.py.

In the repository https://github.com/danielhl96/Dataset-AI, you’ll find a script to create your own dataset.

# Installation
pip install -r requirements.txt

# Training

c and b: False means that the entire model will be trained.

python3 rcnn_script_model.py -t "path/train.txt" -v "path/vali.txt" -c false -b false

# Running

To execute the example, use the following command:

python3 example.py
