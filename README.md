# Multi-label Style Change Detection by Solving a Binary Classification Problem



Source code for the paper submitted to the PAN Style Change Detection task @CLEF2021, achieving the best score on task 1 (0.795) and the second best score on task 2 (0.707). 

<!--- [Link to the original paper.]() 
--->


<!--- If you use this resource, please cite the paper:
```

```
--->

To reproduce the results of the paper:

1. Download data from the official task site and save to `./data/train` and `./data/validation`
2. Run `generate_embeddings.py` and `generate_text_features.py` to generate and save feature vectors for training and validation.
3. Run `task1.py`, `task2.py` and `task3.py` to train and save the LightGBM model and stacking ensemble model for each task. This step also validates the models on the 
validation set.
4. To run the model on new data i.e., the test set, run `main.py -i INPUT_FOLDER -o OUTPUT_FOLDER` where INPUT_FOLDER is the directory with document cases and OUTPUT_FOLDER is the directory to write prediction.json files to. 
5. To evaluate, run `evaluator.py -p PREDICTIONS -t SOLUTIONS -o OUTPUT` where PREDICTIONS is the directory with predictions, SOLUTIONS is the directory with solutions and OUTPUT is the directory to output the results file.

## License
Licensed under the [MIT](https://choosealicense.com/licenses/mit/) license.
