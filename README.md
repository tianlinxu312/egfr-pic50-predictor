# Predicting potency value (pIC50) for compounds targeting EGFR

## Train and Predict
The PotencyPredictor class provides methods to load data, encode molecules, train a model, predict, and evaluate.

The model options are:
- Logistic Regression: "LogReg"
- Support Vector Machine: "SVM"
- Random Forest: "RF"
- Multi-Layer Perceptron: "MLP"

Users can define the parameters of each model and pass it to the PotencyPredictor.train(args) method. 

The inference method allows users to choose a trained model, and perform inference on a new dataset (data and corresponding labels).

## Install dependencies

Install packages in the [requirements.txt](./requirements.txt): 

```
pip install -r requirements.txt
```

# Run it

Run the code, for example 

```
python3 main.py -dp "path/to/data" -pic 0.8 -em 'maccs' -nb 2048 -mns "RF, MLP" -vb True 
```

More parameter setting are avalable, see [arguments and docstrings in the PotencyPredictor class](./main.py).

## Evaluate 
The performance of the models can be evaluated on accuracy, sensitivity, specificity as well as area under curve (AUC) metrics. 

<img src="./imgs/performance.png" alt="performance" width="700"/>

<img src="./imgs/roc.png" alt="roc" width="700"/>



