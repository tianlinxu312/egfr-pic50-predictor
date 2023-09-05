from warnings import filterwarnings
import os
import json
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

# Silence some expected warnings
filterwarnings("ignore")
# Fix seed for reproducible results
SEED = 22


class PotencyPredictor:
    """
    Class to predict the potency of a compound based on its molecular fingerprint.

    The class provides methods to load data, encode molecules, train a model, predict, and evaluate.
    The model options are:
    - Logistic Regression: "LogReg"
    - Support Vector Machine: "SVM"
    - Random Forest: "RF"
    - Multi-Layer Perceptron: "MLP"

    :args:
        data_path: str
            Path to the data file.
        pic50_threshold: float
            Threshold to define active compounds.
        encode_method: str
            Method to encode the molecules.
        n_bits: int
            Number of bits for the fingerprint.
    :attributes:
        load_data: pd.DataFrame
            Load a ChEMBL DataFrame.
        get_train_and_test_data: None
            Process data and label, and split it into train and test set.
        smiles_to_fp: np.array
            Encode a molecule from a SMILES string into a fingerprint.
        model_training_and_validation: tuple
            Fit a machine learning model on a random train-test split of the data
            and return the performance measures.
        model_performance: tuple
            Calculate model performance results.
        plot_roc_curve: None
            Plot the ROC curve.
        train: None
            Train a model on the training set and save trained model.
    """
    def __init__(self, data_path: str = "https://raw.githubusercontent.com/volkamerlab/"
                                        "teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/"
                                        "data/EGFR_compounds_lipinski.csv",
                 pic50_threshold: float = 8.0,
                 encode_method: str = "maccs",
                 n_bits=2048):
        super().__init__()

        self.data_path = data_path
        self.pic50_threshold = pic50_threshold
        self.data_encode_method = encode_method
        self.n_bits = n_bits
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.models = {}
        self.performance_measures = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the data path provided to this class.
        ----------
        Returns
        -------
        pd.DataFrame
        """
        chembl_df = pd.read_csv(self.data_path, index_col=0)
        chembl_df = chembl_df[["molecule_chembl_id", "smiles", "pIC50"]]

        # label data
        chembl_df["active"] = np.zeros(len(chembl_df))
        # Mark every molecule as active with an pIC50 of >= 6.3, 0 otherwise
        chembl_df.loc[chembl_df[chembl_df.pIC50 >= self.pic50_threshold].index, "active"] = 1.0
        return chembl_df

    def get_train_and_test_data(self, chembl_df: pd.DataFrame) -> None:
        """
        Process data and label, and split it into train and test set.
        Parameters
        ----------
        chembl_df : pd.DataFrame
            The ChEMBL DataFrame.
        Returns
        -------
        None
        """
        compound_df = chembl_df.copy()
        compound_df["fp"] = compound_df["smiles"].apply(self._smiles_to_fp)

        fingerprint_to_model = compound_df.fp.tolist()
        label_to_model = compound_df.active.tolist()

        # Split data into train (80%) and test set (20%)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(fingerprint_to_model,
                                                                                label_to_model,
                                                                                test_size=0.2,
                                                                                random_state=SEED)

    def _smiles_to_fp(self, smiles: str) -> np.array:
        """
        Encode a molecule from a SMILES string into a fingerprint.
        Parameters
        ----------
        smiles : str
            The SMILES string defining the molecule.
        Returns
        -------
        np.array
            The fingerprint array.
        """
        # convert smiles to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        if self.data_encode_method == "maccs":
            return np.array(MACCSkeys.GenMACCSKeys(mol))
        if self.data_encode_method == "morgan2":
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.n_bits)
            return np.array(fpg.GetFingerprint(mol))
        if self.data_encode_method == "morgan3":
            fpg = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=self.n_bits)
            return np.array(fpg.GetFingerprint(mol))
        else:
            # NBVAL_CHECK_OUTPUT
            print(f"Warning: Wrong method specified: {self.data_encode_method}. Default will be used instead.")
            return np.array(MACCSkeys.GenMACCSKeys(mol))

    def _model_training(self, ml_model, model_name: str) -> None:
        """
        Fit a machine learning model on a random train-test split of the data
        and return the performance measures.

        Parameters
        ----------
        ml_model: sklearn model object
            The machine learning model to train.
        model_name: str
            Name of the model (for printing purposes)
        Returns
        -------
        None
        """
        # Fit the model
        ml_model.fit(self.train_x, self.train_y)
        self.models[model_name] = ml_model

    def inference(self, model_name: str, test_x: np.array = None,
                  test_y: np.array = None, verbose: bool = True,
                  rf_n_estimators: int = 100,
                  rf_criterion: str = "entropy",
                  svm_kernel: str = "rbf",
                  svm_gamma: float = 0.1,
                  logr_penalty: str or None = 'l2',
                  mlp_hidden_units: int = 256, mlp_iters: int = 3000,
                  mlp_reg_strength: float = 0.01) -> None:
        """
        Helper function for inference and reporting model performance

        Parameters
        ----------
        model_name: str
            Name of the model (for printing purposes)
        test_x: np.array
            Test set features
        test_y: np.array
            Test set labels
        verbose: bool
            Print performance measure (default = True)
        rf_n_estimators: int
            Number of trees to grow in random forest (default = 100)
        rf_criterion: str
            Cost function to be optimized for a split in random forest (default = "entropy")
        svm_kernel: str
            Kernel function to be used in SVM (default = "rbf")
        svm_gamma: float
            Kernel coefficient for "rbf" kernel in SVM (default = 0.1)
        logr_penalty: str or None
            Penalty to be used in logistic regression (default = 'l2')
        mlp_hidden_units: int
            Number of hidden units in each hidden layer of MLP (default = 256)
        mlp_iters: int
            Number of iterations to train MLP (default = 3000)
        mlp_reg_strength: float
            Regularization strength for MLP (default = 0.01)
        Returns
        -------
        tuple:
            Accuracy, sensitivity, specificity, auc on test set.
        """
        if model_name not in self.models.keys():
            self.train_and_eval(model_name=model_name, verbose=False, rf_n_estimators=rf_n_estimators,
                                rf_criterion=rf_criterion, svm_kernel=svm_kernel, svm_gamma=svm_gamma,
                                logr_penalty=logr_penalty, mlp_hidden_units=mlp_hidden_units, mlp_iters=mlp_iters,
                                mlp_reg_strength=mlp_reg_strength)

        if test_x is None or test_y is None:
            test_x = self.test_x
            test_y = self.test_y

        trained_model = self.models[model_name]
        # Prediction probability on test set
        test_prob = trained_model.predict_proba(test_x)[:, 1]

        # Prediction class on test set
        test_pred = trained_model.predict(test_x)

        # Performance of model on test set
        accuracy = accuracy_score(test_y, test_pred)
        sens = recall_score(test_y, test_pred)
        spec = recall_score(test_y, test_pred, pos_label=0)
        auc = roc_auc_score(test_y, test_prob)

        if verbose:
            # Print performance results
            # NBVAL_CHECK_OUTPUT
            print('--------------------------------------')
            print(f'{model_name} model test performance:')
            print(f"Accuracy: {accuracy:.2}")
            print(f"Sensitivity: {sens:.2f}")
            print(f"Specificity: {spec:.2f}")
            print(f"AUC: {auc:.2f}")

        if test_x is None or test_y is None:
            self.performance_measures[model_name+" new_xy"] = {'accuracy': accuracy, 'sensitivity': sens,
                                                               'specificity': spec, 'auc': auc}
        else:
            self.performance_measures[model_name] = {'accuracy': accuracy, 'sensitivity': sens,
                                                     'specificity': spec, 'auc': auc}

    def plot_roc_curves_for_models(self, test_x: np.array = None, test_y: np.array = None) -> plt.figure:
        """
        Helper function to plot customized roc curve.
        Parameters
        ----------
        Returns
        -------
        plt.figure
        """
        if test_x is None or test_y is None:
            test_x = self.test_x
            test_y = self.test_y

        fig = plt.figure()        # Below for loop iterates through your models list
        for model_name, ml_model in self.models.items():
            # Prediction probability on test set
            test_prob = ml_model.predict_proba(test_x)[:, 1]
            # Prediction class on test set
            # test_pred = ml_model.predict(test_x)
            # Compute False postive rate and True positive rate
            fpr, tpr, thresholds = metrics.roc_curve(test_y, test_prob)
            # Calculate Area under the curve to display on the plot
            auc = roc_auc_score(test_y, test_prob)
            # Plot the computed values
            plt.plot(fpr, tpr, label=f"{model_name} AUC area = {auc:.2f}")

        # Custom settings for the plot
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        return fig

    def train_and_eval(self, model_name: str = 'RF', verbose: bool = True,
                       rf_n_estimators: int = 100,
                       rf_criterion: str = "entropy",
                       svm_kernel: str = "rbf",
                       svm_gamma: float = 0.1,
                       logr_penalty: str or None = 'l2',
                       mlp_hidden_units: int = 256, mlp_iters: int = 3000,
                       mlp_reg_strength: float = 0.01) -> None:
        """
        Train a machine learning model on the training data and save the model.

        Parameters
        ----------
        model_name: str
            Name of the machine learning model to train.
        verbose: bool
            Print performance measure (default = True)
        rf_n_estimators: int
            Number of trees to grow in random forest (default = 100)
        rf_criterion: str
            Cost function to be optimized for a split in random forest (default = "entropy")
        svm_kernel: str
            Kernel function to be used in SVM (default = "rbf")
        svm_gamma: float
            Kernel coefficient for "rbf" kernel in SVM (default = 0.1)
        logr_penalty: str or None
            Penalty to be used in logistic regression (default = 'l2')
        mlp_hidden_units: int
            Number of hidden units in each hidden layer of MLP (default = 256)
        mlp_iters: int
            Number of iterations to train MLP (default = 3000)
        mlp_reg_strength: float
            Regularization strength for MLP (default = 0.01)
        Returns
        -------
        sklearn model object:
            The trained machine learning model.
        """
        if model_name == "RF":
            # Set model parameter for random forest
            param = {
                "n_estimators": rf_n_estimators,  # number of trees to grows
                "criterion": rf_criterion,  # cost function to be optimized for a split
            }

            model = RandomForestClassifier(**param)
        elif model_name == "SVM":
            model = svm.SVC(kernel=svm_kernel, C=1, gamma=svm_gamma, probability=True)
        elif model_name == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(mlp_hidden_units,
                                                      mlp_hidden_units,
                                                      mlp_hidden_units),
                                  max_iter=mlp_iters,
                                  alpha=mlp_reg_strength)
        elif model_name == "LogReg":
            model = LogisticRegression(penalty=logr_penalty)
        else:
            raise ValueError("Wrong model name. Please choose from {RF, SVM, MLP, LogReg}.")

        # fit model
        self._model_training(model, model_name)
        # get model performance
        self.inference(model_name, self.test_x, self.test_y, verbose)


def main(args):
    # create an instance of the class
    predictor = PotencyPredictor(data_path=args.data_path, pic50_threshold=args.pic50_threshold,
                                 encode_method=args.encode_method, n_bits=args.n_bits)
    # load data
    chembl_df = predictor.load_data()
    # encode data, and split into train and test set
    predictor.get_train_and_test_data(chembl_df)
    # fit models
    model_names = args.model_names.split(",")
    for model_name in model_names:
        predictor.train_and_eval(model_name=model_name, verbose=args.verbose, rf_n_estimators=args.rf_n_estimators,
                                 rf_criterion=args.rf_criterion, svm_kernel=args.svm_kernel, svm_gamma=args.svm_gamma,
                                 logr_penalty=args.logr_penalty, mlp_hidden_units=args.mlp_hidden_units,
                                 mlp_iters=args.mlp_iters, mlp_reg_strength=args.mlp_reg_strength)

    # plot roc curve
    if args.plot_fig:
        fig = predictor.plot_roc_curves_for_models()
        fig.show()
        if args.save_fig:
            fig.savefig("roc_curve.png")

    if args.save_perf:
        with open(os.path.join(args.save_path, "performance_measures"), "w") as fp:
            json.dump(predictor.performance_measures, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cot')
    parser.add_argument('-dp', '--data_path',
                        type=str, default="https://raw.githubusercontent.com/volkamerlab/"
                                          "teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/"
                                          "data/EGFR_compounds_lipinski.csv")
    parser.add_argument('-pic', '--pic50_threshold', type=float, default=0.8)
    parser.add_argument('-em', '--encode_method', type=str, default='maccs')
    parser.add_argument('-nb', '--n_bits', type=int, default=2048)
    parser.add_argument('-mns', '--model_names', type=str, default="RF,SVM,MLP,LogReg")
    parser.add_argument('-vb', '--verbose', type=bool, default=True)
    parser.add_argument('-rfne', '--rf_n_estimators', type=int, default=100)
    parser.add_argument('-rfc', '--rf_criterion', type=str, default="entropy")
    parser.add_argument('-sv', '--svm_kernel', type=str, default="rbf")
    parser.add_argument('-sg', '--svm_gamma', type=float, default=0.1)
    parser.add_argument('-lp', '--logr_penalty', type=str or None, default="l2")
    parser.add_argument('-mhu', '--mlp_hidden_units', type=int, default=256)
    parser.add_argument('-mi', '--mlp_iters', type=int, default=3000)
    parser.add_argument('-mrs', '--mlp_reg_strength', type=float, default=0.01)
    parser.add_argument('-pl', '--plot_fig', type=bool, default=True)
    parser.add_argument('-sf', '--save_fig', type=bool, default=False)
    parser.add_argument('-sp', '--save_performance', type=bool, default=False)
    parser.add_argument('-spath', '--save_path', type=str, default="./")

    args = parser.parse_args()
    main(args)