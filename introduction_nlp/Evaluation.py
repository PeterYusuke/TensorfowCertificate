from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def caluculate_results(y_true, y_pre):
    """
    Calculate model accuracy, prediction, recall and f1 score of a bianary classificaion modal.
    """

    # Calculate accuracy score
    model_accuracy = accuracy_score(y_true, y_pre) * 100
    
    # Calculate model prediction, recall and f1-score using "weighted" Average
    model_prediction, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pre, average="weighted")
    model_results = {"accuracy" : model_accuracy,
                     "prediction" : model_prediction,
                     "recall" : model_recall,
                     "f1" : model_f1
    }

    return model_results

