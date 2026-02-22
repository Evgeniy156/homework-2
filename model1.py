from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from config import config
from data import get_data


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
    auc = roc_auc_score(y_true=y_test, y_score=y_prob, multi_class="ovo")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")
    print(f"AUC ROC: {auc}")
    
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("F1 Score", f1)
    mlflow.log_metric("AUC ROC", auc)
    
    # Сгенерировать и залогировать Confusion Matrix как изображение
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    mlflow.set_experiment("Homework 2")
    
    with mlflow.start_run(run_name="Logistic Regression"):
        logistic_regression_model = LogisticRegression(
            max_iter=config["logistic_regression"]["max_iter"],
        )

        data = get_data()
        train(logistic_regression_model, data["x_train"], data["y_train"])
        
        # Регуляризация (C, penalty) и коэффициенты
        mlflow.log_param("C", logistic_regression_model.C)
        mlflow.log_param("penalty", logistic_regression_model.penalty)
        mlflow.log_text(str(logistic_regression_model.coef_), "regression_coefficients.txt")
        
        test(logistic_regression_model, data["x_test"], data["y_test"])
