from sklearn.tree import DecisionTreeClassifier
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
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


if __name__ == "__main__":
    mlflow.set_experiment("Homework 2")
    
    with mlflow.start_run(run_name="Decision Tree"):
        decision_tree_model = DecisionTreeClassifier(
            random_state=config["random_state"],
            max_depth=config["decision_tree"]["max_depth"]
        )

        data = get_data()
        train(decision_tree_model, data["x_train"], data["y_train"])
        
        # Глубина дерева, количество листьев, критерии разделения
        mlflow.log_param("max_depth", decision_tree_model.max_depth)
        mlflow.log_param("n_leaves", int(decision_tree_model.get_n_leaves()))
        mlflow.log_param("criterion", decision_tree_model.criterion)
        
        test(decision_tree_model, data["x_test"], data["y_test"])
