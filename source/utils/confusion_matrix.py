from ctypes import Union
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go


def save_confusion_matrix(path: str, y_true: ndarray, y_pred: ndarray, display_labels=None):

    cm = confusion_matrix(y_true, y_pred, labels=display_labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    # fig, ax = plt.subplots(figsize=(20, 20))
    
    # disp.plot(ax=ax)
    # disp.figure_.savefig(os.path.join(path, 'conf_mat.png'), dpi=300)

    fig = plot_confusion_matrix(cm, display_labels, 'Confusion matrix')
    fig.show()
    fig.write_image(os.path.join(path, 'conf_mat.png'))


def plot_confusion_matrix(cm, labels, title):
    # cm : confusion matrix list(list)
    # labels : name of the data list(str)
    # title : title for the heatmap
    data = go.Heatmap(z=cm, y=labels, x=labels)
    annotations = []
    for i, row in enumerate(cm):
        for j, value in reversed(list(enumerate(row))):
            annotations.append(
                {
                    "x": labels[i],
                    "y": labels[j],
                    "font": {"color": "white"},
                    "text": str(value),
                    "xref": "x1",
                    "yref": "y1",
                    "showarrow": False
                }
            )
    layout = {
        "title": title,
        "xaxis": {"title": "Real label"},
        "yaxis": {"title": "Predicted label"},
        "annotations": annotations
    }
    fig = go.Figure(data=data, layout=layout)
    fig['layout']['xaxis']['autorange'] = "reversed"
    return fig

