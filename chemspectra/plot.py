import numpy as np
import pandas as pd
from bokeh.palettes import Category10_10
from bokeh.plotting import figure
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.decomposition import PCA


def plot_spectra(data, show_legend=False, title='Mediterranean Honeys FTIR'):
    wavelengths = data.columns.tolist()
    wavelengths = list(map(float, wavelengths))
    samples = data.index.tolist()

    p = figure(title=str(title), x_axis_label='Wavelength', y_axis_label='Intensity',
               width=1600, height=450)
    colors = Category10_10

    for i, sample in enumerate(samples):
        intensity_values = data.loc[sample].tolist()
        intensity_values = list(map(float, intensity_values))
        line = p.line(x=wavelengths, y=intensity_values, line_width=2, color=colors[i % len(colors)])

    if show_legend:
        p.legend.title = 'Samples'
        p.legend.location = 'top_right'

    # p.xgrid.grid_line_color = None
    # p.ygrid.grid_line_color = None

    return p


def plot_oneclasssvm_predictions(svm, x_reduced, x_predicted):
    """
    Plot predictions of OneClassSVM with decision boundary
    :param svm: The OneClassSVM model used
    :param x_reduced: The 2D data after dim. reduction (NEED TO BE 2D)
    :param x_predicted: The predictions of OneClassSVM
    :return: Shows a plot of the predictions of OneClassSVM
    """
    x_min, x_max = x_reduced[:, 0].min() - 0.03, x_reduced[:, 0].max() + 0.03
    y_min, y_max = x_reduced[:, 1].min() - 0.03, x_reduced[:, 1].max() + 0.03

    x_ = np.linspace(x_min, x_max, 500)
    y_ = np.linspace(y_min, y_max, 500)

    xx, yy = np.meshgrid(x_, y_)

    z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')
    b = plt.scatter(x_reduced[x_predicted == 1, 0], x_reduced[x_predicted == 1, 1], c='white', edgecolors='k')
    c = plt.scatter(x_reduced[x_predicted == -1, 0], x_reduced[x_predicted == -1, 1], c='gold', edgecolors='k')
    plt.legend([a.collections[0], b, c], ['learned frontier', 'regular observations', 'abnormal observations'],
               bbox_to_anchor=(1.05, 1))
    plt.axis('tight')
    plt.show()


def plot_oneclasssvm_performance_pca(model, X_train, X_test, y_test, ax=None):
    """
    Plots the decision boundary of a fitted One-Class SVM model and visualizes the correctly
    and incorrectly classified samples from the test set, using PCA to reduce to 2D.

    Parameters:
    - model: The trained One-Class SVM model.
    - X_train: Training data used for fitting the One-Class SVM.
    - X_test: Test data used for prediction.
    - y_test: Ground truth labels for the test set (1 for inliers, -1 for outliers).
    - ax: Optional matplotlib axis object for plotting.
    """
    # Use PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2)
    X_train_2D = pca.fit_transform(X_train)
    X_test_2D = pca.transform(X_test)

    plt.figure(figsize=(10, 6))  # Make the figure wider (adjust values as needed)
    plt.rcParams.update({'font.size': 12})  # Set the font size (adjust the value as needed)

    # Create grid for decision boundary visualization
    xx, yy = np.meshgrid(np.linspace(X_train_2D[:, 0].min() - 1, X_train_2D[:, 0].max() + 1, 500),
                         np.linspace(X_train_2D[:, 1].min() - 1, X_train_2D[:, 1].max() + 1, 500))

    # Predict on the grid points for decision boundary
    Z = model.decision_function(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Predict on the test set and calculate decision function
    decision_scores = model.decision_function(X_test)

    if ax is None:
        ax = plt.gca()

    # Plot decision boundary (the 0 level curve)
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 100), cmap=cm.coolwarm, alpha=0.75)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    # Correct classification:
    # - Outliers (y_test == -1) are correctly classified if decision_scores < 0
    # - Inliers (y_test == 1) are correctly classified if decision_scores >= 0
    correct = ((y_test == 0) & (decision_scores < 0)) | ((y_test == 1) & (decision_scores >= 0))

    # Incorrect classification is the opposite
    incorrect = ~correct

    # Plot the samples
    ax.scatter(X_test_2D[correct, 0], X_test_2D[correct, 1], c='green', label='Correctly classified', edgecolors='k', s=50)
    ax.scatter(X_test_2D[incorrect, 0], X_test_2D[incorrect, 1], c='red', label='Incorrectly classified', edgecolors='k', s=50, marker='x')

    # Expand the plot limits to include the full decision boundary and all points
    ax.set_xlim([xx.min(), xx.max()])
    ax.set_ylim([yy.min(), yy.max()])

    # Formatting
    ax.set_title("One-Class SVM Decision Boundary and Predictions on Adulterated Problem")
    ax.legend()

    # Tight layout to ensure nothing gets cut off
    plt.tight_layout()

    # Show plot
    plt.show()
