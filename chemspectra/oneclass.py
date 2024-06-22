from abc import ABC

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from bokeh.models import ColumnDataSource, LabelSet, HoverTool
from bokeh.plotting import figure, show
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.distributions import chi2


class DDSimca(BaseEstimator, TransformerMixin):

    def __init__(self, ncomps=2, alpha=0.01, gamma=0.01):
        self.n_comps = ncomps
        self.alpha = alpha
        self.gamma = gamma
        self.initialize_variables()

    def initialize_variables(self):
        self.od = None
        self.sd = None
        self.extreme_objs = None
        self.outlier_objs = None
        self.scores = None
        self.loadings = None
        self.eigenmatrix = None
        self.od_mean = None
        self.sd_mean = None
        self.dof_od = None
        self.dof_sd = None
        self.od_crit = None
        self.sd_crit = None
        self.od_out = None
        self.sd_out = None
        self.training_set = None
        self.target_class = None
        self.test_set = None
        self.test_set_labels = None
        self.od_test = None
        self.sd_test = None
        self.external_objs_test = None
        self.conf_matrix = None
        self.metrics_list = None

    def fit(self, X):
        self.training_set = X.iloc[:, 1:]
        self.target_class = X.iloc[0, 0]

        D, P = self.decomp(self.training_set)
        self.loadings = P[:, :self.n_comps]
        self.eigenmatrix = D

        sd_vector = self.calculate_sd(self.training_set, self.loadings, D)
        od_vector = self.calculate_od(self.training_set, self.loadings)

        self.dof_sd, self.sd_mean = self.calculate_dof(sd_vector)
        self.dof_od, self.od_mean = self.calculate_dof(od_vector)

        norm_sd_vector = sd_vector / self.sd_mean
        norm_od_vector = od_vector / self.od_mean

        self.sd_crit, self.od_crit = self.calculate_border(self.dof_sd, self.dof_od, self.alpha)
        self.extreme_objs = self.find_extremes(norm_sd_vector, norm_od_vector, self.sd_crit, self.od_crit)

        alpha_out = 1 - ((1 - self.gamma) ** (1 / len(self.training_set)))
        self.sd_out, self.od_out = self.calculate_border(self.dof_sd, self.dof_od, alpha_out)
        self.outlier_objs = self.find_extremes(norm_sd_vector, norm_od_vector, self.sd_out, self.od_out)

        self.scores = self.training_set @ self.loadings
        self.od = od_vector
        self.sd = sd_vector

    def decomp(self, X):
        U, s, VT = np.linalg.svd(X, full_matrices=False)
        D = np.diag(s[:self.n_comps])
        P = VT.T
        return D, P

    def calculate_sd(self, X, P, D):
        T = X @ P
        v_lambda = D.diagonal()
        v_sd = np.sum((T / v_lambda) ** 2, axis=1)
        return v_sd

    def calculate_od(self, X, P):
        E = X @ (np.eye(X.shape[1]) - P @ P.T)
        v_od = np.mean(E ** 2, axis=1)
        return v_od

    def calculate_dof(self, v):
        av = np.mean(v)
        dof = round(2 * (av / np.std(v)) ** 2)
        return dof, av

    def calculate_border(self, dof_sd, dof_od, error):
        d_crit = chi2.ppf(1 - error, dof_sd + dof_od)
        sd_crit = d_crit / dof_sd
        od_crit = d_crit / dof_od
        return sd_crit, od_crit

    def find_extremes(self, norm_sd_vector, norm_od_vector, sd_crit, od_crit):
        od_cur = od_crit * (1 - norm_sd_vector / sd_crit)
        extr_vector = (norm_sd_vector > sd_crit) | (norm_od_vector > od_cur)
        return extr_vector

    def acceptance_plot(self):
        oD = [0 for _ in range(len(self.od))]
        sD = [0 for _ in range(len(self.sd))]

        for i in range(len(self.od)):
            oD[i] = self.transform_(self.od[i] / self.od_mean)
        for i in range(len(self.sd)):
            sD[i] = self.transform_(self.sd[i] / self.sd_mean)

        self.training_set.reset_index(drop=False, inplace=True)
        training_set_names = list(self.training_set['Sample'])
        self.training_set.set_index('Sample', inplace=True)

        point_type_list = [None for _ in range(len(self.extreme_objs))]
        color_list = [None for _ in range(len(self.extreme_objs))]
        for i in range(len(self.extreme_objs)):
            if (not self.extreme_objs[i]) and (not self.outlier_objs[i]):
                point_type_list[i] = 'Regular'
                color_list[i] = 'lime'
            elif not self.outlier_objs[i]:
                point_type_list[i] = 'Extreme'
                color_list[i] = 'red'
            else:
                point_type_list[i] = 'Outlier'
                color_list[i] = 'red'

        plot_df = pd.DataFrame({'Sample': training_set_names, 'sD': sD, 'oD': oD,
                                'Type': point_type_list, 'Color': color_list})
        source = ColumnDataSource(plot_df)
        hover = HoverTool(
            tooltips=[
                ('Sample: ', '@Sample')
            ]
        )

        p = figure(title="Acceptance Plot - Training Set", width=600, height=600)
        p.add_tools(hover)
        p.xaxis.axis_label = "log(1 + h/h_0)"
        p.yaxis.axis_label = "log(1 + v/v_0)"

        x, y = self.border_plot(self.sd_crit, self.od_crit)
        p.line(x, y, line_width=2, color='green')
        x, y = self.border_plot(self.sd_out, self.od_out)
        p.line(x, y, line_width=2, color='red')

        p.scatter('sD', 'oD', size=8, source=source, color='Color', alpha=0.5)
        return p

    def border_plot(self, sd_crit, od_crit):
        x = np.linspace(0, self.transform_(sd_crit), num=100)
        y = np.maximum(0, od_crit / sd_crit * (sd_crit - self.transform_reverse(x)))
        y = self.transform_(y)
        return x, y

    def transform_(self, input):
        return np.log1p(input)

    def transform_reverse(self, input):
        return np.expm1(input)

    def predict(self, Xtest):
        self.test_set = Xtest.iloc[:, 1:]
        self.test_set_labels = Xtest.iloc[:, 0]
        sd_vector_pred = self.calculate_sd(self.test_set, self.loadings, self.eigenmatrix)
        od_vector_pred = self.calculate_od(self.test_set, self.loadings)

        norm_sd_vector = sd_vector_pred / self.sd_mean
        norm_od_vector = od_vector_pred / self.od_mean

        self.external_objs_test = self.find_extremes(norm_sd_vector, norm_od_vector, self.sd_crit, self.od_crit)

        self.sd_test = sd_vector_pred
        self.od_test = od_vector_pred

    def pred_acceptance_plot(self):
        oD = [0 for _ in range(len(self.od_test))]
        sD = [0 for _ in range(len(self.sd_test))]

        for i in range(len(self.od_test)):
            oD[i] = self.transform_(self.od_test[i] / self.od_mean)
        for i in range(len(self.sd_test)):
            sD[i] = self.transform_(self.sd_test[i] / self.sd_mean)

        self.test_set.reset_index(drop=False, inplace=True)
        test_set_names = list(self.test_set['Sample'])
        self.test_set.set_index('Sample', inplace=True)

        point_type_list = [None for _ in range(len(self.external_objs_test))]
        color_list = [None for _ in range(len(self.external_objs_test))]
        for i in range(len(self.external_objs_test)):
            if not self.external_objs_test[i]:
                point_type_list[i] = 'Regular'
                color_list[i] = 'lime' if self.test_set_labels.iloc[i] == 1 else 'red'
            else:
                point_type_list[i] = 'Extreme'
                color_list[i] = 'lime' if self.test_set_labels.iloc[i] == 0 else 'red'

        plot_df = pd.DataFrame({'Sample': test_set_names, 'sD': sD, 'oD': oD,
                                'Type': point_type_list, 'Color': color_list})
        source = ColumnDataSource(plot_df)
        hover = HoverTool(
            tooltips=[
                ('Sample: ', '@Sample')
            ]
        )

        p = figure(title="Acceptance Plot - Test Set", width=600, height=600)
        p.add_tools(hover)
        p.xaxis.axis_label = "log(1 + h/h_0)"
        p.yaxis.axis_label = "log(1 + v/v_0)"

        x, y = self.border_plot(self.sd_crit, self.od_crit)
        p.line(x, y, line_width=2, color='green')

        p.scatter('sD', 'oD', size=8, source=source, color='Color', alpha=0.5)
        return p

    def confusion_matrix(self, plot_cm='off', print_metrics='on'):
        cm_pred = ~np.array(self.external_objs_test)
        cm_actual = (self.target_class == self.test_set_labels).to_numpy()

        self.conf_matrix = metrics.confusion_matrix(cm_actual, cm_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=self.conf_matrix, display_labels=[False, True])

        if plot_cm == 'on':
            cm_display.plot()
            plt.show()

        self.metrics_list = [
            metrics.accuracy_score(cm_actual, cm_pred),
            metrics.precision_score(cm_actual, cm_pred),
            metrics.recall_score(cm_actual, cm_pred),
            metrics.recall_score(cm_actual, cm_pred, pos_label=0),
            metrics.f1_score(cm_actual, cm_pred)
        ]

        if print_metrics == 'on':
            print(f"Accuracy: {self.metrics_list[0]}\n"
                  f"Precision: {self.metrics_list[1]}\n"
                  f"Sensitivity Recall: {self.metrics_list[2]}\n"
                  f"Specificity: {self.metrics_list[3]}\n"
                  f"F1_score: {self.metrics_list[4]}")
        return self.metrics_list
