import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from logger import logger


class Plotter:
    def plot_hist(self, df: pd.DataFrame, column: str, color: str) -> None:
        # plt.figure(figsize=(15, 10))
        # fig, ax = plt.subplots(1, figsize=(12, 7))
        try:
            sns.displot(data=df, x=column, color=color,
                        kde=True, height=7, aspect=2)
            plt.title(f'Distribution of {column}', size=20, fontweight='bold')
            # plt.show()
            logger.info(f'successfully displayed histogram plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_count(self, df: pd.DataFrame, column: str) -> None:
        try:

            plt.figure(figsize=(12, 7))
            sns.countplot(data=df, x=column)
            plt.title(f'Distribution of {column}', size=20, fontweight='bold')
            # plt.show()
            logger.info(f'successfully displayed count plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_count_with_hue(self, df: pd.DataFrame, x: str, y: str, title: str) -> None:
        try:
            fig, (axis1) = plt.subplots(1, 1, figsize=(15, 4))
            sns.countplot(x=x, hue=y, data=df, palette="husl", ax=axis1)
            plt.title(title, size=20, fontweight='bold')
            # plt.show()
            logger.info(f'successfully displayed count plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_bar(self, df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.barplot(data=df, x=x_col, y=y_col)
            plt.title(title, size=20)
            plt.xticks(rotation=75, fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(ylabel, fontsize=16)
            # plt.show()
            logger.info(f'successfully displayed bar plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_heatmap(self, df: pd.DataFrame, title: str, cbar=False) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                        vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
            plt.title(title, size=18, fontweight='bold')
            # plt.show()
            logger.info(f'successfully displayed heatmap plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_box(self, df: pd.DataFrame, x_col: str, title: str) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df, x=x_col)
            plt.title(title, size=20)
            plt.xticks(rotation=75, fontsize=14)
            # plt.show()
            logger.info(f'successfully displayed box plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_box_multi(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.boxplot(data=df, x=x_col, y=y_col)
            plt.title(title, size=20)
            plt.xticks(rotation=75, fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()
            logger.info(f'successfully displayed box multi plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, title: str,) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.scatterplot(data=df, x=x_col, y=y_col)
            plt.title(title, size=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()
            logger.info(f'successfully displayed scatter plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_catplot(self, df: pd.DataFrame, x_col: str, y_col: str, hue: str, kind: str, title: str) -> None:
        try:
            plt.figure(figsize=(15, 10))
            sns.catplot(data=df, x=x_col, y=y_col, hue=hue, kind=kind)
            # plt.title(title, size=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()
            logger.info(f'successfully displayed catplot plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_catplot_multi(self, df: pd.DataFrame, x_col: str, y_col: str, col: str, col_order: list) -> None:
        try:
            plt.figure(figsize=(12, 7))
            sns.catplot(data=df, x=x_col, y=y_col,
                        col=col, col_order=col_order)
            # plt.title(title, size=20)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # plt.show()
            logger.info(f'successfully displayed catplot plot')
        except Exception as e:
            logger.error(e)
        return plt

    def plot_pie(self, data, labels, title) -> None:
        """Plot pie chart of the data.

        Args:
            data (list): Data to be plotted.
            labels (list): labels of the data.
            colors (list): colors of the data.
        """
        plt.figure(figsize=(12, 7))
        colors = sns.color_palette('bright')
        plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
        plt.title(title, size=20)
        return plt
        # plt.show()

    def plot_time_series(self, data, scaled_data):
        """
         A simple function to plot a time series
        """

        fig = plt.figure()
        gs = GridSpec(2, 1, figure=fig)

        fig.set_figheight(20)
        fig.set_figwidth(30)
        fig.tight_layout(pad=15)

        M = 100
        xticks = ticker.MaxNLocator(M)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(data.index, data.Sales, 'b-')
        ax1.xaxis.set_major_locator(xticks)
        ax1.tick_params(labelrotation=90)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Thousands of Units')
        ax1.title.set_text('Time Series Plot of Sales')
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(scaled_data.index, scaled_data.Sales, 'g-')
        ax2.xaxis.set_major_locator(xticks)
        ax2.tick_params(labelrotation=90)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Scaled Units')
        ax2.title.set_text(
            'Time Series Plot of Scaled Sales')
        ax2.grid(True)
        plt.savefig('../images/timeseriesandtimescaled.png')
        plt.show()

    def plot_histogram(self, data, scaled_data):
        """
         a simple function to plot time series histogram
        """
        fig = plt.figure()
        gs = GridSpec(2, 1, figure=fig)

        fig.set_figheight(10)
        fig.set_figwidth(30)
        fig.tight_layout(pad=6)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(data.Sales, density=True, bins=60)
        ax1.title.set_text('Histogram of Sales')
        ax1.grid(True)

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(scaled_data.Sales, density=True, bins=60)
        ax2.title.set_text('Histogram of the of Scaled  Sales')
        ax2.grid(True)
        plt.savefig('../images/histogramofsales.png')
        plt.show()

    def plot_correlations(self, array: np.array, prefix: str):
        plt.figure(figsize=(30, 5))
        plt.title(
            f"{prefix}  Autocorrelations of Scaled Sales")
        plt.bar(range(len(array)), array)
        plt.grid(True)
        plt.savefig(f'../images/{prefix}autocorrelation.png')
        plt.show()
