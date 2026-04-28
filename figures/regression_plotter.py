import matplotlib.pyplot as plt
import numpy as np


class RegressionPlotter:

    def __init__(self, x, y, model, r2):
        self.x = x
        self.y = y
        self.model = model
        self.r2 = r2
        self.y_pred = model.predict(x)


    def plot_single(self):

        fig, ax = plt.subplots()

        ax.scatter(self.x, self.y, label="Dados reais")
        ax.plot(self.x, self.y_pred, label="Reta de regressão")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Regressão Linear Simples")

        ax.text(
            0.05, 0.90,
            f"$R^2$ = {self.r2:.4f}",
            transform=ax.transAxes
        )

        ax.legend()

        fig.savefig("figures/plot_single.png")
        plt.show()


    def plot_multi(self):

        fig, ax = plt.subplots()

        ax.scatter(self.y, self.y_pred)

        ax.set_xlabel("Valores Reais")
        ax.set_ylabel("Valores Previstos")
        ax.set_title("Regressão Linear Múltipla")

        min_val = min(min(self.y), min(self.y_pred))
        max_val = max(max(self.y), max(self.y_pred))

        ax.text(
            0.05, 0.90,
            f"$R^2$ = {self.r2:.4f}",
            transform=ax.transAxes
        )

        ax.plot([min_val, max_val], [min_val, max_val])

        fig.savefig("figures/plot_multi.png")
        plt.show()