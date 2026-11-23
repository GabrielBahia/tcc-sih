import copy
import importlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

rm = importlib.import_module("5_roda_modelos")

REGISTRO = rm.config.RESULTADOS / "tabelas" / "busca_em_grade.csv"
PACIENCIA = 20
ARVORES_MAXIMAS = 2000
EPOCAS_MAXIMAS = 1000
GRADES = {
    "random_forest": {"n_estimators": [300], "max_depth": [20, None], "min_samples_leaf": [1, 2],
                      "max_features": ["sqrt", 0.1, 0.33], "max_samples": [None, 0.5]},
    "xgboost": {"learning_rate": [0.05, 0.1], "max_depth": [6, 9], "subsample": [1.0],
                "colsample_bytree": [0.3, 0.5, 0.8], "min_child_weight": [1, 5]},
    "mlp": {"hidden_layer_sizes": [(64,), (128, 64)], "learning_rate_init": [0.001, 0.0001],
            "alpha": [0.01, 0.1, 1.0, 3.0], "activation": ["relu", "tanh"],
            "relu_na_saida": [False, True]},
}


class MLPComReLU(MLPRegressor):
    def _initialize(self, y, layer_units, dtype):
        super()._initialize(y, layer_units, dtype)
        self.out_activation_ = "relu"

    def _compute_loss_grad(self, layer, sw_sum, activations, deltas, coef_grads, intercept_grads):
        if layer == self.n_layers_ - 2:
            deltas[layer][activations[-1] == 0] = 0
        super()._compute_loss_grad(layer, sw_sum, activations, deltas, coef_grads, intercept_grads)


def em_dias(valores, log):
    return np.expm1(valores) if log else np.asarray(valores)


def troca_negativos(previsto):
    return np.maximum(previsto, 0)


def treina_random_forest(parametros, x, y, xv, yv, log):
    modelo = RandomForestRegressor(random_state=rm.SEMENTE, n_jobs=-1, **parametros)
    return modelo.fit(x, y), ""


def treina_xgboost(parametros, x, y, xv, yv, log):
    def mae_em_dias(real, previsto):
        return mean_absolute_error(em_dias(real, log), troca_negativos(em_dias(previsto, log)))

    modelo = XGBRegressor(n_estimators=ARVORES_MAXIMAS, early_stopping_rounds=PACIENCIA,
                          eval_metric=mae_em_dias, disable_default_eval_metric=True,
                          random_state=rm.SEMENTE, **parametros)
    modelo.fit(x, y, eval_set=[(xv, yv)], verbose=False)
    return modelo, modelo.best_iteration + 1


def treina_mlp(parametros, x, y, xv, yv, log):
    parametros = dict(parametros)
    classe = MLPComReLU if parametros.pop("relu_na_saida") else MLPRegressor
    modelo = classe(random_state=np.random.RandomState(rm.SEMENTE), **parametros)
    real = em_dias(yv, log)
    melhor, menor_erro, melhor_epoca = None, np.inf, 0
    for epoca in range(1, EPOCAS_MAXIMAS + 1):
        modelo.partial_fit(x, y)
        erro = mean_absolute_error(real, troca_negativos(em_dias(modelo.predict(xv), log)))
        if erro < menor_erro:
            melhor, menor_erro, melhor_epoca = copy.deepcopy(modelo), erro, epoca
        elif epoca - melhor_epoca == PACIENCIA:
            break
    return melhor, melhor_epoca


TREINOS = {"random_forest": treina_random_forest, "xgboost": treina_xgboost, "mlp": treina_mlp}


def registra(nome, log, parametros, parada, real, bruto, segundos):
    previsto = troca_negativos(bruto)
    linha = pd.DataFrame([{
        "modelo": nome, "treino": "log" if log else "dias",
        "parametros": str(parametros), "parada": parada, "n": len(real),
        "mae": round(mean_absolute_error(real, previsto), 4),
        "rmse": round(root_mean_squared_error(real, previsto), 4),
        "r2": round(r2_score(real, previsto), 4),
        "mae_sem_troca": round(mean_absolute_error(real, bruto), 4),
        "negativos": int((bruto < 0).sum()), "segundos": round(segundos), "semente": rm.SEMENTE}])
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    linha.to_csv(REGISTRO, mode="a", header=not REGISTRO.exists(), index=False, encoding="utf-8",
                 sep=";", decimal=",")
    return linha.iloc[0]


def main(nomes, log=False):
    treino, validacao = rm.le("treino"), rm.le("validacao")
    grupos = rm.aprende_grupos(treino)
    x_treino, x_validacao = rm.prepara(treino, grupos), rm.prepara(validacao, grupos)
    y = np.log1p(treino[rm.ALVO]) if log else treino[rm.ALVO]
    yv = np.log1p(validacao[rm.ALVO]) if log else validacao[rm.ALVO]
    real = validacao[rm.ALVO]
    for nome in nomes:
        codificador = rm.cria_codificador(nome, x_treino)
        x, xv = codificador.transform(x_treino), codificador.transform(x_validacao)
        melhor = None
        for parametros in ParameterGrid(GRADES[nome]):
            inicio = time.time()
            modelo, parada = TREINOS[nome](parametros, x, y, xv, yv, log)
            bruto = em_dias(modelo.predict(xv), log)
            linha = registra(nome, log, parametros, parada, real, bruto, time.time() - inicio)
            print(f"{nome}{' (alvo em log)' if log else ''} {parametros} parada {parada}: "
                  f"MAE {linha.mae:.4f}  RMSE {linha.rmse:.4f}  R2 {linha.r2:.4f}  "
                  f"{linha.segundos}s", flush=True)
            if melhor is None or linha.mae < melhor.mae:
                melhor = linha
        print(f"MELHOR {nome}{' (alvo em log)' if log else ''}: {melhor.parametros} "
              f"parada {melhor.parada}: MAE {melhor.mae:.4f}", flush=True)


if __name__ == "__main__":
    log = "log" in sys.argv
    main([a for a in sys.argv[1:] if a != "log"] or list(GRADES), log)
