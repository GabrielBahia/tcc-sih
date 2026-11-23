import ast
import copy
import importlib
import json
import re
import time
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.tree import export_graphviz
from xgboost.callback import TrainingCallback

aj = importlib.import_module("6_ajusta_modelos")
rm = aj.rm

REGISTRO = rm.config.RESULTADOS / "tabelas" / "painel.csv"
LOTE_DE_ARVORES = 25
NOMES = {"regressao_linear": "Regressão linear", "random_forest": "Random Forest", "xgboost": "XGBoost",
         "mlp": "MLP"}
DESCRICAO = {
    "SEXO": "sexo",
    "CAR_INT": "caráter da internação",
    "RACA_COR": "raça ou cor",
    "NAT_JUR": "natureza jurídica do hospital",
    "ESPEC": "especialidade do leito",
    "CNES": "hospital",
    "cid_item": "item da Lista de Morbidade",
    "MUNIC_RES": "município de residência",
    "PROC_SOLIC": "procedimento solicitado no laudo",
    "idade": "idade em anos, com casa decimal",
    "DT_INTER": "data de entrada",
}
AJUDA = {
    "n_estimators": "Número de árvores.",
    "max_depth": "Profundidade máxima de cada árvore. None deixa sem limite.",
    "min_samples_leaf": "Mínimo de internações em cada folha.",
    "max_features": "Colunas sorteadas em cada divisão. sqrt é a raiz quadrada do total, um decimal é a fração "
                    "do total e um inteiro é a quantidade.",
    "learning_rate": "Fator que multiplica a contribuição de cada árvore nova. Menor dá passos menores.",
    "subsample": "Fração das internações sorteadas para cada árvore.",
    "colsample_bytree": "Fração das colunas sorteadas para cada árvore.",
    "hidden_layer_sizes": "Neurônios de cada camada oculta, como (128, 64).",
    "learning_rate_init": "Tamanho do passo inicial no ajuste dos pesos.",
    "alpha": "Força da penalidade L2 sobre os pesos.",
    "relu_na_saida": "True põe a ReLU na saída da rede, que então nunca prevê menos de 0.",
    "max_samples": "Fração das internações sorteadas para cada árvore. None usa o tamanho do treino inteiro.",
    "min_child_weight": "Peso mínimo das internações em cada folha. Maior deixa as árvores mais simples.",
    "activation": "Ativação das camadas ocultas.",
}
FAIXAS = [(0, 3, "até 3 dias"), (4, 14, "4 a 14 dias"), (15, np.inf, "15 ou mais")]
VIRGULA = alt.Axis(labelExpr="replace(datum.label, '.', ',')")
PRETO, CINZA_ESCURO, CINZA = "#000000", "#555555", "#999999"


@st.cache_resource
def carrega():
    treino, validacao = rm.le("treino"), rm.le("validacao")
    grupos = rm.aprende_grupos(treino)
    return treino, validacao, grupos, rm.prepara(treino, grupos), rm.prepara(validacao, grupos)


@st.cache_resource
def codificador(nome):
    return rm.cria_codificador(nome, carrega()[3])


@st.cache_resource(max_entries=1)
def matrizes(nome):
    _, _, _, x_treino, x_validacao = carrega()
    return codificador(nome).transform(x_treino), codificador(nome).transform(x_validacao)


def nomes_das_colunas(nome):
    nomes = []
    for coluna in codificador(nome).get_feature_names_out():
        grupo, resto = coluna.split("__", 1)
        variavel = next((c for c in rm.CATEGORICAS if grupo == "categoricas" and resto.startswith(c + "_")), None)
        nomes.append(f"{variavel} = {resto[len(variavel) + 1:]}" if variavel else resto)
    return nomes


def br(valor, casas=4):
    return f"{valor:,.{casas}f}".replace(",", "x").replace(".", ",").replace("x", ".")


def mostra_tabela(tabela, onde=st):
    tabela = tabela.copy()
    for coluna in tabela.select_dtypes("number"):
        inteiros = bool((tabela[coluna].dropna() % 1 == 0).all())
        tabela[coluna] = tabela[coluna].map(lambda v: "" if pd.isna(v) else br(v, 0 if inteiros else 4))
    onde.dataframe(tabela.fillna(""), hide_index=True, height="content")


def le_registro(caminho):
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, sep=";", decimal=",")


def grava(linha):
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([linha]).to_csv(REGISTRO, mode="a", header=not REGISTRO.exists(), index=False, encoding="utf-8",
                                 sep=";", decimal=",")


def melhor_da_grade(nome, escala):
    busca = le_registro(aj.REGISTRO)
    if busca.empty:
        return None
    busca = busca[(busca.modelo == nome) & (busca.treino == escala)]
    if busca.empty:
        return None
    return busca.loc[busca.mae.idxmin()]


def mae_da_configuracao_padrao(nome, escala):
    padrao = le_registro(rm.REGISTRO)
    padrao = padrao[(padrao.experimento == "configuracao_padrao") & (padrao.modelo == nome)
                    & (padrao.observacao.fillna("") == ("alvo em log" if escala == "log" else ""))]
    if padrao.empty:
        return None
    return float(str(padrao.mae.iloc[-1]).replace(",", "."))


def valor(texto):
    try:
        return ast.literal_eval(texto)
    except (ValueError, SyntaxError):
        return texto


def treina_random_forest(parametros, x, y, xv, yv, log, mostra):
    modelo = RandomForestRegressor(random_state=rm.SEMENTE, n_jobs=-1, warm_start=True, **parametros)
    real = aj.em_dias(yv, log)
    total = parametros["n_estimators"]
    for arvores in [*range(LOTE_DE_ARVORES, total, LOTE_DE_ARVORES), total]:
        modelo.set_params(n_estimators=arvores).fit(x, y)
        mostra(arvores, mean_absolute_error(real, aj.em_dias(modelo.predict(xv), log)))
    return modelo, ""


class Acompanha(TrainingCallback):
    def __init__(self, mostra):
        self.mostra = mostra

    def after_iteration(self, model, epoch, evals_log):
        self.mostra(epoch + 1, evals_log["validation_0"]["mae_em_dias"][-1])
        return False


def treina_xgboost(parametros, x, y, xv, yv, log, mostra):
    return aj.treina_xgboost({**parametros, "callbacks": [Acompanha(mostra)]}, x, y, xv, yv, log)


def treina_mlp(parametros, x, y, xv, yv, log, mostra):
    parametros = dict(parametros)
    classe = aj.MLPComReLU if parametros.pop("relu_na_saida") else MLPRegressor
    modelo = classe(random_state=np.random.RandomState(rm.SEMENTE), **parametros)
    real = aj.em_dias(yv, log)
    melhor, menor_erro, melhor_epoca = None, np.inf, 0
    for epoca in range(1, aj.EPOCAS_MAXIMAS + 1):
        modelo.partial_fit(x, y)
        erro = mean_absolute_error(real, aj.troca_negativos(aj.em_dias(modelo.predict(xv), log)))
        mostra(epoca, erro)
        if erro < menor_erro:
            melhor, menor_erro, melhor_epoca = copy.deepcopy(modelo), erro, epoca
        elif epoca - melhor_epoca == aj.PACIENCIA:
            break
    return melhor, melhor_epoca


TREINOS = {"random_forest": treina_random_forest, "xgboost": treina_xgboost, "mlp": treina_mlp}


def parametros_padrao(nome, modelo):
    if nome == "xgboost":
        arvore = json.loads(modelo.get_booster().save_config())["learner"]["gradient_booster"]["tree_train_param"]
        return {"n_estimators": modelo.get_booster().num_boosted_rounds(),
                "learning_rate": round(float(arvore["eta"]), 4), "max_depth": int(arvore["max_depth"]),
                "subsample": float(arvore["subsample"]), "colsample_bytree": float(arvore["colsample_bytree"])}
    parametros = modelo.get_params()
    return {chave: parametros[chave] for chave in aj.GRADES.get(nome, {}) if chave in parametros}


def grafico_curva(curva, eixo, parada=None):
    curva = curva.assign(mae_texto=curva.mae.map(br))
    x = alt.X("passo:Q", title=eixo.capitalize(), axis=alt.Axis(format="d"))
    y = alt.Y("mae:Q", title="MAE na validação (dias)", scale=alt.Scale(zero=False), axis=VIRGULA)
    dica = [alt.Tooltip("passo:Q", title=eixo.capitalize()), alt.Tooltip("mae_texto:N", title="MAE")]
    grafico = alt.Chart(curva).mark_line(strokeWidth=2, color=PRETO).encode(x=x, y=y) + \
        alt.Chart(curva).mark_circle(size=30 if len(curva) <= 150 else 4, color=PRETO).encode(x=x, y=y, tooltip=dica)
    if parada:
        grafico += alt.Chart(pd.DataFrame({"passo": [parada]})).mark_rule(strokeDash=[4, 4], color=CINZA).encode(
            x="passo:Q", tooltip=[alt.Tooltip("passo:Q", title="Ponto escolhido")])
    return grafico


def grafico_previsto(real, previsto):
    tabela = pd.DataFrame({"real": real, "previsto": previsto}).groupby("real").previsto.agg(["mean", "size"])
    tabela = tabela[tabela["size"] >= 30].reset_index()
    tabela["media_texto"] = tabela["mean"].map(lambda v: br(v, 2))
    limite = float(tabela.real.max())
    diagonal = alt.Chart(pd.DataFrame({"real": [0, limite], "mean": [0, limite]})).mark_line(
        strokeDash=[4, 4], color=CINZA).encode(x="real:Q", y="mean:Q")
    pontos = alt.Chart(tabela).mark_line(strokeWidth=2, color=PRETO,
                                         point=alt.OverlayMarkDef(size=40, color=PRETO)).encode(
        x=alt.X("real:Q", title="Dias reais de internação"),
        y=alt.Y("mean:Q", title="Dias previstos (média)"),
        tooltip=[alt.Tooltip("real:Q", title="Dias reais"), alt.Tooltip("media_texto:N", title="Previsão média"),
                 alt.Tooltip("size:Q", title="Internações")])
    return diagonal + pontos


def grafico_faixas(real, previsto):
    linhas = []
    for inicio, fim, rotulo in FAIXAS:
        dentro = (real >= inicio) & (real <= fim)
        mae = mean_absolute_error(real[dentro], previsto[dentro])
        linhas.append({"faixa": f"{rotulo}: MAE {br(mae, 2)}", "mae": mae, "internacoes": br(dentro.sum(), 0)})
    return alt.Chart(pd.DataFrame(linhas)).mark_bar(size=48, color=CINZA_ESCURO).encode(
        x=alt.X("faixa:N", sort=None, title="Dias reais de internação", axis=alt.Axis(labelAngle=0, labelLimit=0)),
        y=alt.Y("mae:Q", title="MAE (dias)"),
        tooltip=[alt.Tooltip("faixa:N", title="Faixa"), alt.Tooltip("internacoes:N", title="Internações")])


def virgula_nos_rotulos(dot):
    return re.sub(r'label="[^"]*"', lambda m: re.sub(r"(\d)\.(\d)", r"\1,\2", m.group(0)), dot)


def traduz_sklearn(dot):
    return virgula_nos_rotulos(dot.replace("samples = ", "internações = ").replace("value = ", "média = ")
                               .replace('headlabel="True"', 'headlabel="sim"')
                               .replace('headlabel="False"', 'headlabel="não"'))


def traduz_xgboost(dot, niveis):
    nivel = {"0": 0}
    for pai, filho in re.findall(r"(\d+) -> (\d+)", dot):
        nivel[filho] = nivel[pai] + 1
    linhas = []
    for linha in dot.splitlines():
        no = re.match(r"\s*(\d+)(?: -> (\d+))?", linha)
        if not no or nivel[no.group(2) or no.group(1)] <= niveis:
            linhas.append(linha)
    dot = "\n".join(linhas)
    dot = re.sub(r' color="#[0-9A-F]{6}"', "", dot).replace("graph [ rankdir=TB ]",
                                                              "graph [ rankdir=TB ] node [ shape=box, style=rounded ]")
    return virgula_nos_rotulos(dot.replace('"yes, missing"', '"sim"').replace('"no, missing"', '"não"')
                               .replace('"yes"', '"sim"').replace('"no"', '"não"').replace("leaf=", "folha: "))


def executa(nome, escala, procedimento, combinacoes):
    log = escala == "log"
    treino, validacao, _, _, _ = carrega()
    x, xv = matrizes(nome)
    y = np.log1p(treino[rm.ALVO]) if log else treino[rm.ALVO]
    yv = np.log1p(validacao[rm.ALVO]) if log else validacao[rm.ALVO]
    real = validacao[rm.ALVO].to_numpy()
    eixo = "épocas" if nome == "mlp" else "árvores"
    linhas, registros, melhor = [], [], None
    st.subheader("2. Execução")
    barra = st.progress(0.0)
    area_passo, area_curva, area_tabela, area_log = st.empty(), st.empty(), st.empty(), st.empty()

    def escreve(texto):
        linhas.append(f"{datetime.now():%H:%M:%S} {texto}")
        area_log.code("\n".join(linhas), language=None)

    escreve(f"{NOMES[nome]}, treino em {escala}, {len(combinacoes)} combinação(ões), "
            f"{x.shape[0]} internações no treino, {xv.shape[0]} na validação e {x.shape[1]} colunas")
    for indice, parametros in enumerate(combinacoes, 1):
        barra.progress((indice - 1) / len(combinacoes), f"Combinação {indice} de {len(combinacoes)}")
        escreve(f"Começou {parametros or 'a configuração padrão'}")
        pontos = []
        inicio = time.time()

        def mostra(passo, erro):
            pontos.append({"passo": passo, "mae": erro})
            if nome != "xgboost" or passo % 20 == 0:
                area_passo.caption(f"{passo} {eixo}, MAE {br(erro)}, {time.time() - inicio:.0f} s")
                area_curva.altair_chart(grafico_curva(pd.DataFrame(pontos), eixo))
            if passo % (200 if nome == "xgboost" else 10) == 0:
                escreve(f"  {passo} {eixo}: MAE {erro:.4f}")

        if procedimento == "padrao":
            with st.spinner("Treinando...", show_time=True):
                modelo = clone(rm.MODELOS[nome]).fit(x, y)
            parada = getattr(modelo, "n_iter_", "")
            parametros = parametros_padrao(nome, modelo)
        else:
            modelo, parada = TREINOS[nome](parametros, x, y, xv, yv, log, mostra)
        bruto = aj.em_dias(modelo.predict(xv), log)
        previsto = aj.troca_negativos(bruto)
        linha = {"data": f"{datetime.now():%Y-%m-%d %H:%M}", "procedimento": procedimento, "modelo": nome,
                 "treino": escala, "parametros": str(parametros), "parada": parada, "n": len(real),
                 "mae": round(mean_absolute_error(real, previsto), 4),
                 "rmse": round(root_mean_squared_error(real, previsto), 4), "r2": round(r2_score(real, previsto), 4),
                 "negativos": int((bruto < 0).sum()), "segundos": round(time.time() - inicio),
                 "semente": rm.SEMENTE}
        grava(linha)
        registros.append(linha)
        escreve(f"Terminou {parametros} parada {parada}: MAE {linha['mae']:.4f}  RMSE {linha['rmse']:.4f}  "
                f"R2 {linha['r2']:.4f}  negativos {linha['negativos']}  {linha['segundos']} s")
        mostra_tabela(pd.DataFrame(registros), area_tabela)
        if melhor is None or linha["mae"] < melhor["linha"]["mae"]:
            melhor = {"linha": linha, "modelo": modelo, "previsto": previsto, "curva": pd.DataFrame(pontos)}
    escreve(f"MELHOR {melhor['linha']['parametros']} parada {melhor['linha']['parada']}: "
            f"MAE {melhor['linha']['mae']:.4f}")
    st.session_state.rodada = {"nome": nome, "escala": escala, "procedimento": procedimento, "eixo": eixo,
                               "real": real, "log": linhas, "registros": pd.DataFrame(registros), **melhor}
    st.rerun()


def mostra_variaveis():
    treino, validacao, grupos, x_treino, _ = carrega()
    arvores = pd.Series([n.split(" = ")[0] for n in nomes_das_colunas("random_forest")]).value_counts()
    linear = pd.Series([n.split(" = ")[0] for n in nomes_das_colunas("regressao_linear")]).value_counts()
    linhas = []
    for coluna in rm.CATEGORICAS:
        frequencia = x_treino[coluna].value_counts(normalize=True)
        resumo = f"{br(treino[coluna].nunique(), 0)} valores"
        if x_treino[coluna].nunique() != treino[coluna].nunique():
            resumo += f", {br(x_treino[coluna].nunique(), 0)} depois de agrupar"
        resumo += f". Mais comum: {frequencia.index[0]} ({frequencia.iloc[0]:.0%})"
        linhas.append([coluna, DESCRICAO[coluna], resumo, arvores[coluna], linear[coluna]])
    idade = treino.idade
    linhas.append(["idade", DESCRICAO["idade"],
                   f"de {idade.min():.0f} a {idade.max():.0f} anos, mediana {idade.median():.0f}", 1, 1])
    entrada = pd.to_datetime(treino.DT_INTER.astype(str), format="%Y%m%d")
    linhas.append(["DT_INTER", DESCRICAO["DT_INTER"],
                   f"de {entrada.min():%d/%m/%Y} a {entrada.max():%d/%m/%Y}", len(rm.DATAS), len(rm.DATAS)])
    tabela = pd.DataFrame(linhas, columns=["preditora", "o que é", "no treino", "colunas (Random Forest, "
                                           "XGBoost e MLP)", "colunas (regressão linear)"])
    mostra_tabela(tabela)
    st.caption(f"Total: {tabela.iloc[:, 3].sum()} colunas no Random Forest, no XGBoost e na MLP, e "
               f"{tabela.iloc[:, 4].sum()} na regressão linear. Cada valor vira uma coluna de 0 ou 1 (com dois "
               f"valores, uma coluna só). A regressão linear tira também a coluna do valor mais comum.")
    procedimentos, formas, municipios = grupos
    st.markdown(f"- Procedimento: ficam os {len(procedimentos)} mais frequentes, que juntos cobrem 95% das "
                f"internações do treino. Os outros viram a forma de organização, os 6 primeiros dígitos "
                f"({len(formas)} formas). Forma que não apareceu no treino vira `desconhecido`.\n"
                f"- Município de residência: ficam os {len(municipios)} mais frequentes, que cobrem 95% do "
                f"treino, e o resto vira `outros`.\n"
                f"- Data de entrada: o dia da semana e o dia do mês entram como seno e cosseno, e o feriado "
                f"como 0 ou 1.\n"
                f"- Idade: na MLP entra padronizada (média 0 e desvio padrão 1). Nos outros modelos entra como "
                f"está.")
    st.subheader("Alvo")
    alvo = pd.DataFrame([{"conjunto": conjunto, "internações": br(len(d), 0), "média": br(d[rm.ALVO].mean(), 2),
                          "mediana": br(d[rm.ALVO].median(), 0), "máximo": br(d[rm.ALVO].max(), 0)}
                         for conjunto, d in [("treino", treino), ("validação", validacao)]])
    st.caption("`los`: dias de internação, a coluna que o modelo tenta acertar.")
    mostra_tabela(alvo)


def mostra_configuracao():
    st.subheader("1. Configuração")
    c1, c2, c3 = st.columns(3)
    nome = c1.selectbox("Modelo", list(NOMES), format_func=NOMES.get, key="modelo")
    escala = c2.radio("Treino", ["dias", "log"], horizontal=True, key="escala",
                      format_func={"dias": "em dias", "log": "em log(1 + dias)"}.get,
                      help="Em log, o modelo aprende log(1 + dias) e a previsão volta para dias antes de medir.")
    if nome == "regressao_linear":
        c3.caption("A regressão linear não tem parâmetro para ajustar.")
        return nome, escala, "padrao", [{}]
    procedimento = c3.radio("Parâmetros", ["padrao", "ajuste"], horizontal=True, key="procedimento",
                            format_func={"padrao": "padrão", "ajuste": "escolher"}.get,
                            help="Padrão: os valores da biblioteca, como no 5_roda_modelos.py. Escolher: como no "
                                 "6_ajusta_modelos.py, com parada quando o MAE da validação para de melhorar.")
    if procedimento == "padrao":
        return nome, escala, procedimento, [{}]
    melhor = melhor_da_grade(nome, escala)
    inicial = ast.literal_eval(melhor.parametros) if melhor is not None else {}
    escolhas = {}
    for coluna, (parametro, valores) in zip(st.columns(len(aj.GRADES[nome])), aj.GRADES[nome].items()):
        opcoes = [str(v) for v in valores]
        padrao = [str(inicial[parametro])] if parametro in inicial else opcoes[:1]
        escolhidos = coluna.multiselect(parametro, opcoes + [p for p in padrao if p not in opcoes], default=padrao,
                                        accept_new_options=True, help=AJUDA[parametro],
                                        key=f"{nome}_{escala}_{parametro}")
        escolhas[parametro] = [valor(t) for t in escolhidos]
    combinacoes = list(ParameterGrid(escolhas)) if all(escolhas.values()) else []
    st.caption("Começa na melhor combinação da busca em grade. Mais de um valor testa todas as combinações. "
               "Para digitar um valor novo, use ponto no decimal.")
    return nome, escala, procedimento, combinacoes


def mostra_por_dentro(r):
    nome, modelo = r["nome"], r["modelo"]
    nomes = nomes_das_colunas(nome)
    if nome == "regressao_linear":
        coeficientes = pd.DataFrame({"coluna": nomes, "coeficiente": modelo.coef_})
        maiores = coeficientes.loc[coeficientes.coeficiente.abs().sort_values(ascending=False).index].head(20)
        st.caption(f"Previsão = {br(modelo.intercept_)} + soma de cada coeficiente vezes a sua coluna"
                   f"{', em log(1 + dias)' if r['escala'] == 'log' else ', em dias'}. Os 20 maiores, de "
                   f"{len(nomes)} colunas.")
        mostra_tabela(maiores)
    elif nome == "random_forest":
        c1, c2 = st.columns(2)
        indice = c1.number_input("Árvore", 1, len(modelo.estimators_), 1, key="arvore_rf")
        niveis = c2.slider("Níveis mostrados", 1, 5, 3, key="niveis_rf")
        arvore = modelo.estimators_[indice - 1]
        st.caption(f"Esta árvore tem {br(arvore.tree_.node_count, 0)} nós e {arvore.get_depth()} níveis. A "
                   f"previsão é a média das {len(modelo.estimators_)} árvores"
                   f"{', em log(1 + dias)' if r['escala'] == 'log' else ''}. `ESPEC = 03 <= 0,5` quer dizer "
                   f"que não é da especialidade 03.")
        dot = export_graphviz(arvore, max_depth=niveis, feature_names=nomes, impurity=False, rounded=True,
                              precision=2)
        st.graphviz_chart(traduz_sklearn(dot))
    elif nome == "xgboost":
        booster = modelo.get_booster()
        total = r["linha"]["parada"] or booster.num_boosted_rounds()
        c1, c2 = st.columns(2)
        indice = c1.number_input("Árvore", 1, int(total), 1, key="arvore_xgb")
        niveis = c2.slider("Níveis mostrados", 1, 6, 3, key="niveis_xgb")
        arvore = booster[indice - 1:indice]
        arvore.feature_names = nomes
        st.caption(f"{total} árvores. Cada uma soma o valor da sua folha à previsão"
                   f"{', em log(1 + dias)' if r['escala'] == 'log' else ', em dias'}, corrigindo o erro das "
                   f"anteriores. `ESPEC = 03<1` quer dizer que não é da especialidade 03.")
        st.graphviz_chart(traduz_xgboost(arvore.get_dump(dump_format="dot")[0], niveis))
    else:
        camadas = [modelo.coefs_[0].shape[0], *[c.shape[1] for c in modelo.coefs_]]
        rotulos = ([f"{camadas[0]} entradas"] + [f"{n} neurônios" for n in camadas[1:-1]] + ["1 saída"])
        dot = ("digraph { rankdir=LR; node [shape=box, style=rounded]; "
               + " -> ".join(f'"{rotulo}"' for rotulo in rotulos) + " }")
        pesos = sum(c.size for c in modelo.coefs_) + sum(b.size for b in modelo.intercepts_)
        st.caption(f"{br(pesos, 0)} pesos aprendidos. Ativação: {modelo.activation}. Saída"
                   f"{' em log(1 + dias)' if r['escala'] == 'log' else ' em dias'}"
                   f"{', com ReLU' if modelo.out_activation_ == 'relu' else ''}.")
        st.graphviz_chart(dot)


def mostra_resultado(r):
    linha = r["linha"]
    st.subheader("2. Resultado")
    st.markdown(f"**{NOMES[r['nome']]}, treino em {r['escala']}** `{linha['parametros']}`")
    colunas = st.columns(6)
    colunas[0].metric("MAE (dias)", br(linha["mae"]), help="Erro absoluto médio: quantos dias a previsão erra, "
                                                           "em média.")
    colunas[1].metric("RMSE (dias)", br(linha["rmse"]), help="Raiz do erro quadrático médio. Pesa mais os erros "
                                                             "grandes.")
    colunas[2].metric("R²", br(linha["r2"]), help="Parte da variação dos dias que o modelo explica. 1 acerta tudo, "
                                                  "0 é o mesmo que prever a média.")
    colunas[3].metric("Negativos", br(linha["negativos"], 0), help="Previsões abaixo de zero dia, trocadas por zero antes de medir.")
    colunas[4].metric("Parada", str(linha["parada"]) or "não tem", help="Árvores ou épocas do modelo final.")
    colunas[5].metric("Tempo", f"{linha['segundos']} s")
    padrao, melhor = mae_da_configuracao_padrao(r["nome"], r["escala"]), melhor_da_grade(r["nome"], r["escala"])
    referencias = [f"configuração padrão {br(padrao)}"] if padrao is not None else []
    referencias += [f"melhor da busca em grade {br(melhor.mae)}"] if melhor is not None else []
    if referencias:
        st.caption(f"MAE já registrado para este modelo e treino: {' e '.join(referencias)}.")
    momento = f"{datetime.now():%Y%m%d_%H%M}"
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    c1.download_button("Resultado (CSV)", r["registros"].to_csv(sep=";", decimal=",", index=False)
                       .encode("utf-8-sig"), f"painel_{r['nome']}_{r['escala']}_{momento}.csv", "text/csv",
                       on_click="ignore", icon=":material/download:")
    c2.download_button("Previsões (CSV)", pd.DataFrame({"dias_reais": r["real"],
                                                        "dias_previstos": r["previsto"].round(4)})
                       .to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig"),
                       f"previsoes_{r['nome']}_{r['escala']}_{momento}.csv", "text/csv", on_click="ignore",
                       icon=":material/download:")
    c3.download_button("Log (TXT)", "\n".join(r["log"]).encode("utf-8"),
                       f"log_{r['nome']}_{r['escala']}_{momento}.txt", "text/plain", on_click="ignore",
                       icon=":material/download:")
    if len(r["registros"]) > 1:
        mostra_tabela(r["registros"])
    aba_resultado, aba_curva, aba_dentro, aba_log = st.tabs(["Erros", "Curva de treino", "Por dentro do modelo",
                                                             "Log"])
    with aba_resultado:
        c1, c2 = st.columns(2)
        c1.markdown("**Previsão média por duração real**")
        c1.altair_chart(grafico_previsto(r["real"], r["previsto"]))
        c1.caption("Durações com 30 ou mais internações na validação. Tracejado: previsto igual ao real.")
        c2.markdown("**MAE por faixa de duração real**")
        c2.altair_chart(grafico_faixas(r["real"], r["previsto"]))
    with aba_curva:
        if r["curva"].empty:
            st.caption("A curva aparece com os parâmetros escolhidos, no Random Forest, no XGBoost e na MLP.")
        else:
            st.altair_chart(grafico_curva(r["curva"], r["eixo"], r["linha"]["parada"]))
            st.caption("Tracejado: ponto escolhido, o de menor MAE." if r["linha"]["parada"] else
                       f"Árvores somadas de {LOTE_DE_ARVORES} em {LOTE_DE_ARVORES}.")
    with aba_dentro:
        mostra_por_dentro(r)
    with aba_log:
        st.code("\n".join(r["log"]), language=None)


st.set_page_config(page_title="Painel dos modelos", layout="wide")
st.title("Painel dos modelos")
st.caption("Tempo de internação nos hospitais de Juiz de Fora (SIH/SUS). Treino: entradas de 2024. Validação: "
           "1º semestre de 2025. O teste não é usado.")
aba_variaveis, aba_modelo, aba_historico = st.tabs(["Variáveis", "Modelo", "Histórico"])

with aba_variaveis:
    mostra_variaveis()

with aba_modelo:
    nome, escala, procedimento, combinacoes = mostra_configuracao()
    rotulo = f"Rodar {len(combinacoes)} combinações" if len(combinacoes) > 1 else "Rodar"
    rodar = st.button(rotulo, type="primary", disabled=not combinacoes, key="rodar",
                      help="Mexer na página durante o treino interrompe a rodada.")
    if rodar:
        try:
            executa(nome, escala, procedimento, combinacoes)
        except (ValueError, TypeError) as erro:
            st.error(f"Não rodou: {erro}")
    if "rodada" in st.session_state:
        mostra_resultado(st.session_state.rodada)

with aba_historico:
    st.subheader("Configuração padrão (5_roda_modelos)")
    mostra_tabela(le_registro(rm.REGISTRO))
    busca = le_registro(aj.REGISTRO)
    if not busca.empty:
        st.subheader("Busca em grade (6_ajusta_modelos)")
        st.caption("Melhor combinação de cada modelo e treino:")
        mostra_tabela(busca.loc[busca.groupby(["modelo", "treino"]).mae.idxmin()])
        with st.expander(f"Todas as {len(busca)} combinações"):
            mostra_tabela(busca)
    st.subheader("Rodadas do painel")
    painel = le_registro(REGISTRO)
    if painel.empty:
        st.caption("Nenhuma rodada ainda.")
    else:
        mostra_tabela(painel)
        st.download_button("Baixar rodadas do painel (CSV)", painel.to_csv(sep=";", decimal=",", index=False)
                           .encode("utf-8-sig"), "painel.csv", "text/csv", on_click="ignore")
