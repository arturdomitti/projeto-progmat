"""
solvers_comparativo.py
Comparação Gurobi (comercial) x SCIP (livre) para o Bin Packing de gôndolas
com incompatibilidade sanitária (Alimento x Limpeza).

Formulação (idêntica nos dois solvers):
    min   sum_j y_j
    s.a.  sum_j x_ij = 1                         para todo i      (cobertura)
          sum_i w_i x_ij <= C y_j                para todo j      (capacidade)
          x_ij <= t_j        (i Alimento)        para todo i, j   (sanitária)
          x_ij <= 1 - t_j    (i Limpeza)         para todo i, j   (sanitária)
          y_j >= y_{j+1}                         para todo j      (quebra de simetria)
          x, y, t binárias

Obs.: como Alimento e Limpeza nunca dividem gôndola, o problema na verdade
decompõe em dois BPPs independentes. Isso dá um limite inferior (LB) e um
limite superior (UB, via FFD) válidos por categoria, usados abaixo.
"""

import math
import os
import random
import time
from collections import defaultdict

import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pyscipopt import Model as SCIPModel, quicksum

from data import df_filtrado

# ----------------------------------------------------------------
# 0. CONFIGURAÇÃO
# ----------------------------------------------------------------
CAPACIDADE_GONDOLA = 400      # 4 prateleiras x 100 cm
TEMPO_LIMITE = 60             # segundos, igual para os dois solvers
SEMENTE = 42                  # reprodutibilidade dos subconjuntos
CAT_ALIMENTO = "Alimento"
CAT_LIMPEZA = "Limpeza"

# (número de itens, NUM_FRENTES). Mais frentes = itens mais largos = problema mais difícil.
INSTANCIAS = [(20, 3), (40, 3), (55, 3), (55, 5), (55, 8)]
DETALHAR = (55, 3)            # instância cuja alocação é impressa por extenso

LIMITE_LICENCA_GUROBI = 2000  # variáveis e restrições (licença restrita do pip)
PASTA_INSTANCIAS = "instancias"
ARQUIVO_CSV = "resultados_solvers.csv"


# ----------------------------------------------------------------
# 1. DADOS E INSTÂNCIAS
# ----------------------------------------------------------------
def carregar_itens(df):
    itens = []
    for _, row in df.reset_index(drop=True).iterrows():
        itens.append({
            "id": int(row["Aisle ID"]),
            "nome": row["Product family"],
            "categoria": row["Categoria"],
            "largura_un": float(row["Largura_cm"]),
        })
    return itens


def montar_instancia(base, n, frentes, semente=SEMENTE):
    """Subconjunto estratificado (mantém a proporção Alimento/Limpeza)."""
    if n >= len(base):
        sel = list(base)
    else:
        alim = [it for it in base if it["categoria"] == CAT_ALIMENTO]
        limp = [it for it in base if it["categoria"] == CAT_LIMPEZA]
        k_l = max(1, round(n * len(limp) / len(base)))
        k_a = n - k_l
        rng = random.Random(semente)
        sel = rng.sample(alim, k_a) + rng.sample(limp, k_l)

    itens = [{**it, "largura": it["largura_un"] * frentes} for it in sel]

    for it in itens:
        if it["categoria"] not in (CAT_ALIMENTO, CAT_LIMPEZA):
            raise ValueError(f"Categoria inesperada: {it}")
        if it["largura"] > CAPACIDADE_GONDOLA:
            raise ValueError(f"Item maior que a gôndola: {it}")
    return itens


def salvar_instancia(itens, nome):
    os.makedirs(PASTA_INSTANCIAS, exist_ok=True)
    pd.DataFrame(itens).to_csv(
        os.path.join(PASTA_INSTANCIAS, f"{nome}.csv"), index=False, encoding="utf-8-sig"
    )


# ----------------------------------------------------------------
# 2. LIMITES (LB combinatório e UB via First-Fit Decreasing)
# ----------------------------------------------------------------
def ffd(itens, idxs):
    """Número de gôndolas usado pelo First-Fit Decreasing nos itens idxs."""
    folgas = []
    for i in sorted(idxs, key=lambda k: -itens[k]["largura"]):
        w = itens[i]["largura"]
        for b in range(len(folgas)):
            if folgas[b] >= w - 1e-9:
                folgas[b] -= w
                break
        else:
            folgas.append(CAPACIDADE_GONDOLA - w)
    return len(folgas)


def limites(itens):
    lb = ub = 0
    for cat in (CAT_ALIMENTO, CAT_LIMPEZA):
        idxs = [i for i, it in enumerate(itens) if it["categoria"] == cat]
        if not idxs:
            continue
        soma = sum(itens[i]["largura"] for i in idxs)
        lb += math.ceil(soma / CAPACIDADE_GONDOLA - 1e-9)
        ub += ffd(itens, idxs)
    return lb, ub


def tamanho_modelo(n, M):
    n_vars = n * M + 2 * M
    n_cons = n + M + n * M + (M - 1)
    return n_vars, n_cons


# ----------------------------------------------------------------
# 3. SOLVERS
# ----------------------------------------------------------------
def resultado(status, obj=None, bound=None, nos=None, tempo=None, alocacao=None):
    return {"status": status, "obj": obj, "bound": bound,
            "nos": nos, "tempo": tempo, "alocacao": alocacao}


def indices_por_categoria(itens):
    I = range(len(itens))
    A = [i for i in I if itens[i]["categoria"] == CAT_ALIMENTO]
    L = [i for i in I if itens[i]["categoria"] == CAT_LIMPEZA]
    return A, L


def resolver_gurobi(itens, M, env):
    n = len(itens)
    I, J = range(n), range(M)
    A, L = indices_por_categoria(itens)

    m = gp.Model("BPP_Gondolas_Gurobi", env=env)
    m.setParam("TimeLimit", TEMPO_LIMITE)
    m.setParam("Threads", 1)
    m.setParam("MIPGap", 0.0)

    y = m.addVars(J, vtype=GRB.BINARY, name="y")
    t = m.addVars(J, vtype=GRB.BINARY, name="t")        # 1 = gôndola de alimentos
    x = m.addVars(I, J, vtype=GRB.BINARY, name="x")

    m.setObjective(gp.quicksum(y[j] for j in J), GRB.MINIMIZE)

    m.addConstrs((gp.quicksum(x[i, j] for j in J) == 1 for i in I), name="cobertura")
    m.addConstrs(
        (gp.quicksum(itens[i]["largura"] * x[i, j] for i in I) <= CAPACIDADE_GONDOLA * y[j]
         for j in J), name="capacidade")
    m.addConstrs((x[i, j] <= t[j] for i in A for j in J), name="sanitaria_alim")
    m.addConstrs((x[i, j] <= 1 - t[j] for i in L for j in J), name="sanitaria_limp")
    m.addConstrs((y[j] >= y[j + 1] for j in range(M - 1)), name="simetria")

    inicio = time.perf_counter()
    try:
        m.optimize()
    except gp.GurobiError as e:
        return resultado(f"Erro Gurobi: {e}")
    tempo = time.perf_counter() - inicio

    tem_sol = m.SolCount > 0
    if m.Status == GRB.OPTIMAL:
        status = "Ótima"
    elif m.Status == GRB.TIME_LIMIT:
        status = "Factível (timeout)" if tem_sol else "Timeout sem solução"
    elif m.Status == GRB.INFEASIBLE:
        status = "Infactível"
    else:
        status = f"Outro (código {m.Status})"

    obj = bound = alocacao = None
    if tem_sol:
        obj = m.ObjVal
        alocacao = {i: [j for j in J if x[i, j].X > 0.5] for i in I}
    if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        bound = m.ObjBound

    return resultado(status, obj, bound, int(m.NodeCount), tempo, alocacao)


def resolver_scip(itens, M):
    n = len(itens)
    I, J = range(n), range(M)
    A, L = indices_por_categoria(itens)

    m = SCIPModel("BPP_Gondolas_SCIP")
    m.hideOutput()
    m.setParam("limits/time", float(TEMPO_LIMITE))
    m.setParam("limits/gap", 0.0)

    y = {j: m.addVar(vtype="B", name=f"y_{j}") for j in J}
    t = {j: m.addVar(vtype="B", name=f"t_{j}") for j in J}        # 1 = gôndola de alimentos
    x = {(i, j): m.addVar(vtype="B", name=f"x_{i}_{j}") for i in I for j in J}

    m.setObjective(quicksum(y[j] for j in J), "minimize")

    for i in I:
        m.addCons(quicksum(x[i, j] for j in J) == 1)
    for j in J:
        m.addCons(quicksum(itens[i]["largura"] * x[i, j] for i in I)
                  <= CAPACIDADE_GONDOLA * y[j])
    for j in J:
        for i in A:
            m.addCons(x[i, j] <= t[j])
        for i in L:
            m.addCons(x[i, j] <= 1 - t[j])
    for j in range(M - 1):
        m.addCons(y[j] >= y[j + 1])

    inicio = time.perf_counter()
    m.optimize()
    tempo = time.perf_counter() - inicio

    st = m.getStatus()
    tem_sol = m.getNSols() > 0
    if st == "optimal":
        status = "Ótima"
    elif st == "timelimit":
        status = "Factível (timeout)" if tem_sol else "Timeout sem solução"
    elif st == "infeasible":
        status = "Infactível"
    else:
        status = f"Outro ({st})"

    obj = bound = alocacao = None
    if tem_sol:
        sol = m.getBestSol()
        obj = m.getSolObjVal(sol)
        alocacao = {i: [j for j in J if m.getSolVal(sol, x[i, j]) > 0.5] for i in I}
    if st in ("optimal", "timelimit"):
        bound = m.getDualbound()

    return resultado(status, obj, bound, int(m.getNNodes()), tempo, alocacao)


# ----------------------------------------------------------------
# 4. VALIDAÇÃO E RELATÓRIO
# ----------------------------------------------------------------
def validar_solucao(itens, alocacao):
    """Confere a solução de forma independente do solver.
    Retorna (ok, número de gôndolas usadas)."""
    por_gondola = defaultdict(list)
    for i in range(len(itens)):
        gs = alocacao.get(i, [])
        if len(gs) != 1:
            return False, None
        por_gondola[gs[0]].append(i)
    for idxs in por_gondola.values():
        if sum(itens[i]["largura"] for i in idxs) > CAPACIDADE_GONDOLA + 1e-6:
            return False, len(por_gondola)
        if len({itens[i]["categoria"] for i in idxs}) > 1:
            return False, len(por_gondola)
    return True, len(por_gondola)


def calcular_gap(obj, bound):
    """Gap (%) = (melhor inteira - bound) / melhor inteira, igual para os dois solvers."""
    if obj is None or bound is None:
        return None
    if abs(obj) < 1e-9:
        return 0.0
    return max(0.0, (obj - bound) / abs(obj) * 100)


def imprimir_alocacao(itens, alocacao):
    por_gondola = defaultdict(list)
    for i, gs in alocacao.items():
        por_gondola[gs[0]].append(i)
    for k, j in enumerate(sorted(por_gondola), start=1):
        idxs = por_gondola[j]
        ocup = sum(itens[i]["largura"] for i in idxs)
        cat = itens[idxs[0]]["categoria"]
        print(f"\n  Gôndola #{k} [{cat}] - Ocupação: {ocup:.1f} / {CAPACIDADE_GONDOLA} cm")
        for i in idxs:
            print(f"    └─ {itens[i]['nome']} (ID {itens[i]['id']}): {itens[i]['largura']:.1f} cm")


def montar_linha(nome_inst, n, frentes, solver, res, itens, lb, ub):
    valida = None
    if res["alocacao"] is not None:
        ok, usadas = validar_solucao(itens, res["alocacao"])
        valida = ok and usadas == round(res["obj"])
    gap = calcular_gap(res["obj"], res["bound"])
    return {
        "Instância": nome_inst,
        "Itens": n,
        "Frentes": frentes,
        "Solver": solver,
        "Status": res["status"],
        "Gôndolas": None if res["obj"] is None else round(res["obj"]),
        "Bound": None if res["bound"] is None else round(res["bound"], 3),
        "Gap (%)": None if gap is None else round(gap, 4),
        "Tempo (s)": None if res["tempo"] is None else round(res["tempo"], 4),
        "Nós": res["nos"],
        "LB": lb,
        "UB (FFD)": ub,
        "Solução válida": valida,
    }


# ----------------------------------------------------------------
# 5. EXECUÇÃO
# ----------------------------------------------------------------
def main():
    base = carregar_itens(df_filtrado)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    linhas = []
    for n, frentes in INSTANCIAS:
        nome_inst = f"I{n}_F{frentes}"
        itens = montar_instancia(base, n, frentes)
        salvar_instancia(itens, nome_inst)

        lb, ub = limites(itens)
        M = ub                       # FFD garante que ub gôndolas bastam
        n_vars, n_cons = tamanho_modelo(len(itens), M)
        n_alim, n_limp = (len(v) for v in indices_por_categoria(itens))

        print(f"\n=== {nome_inst}: {len(itens)} itens "
              f"({n_alim} alimentos, {n_limp} limpeza) | LB={lb} UB(FFD)={ub} "
              f"| {n_vars} vars, {n_cons} restrições ===", flush=True)

        # Gurobi
        if n_vars > LIMITE_LICENCA_GUROBI or n_cons > LIMITE_LICENCA_GUROBI:
            res_g = resultado("Excede licença Gurobi")
        else:
            print("  Gurobi...", flush=True)
            res_g = resolver_gurobi(itens, M, env)
        linhas.append(montar_linha(nome_inst, len(itens), frentes, "Gurobi", res_g, itens, lb, ub))

        # SCIP
        print("  SCIP...", flush=True)
        res_s = resolver_scip(itens, M)
        linhas.append(montar_linha(nome_inst, len(itens), frentes, "SCIP", res_s, itens, lb, ub))

        if (n, frentes) == DETALHAR and res_g["alocacao"] is not None:
            print("\n  --- Alocação (Gurobi) ---")
            imprimir_alocacao(itens, res_g["alocacao"])

    df = pd.DataFrame(linhas)
    df.to_csv(ARQUIVO_CSV, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 100)
    print("TABELA COMPARATIVA DE SOLVERS")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"Resultados salvos em {ARQUIVO_CSV}; instâncias em ./{PASTA_INSTANCIAS}/")


if __name__ == "__main__":
    main()