"""
solvers_comparativo.py
Comparação Gurobi (comercial) x SCIP (livre) para o Bin Packing de gôndolas
com incompatibilidade sanitária (Alimento x Limpeza) e 3º tipo compatível
com ambos (Higiene Pessoal).

Formulação (idêntica nos dois solvers):
    min   sum_j y_j
    s.a.  sum_j x_ij = 1                         para todo i      (cobertura)
          sum_i w_i x_ij <= C y_j                para todo j      (capacidade)
          x_ij <= t_j        (i Alimento)        para todo i, j   (sanitária)
          x_ij <= 1 - t_j    (i Limpeza)         para todo i, j   (sanitária)
          y_j >= y_{j+1}                         para todo j      (quebra de simetria)
          x, y, t binárias
    (Itens de Higiene Pessoal não têm restrição com t_j, podendo dividir
     gôndola tanto com Alimento quanto com Limpeza.)
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
CAT_HIGIENE = "Higiene Pessoal"
CATEGORIAS_VALIDAS = (CAT_ALIMENTO, CAT_LIMPEZA, CAT_HIGIENE)

# (número de itens, NUM_FRENTES). Mais frentes = itens mais largos = problema mais difícil.
INSTANCIAS = [(20, 3), (40, 3), (65, 3), (65, 5), (65, 8)]
DETALHAR = (65, 3)            # instância cuja alocação é impressa por extenso

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
    """Subconjunto estratificado (mantém a proporção Alimento/Limpeza/Higiene Pessoal)."""
    if n >= len(base):
        sel = list(base)
    else:
        alim = [it for it in base if it["categoria"] == CAT_ALIMENTO]
        limp = [it for it in base if it["categoria"] == CAT_LIMPEZA]
        hig = [it for it in base if it["categoria"] == CAT_HIGIENE]
        k_l = max(1, round(n * len(limp) / len(base)))
        k_h = max(1, round(n * len(hig) / len(base)))
        k_a = n - k_l - k_h
        rng = random.Random(semente)
        sel = rng.sample(alim, k_a) + rng.sample(limp, k_l) + rng.sample(hig, k_h)

    itens = [{**it, "largura": it["largura_un"] * frentes} for it in sel]

    for it in itens:
        if it["categoria"] not in CATEGORIAS_VALIDAS:
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
def ffd_folgas(itens, idxs, folgas_iniciais=None):
    """Aplica First-Fit Decreasing nos itens idxs aproveitando folgas existentes."""
    folgas = list(folgas_iniciais) if folgas_iniciais is not None else []
    for i in sorted(idxs, key=lambda k: -itens[k]["largura"]):
        w = itens[i]["largura"]
        for b in range(len(folgas)):
            if folgas[b] >= w - 1e-9:
                folgas[b] -= w
                break
        else:
            folgas.append(CAPACIDADE_GONDOLA - w)
    return folgas


def limites(itens):
    A, L, H = indices_por_categoria(itens)
    soma_a = sum(itens[i]["largura"] for i in A)
    soma_l = sum(itens[i]["largura"] for i in L)
    soma_total = soma_a + soma_l + sum(itens[i]["largura"] for i in H)

    lb_a = math.ceil(soma_a / CAPACIDADE_GONDOLA - 1e-9) if A else 0
    lb_l = math.ceil(soma_l / CAPACIDADE_GONDOLA - 1e-9) if L else 0
    lb_tot = math.ceil(soma_total / CAPACIDADE_GONDOLA - 1e-9) if itens else 0
    lb = max(lb_a + lb_l, lb_tot)

    # FFD: empacota A e L separadamente, depois encaixa H nas folgas de ambos
    folgas_a = ffd_folgas(itens, A)
    folgas_l = ffd_folgas(itens, L)
    folgas_finais = ffd_folgas(itens, H, folgas_a + folgas_l)
    ub = len(folgas_finais)
    return lb, ub


def tamanho_modelo(itens, M):
    n = len(itens)
    A, L, _ = indices_por_categoria(itens)
    n_vars = n * M + 2 * M
    n_cons = n + M + (len(A) + len(L)) * M + (M - 1)
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
    H = [i for i in I if itens[i]["categoria"] == CAT_HIGIENE]
    return A, L, H


def resolver_gurobi(itens, M, env):
    n = len(itens)
    I, J = range(n), range(M)
    A, L, _ = indices_por_categoria(itens)

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
    A, L, _ = indices_por_categoria(itens)

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
        cats = {itens[i]["categoria"] for i in idxs}
        if CAT_ALIMENTO in cats and CAT_LIMPEZA in cats:
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
        cat = " + ".join(sorted({itens[i]["categoria"] for i in idxs}))
        print(f"\n  Gôndola #{k} [{cat}] - Ocupação: {ocup:.1f} / {CAPACIDADE_GONDOLA} cm")
        for i in idxs:
            print(f"    └─ {itens[i]['nome']} (ID {itens[i]['id']}, {itens[i]['categoria']}): {itens[i]['largura']:.1f} cm")


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
NUM_EXECUCOES = 10


def executar_media(fn_solver, n_exec=NUM_EXECUCOES):
    """Roda fn_solver() n_exec vezes e retorna o resultado com tempo e nós médios.
    Interrompe as repetições se a 1ª execução atingir o limite de tempo (timeout)."""
    execucoes = []
    for _ in range(n_exec):
        r = fn_solver()
        execucoes.append(r)
        if "timeout" in r["status"].lower():
            print("    [Timeout atingido na 1ª rodada; pulando repetições restantes]", flush=True)
            break
    res = dict(execucoes[-1])
    tempos = [r["tempo"] for r in execucoes if r["tempo"] is not None]
    nos = [r["nos"] for r in execucoes if r["nos"] is not None]
    if tempos:
        res["tempo"] = sum(tempos) / len(tempos)
    if nos:
        res["nos"] = round(sum(nos) / len(nos))
    return res


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
        n_vars, n_cons = tamanho_modelo(itens, M)
        n_alim, n_limp, n_hig = (len(v) for v in indices_por_categoria(itens))

        print(f"\n=== {nome_inst}: {len(itens)} itens "
              f"({n_alim} alimentos, {n_limp} limpeza, {n_hig} higiene) | LB={lb} UB(FFD)={ub} "
              f"| {n_vars} vars, {n_cons} restrições ===", flush=True)

        # Gurobi (média de NUM_EXECUCOES rodadas)
        if n_vars > LIMITE_LICENCA_GUROBI or n_cons > LIMITE_LICENCA_GUROBI:
            res_g = resultado("Excede licença Gurobi")
        else:
            print(f"  Gurobi ({NUM_EXECUCOES}x)...", flush=True)
            res_g = executar_media(lambda: resolver_gurobi(itens, M, env))
            print(f"    -> Média Gurobi: {res_g['tempo']:.4f} s | Nós: {res_g['nos']}", flush=True)
        linhas.append(montar_linha(nome_inst, len(itens), frentes, "Gurobi", res_g, itens, lb, ub))

        # SCIP (média de NUM_EXECUCOES rodadas)
        print(f"  SCIP ({NUM_EXECUCOES}x)...", flush=True)
        res_s = executar_media(lambda: resolver_scip(itens, M))
        print(f"    -> Média SCIP  : {res_s['tempo']:.4f} s | Nós: {res_s['nos']}", flush=True)
        linhas.append(montar_linha(nome_inst, len(itens), frentes, "SCIP", res_s, itens, lb, ub))

        if (n, frentes) == DETALHAR and res_g["alocacao"] is not None:
            print("\n  --- Alocação (Gurobi) ---")
            imprimir_alocacao(itens, res_g["alocacao"])

    df = pd.DataFrame(linhas)
    df.to_csv(ARQUIVO_CSV, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 100)
    print(f"TABELA COMPARATIVA DE SOLVERS (MÉDIA DE {NUM_EXECUCOES} EXECUÇÕES)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    print(f"Resultados salvos em {ARQUIVO_CSV}; instâncias em ./{PASTA_INSTANCIAS}/")


if __name__ == "__main__":
    main()