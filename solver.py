import math
import time
import pulp
from data import df_filtrado

# ---------------------------------------------------------
# 1. PREPARAÇÃO DOS DADOS E PARÂMETROS
# ---------------------------------------------------------
NUM_FRENTES = 3              # Quantas unidades expostas lado a lado
NUM_PRATELEIRAS = 4          # Prateleiras por gôndola
CAPACIDADE_PRATELEIRA = 120  # 120 cm por prateleira
CAPACIDADE_GONDOLA = NUM_PRATELEIRAS * CAPACIDADE_PRATELEIRA  # 480 cm no total

itens = []
for idx, row in df_filtrado.reset_index(drop=True).iterrows():
    largura_total = row['Largura_cm'] * NUM_FRENTES
    itens.append({
        'id': row['Aisle ID'],
        'nome': row['Product family'],
        'categoria': row['Categoria'],
        'largura': largura_total
    })

n_itens = len(itens)

# Checagem: item maior que a prateleira deixa o modelo inviável
for it in itens:
    if it['largura'] > CAPACIDADE_PRATELEIRA:
        raise ValueError(f"Item maior que a prateleira ({CAPACIDADE_PRATELEIRA} cm): {it}")

# Limite superior de gôndolas
lb = math.ceil(sum(it['largura'] for it in itens) / CAPACIDADE_GONDOLA)
max_gondolas = min(n_itens, 2 * lb + 2)

print("Iniciando Otimização...")
print(f"Total de itens a alocar: {n_itens}")
print(f"Estrutura da gôndola: {NUM_PRATELEIRAS} prateleiras x {CAPACIDADE_PRATELEIRA} cm = {CAPACIDADE_GONDOLA} cm")
print(f"Limite inferior de gôndolas: {lb} | Máximo considerado: {max_gondolas}\n")

# ---------------------------------------------------------
# 2. CRIAÇÃO DO MODELO MATEMÁTICO (PuLP / MILP)
# ---------------------------------------------------------
prob = pulp.LpProblem("BinPacking_Gondolas_Prateleiras", pulp.LpMinimize)

G = range(max_gondolas)
P = range(NUM_PRATELEIRAS)

# y[j] = 1 se a gôndola j é usada
y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in G}
# x[i, j, p] = 1 se o item i está na prateleira p da gôndola j
x = {(i, j, p): pulp.LpVariable(f"x_{i}_{j}_{p}", cat="Binary")
     for i in range(n_itens) for j in G for p in P}
# t[j] = 1 se a gôndola j permite Alimentos, 0 se permite Limpeza
# (Higiene Pessoal é compatível com ambos e não sofre restrição de t[j])
t = {j: pulp.LpVariable(f"t_{j}", cat="Binary") for j in G}

# (a) Função objetivo: minimizar o número de gôndolas
prob += pulp.lpSum(y[j] for j in G)

# (b) Cada item em exatamente 1 prateleira de 1 gôndola
for i in range(n_itens):
    prob += pulp.lpSum(x[i, j, p] for j in G for p in P) == 1

# (c) Capacidade de cada prateleira p da gôndola j
for j in G:
    for p in P:
        prob += (pulp.lpSum(itens[i]['largura'] * x[i, j, p] for i in range(n_itens))
                 <= CAPACIDADE_PRATELEIRA * y[j])

# (d) Incompatibilidade sanitária na gôndola (Alimento vs. Limpeza; Higiene Pessoal é livre)
alimentos_idx = [i for i in range(n_itens) if itens[i]['categoria'] == 'Alimento']
limpeza_idx = [i for i in range(n_itens) if itens[i]['categoria'] == 'Limpeza']
higiene_idx = [i for i in range(n_itens) if itens[i]['categoria'] == 'Higiene Pessoal']

for j in G:
    for i in alimentos_idx:
        prob += pulp.lpSum(x[i, j, p] for p in P) <= t[j]
    for i in limpeza_idx:
        prob += pulp.lpSum(x[i, j, p] for p in P) <= 1 - t[j]

# (e) Quebra de simetria: usa as gôndolas em ordem
for j in range(max_gondolas - 1):
    prob += y[j] >= y[j + 1]

# ---------------------------------------------------------
# 3. RESOLUÇÃO E MENSURAÇÃO DE DESEMPENHO
# ---------------------------------------------------------
inicio = time.time()
prob.solve(pulp.PULP_CBC_CMD(msg=False))
fim = time.time()
tempo_execucao = fim - inicio

# ---------------------------------------------------------
# 4. EXIBIÇÃO DOS RESULTADOS
# ---------------------------------------------------------
status = pulp.LpStatus[prob.status]

print("=" * 50)
print("             RESULTADOS DA OTIMIZAÇÃO")
print("=" * 50)
print(f"Status da Solução     : {status}")
if status != "Optimal":
    print("Sem solução ótima. Tente aumentar max_gondolas.")
    raise SystemExit(1)

print(f"Gôndolas Utilizadas   : {round(pulp.value(prob.objective))}")
print(f"Tempo de Processamento: {tempo_execucao:.4f} segundos")
print("=" * 50)

gondolas_usadas = 0
for j in G:
    if pulp.value(y[j]) > 0.5:
        itens_na_gondola = [itens[i] for i in range(n_itens) for p in P
                            if pulp.value(x[i, j, p]) > 0.5]
        if not itens_na_gondola:
            continue
        gondolas_usadas += 1
        largura_gondola = sum(item['largura'] for item in itens_na_gondola)
        cat_gondola = " + ".join(sorted({item['categoria'] for item in itens_na_gondola}))

        print(f"\nGôndola #{gondolas_usadas} [{cat_gondola}] - "
              f"Ocupação Total: {largura_gondola:.1f} / {CAPACIDADE_GONDOLA} cm")
        for p in P:
            itens_prat = [itens[i] for i in range(n_itens) if pulp.value(x[i, j, p]) > 0.5]
            ocup_prat = sum(it['largura'] for it in itens_prat)
            print(f"  Prateleira {p + 1} ({ocup_prat:.1f} / {CAPACIDADE_PRATELEIRA} cm):")
            for item in itens_prat:
                print(f"    └─ {item['nome']} (ID {item['id']}, {item['categoria']}): {item['largura']:.1f} cm")