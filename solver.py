import math
import time
import pulp
from data import df_filtrado

# ---------------------------------------------------------
# 1. PREPARAÇÃO DOS DADOS E PARÂMETROS
# ---------------------------------------------------------
NUM_FRENTES = 3           # Quantas unidades expostas lado a lado
CAPACIDADE_GONDOLA = 400  # 4 prateleiras x 100 cm = 400 cm

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

# Checagem: item maior que a gôndola deixa o modelo inviável
for it in itens:
    if it['largura'] > CAPACIDADE_GONDOLA:
        raise ValueError(f"Item maior que a gôndola: {it}")

# Limite superior de gôndolas (bem menor que n_itens)
lb = math.ceil(sum(it['largura'] for it in itens) / CAPACIDADE_GONDOLA)
max_gondolas = min(n_itens, 2 * lb + 1)

print("Iniciando Otimização...")
print(f"Total de itens a alocar: {n_itens}")
print(f"Capacidade de cada gôndola: {CAPACIDADE_GONDOLA} cm")
print(f"Limite inferior de gôndolas: {lb} | Máximo considerado: {max_gondolas}\n")

# ---------------------------------------------------------
# 2. CRIAÇÃO DO MODELO MATEMÁTICO (PuLP / MILP)
# ---------------------------------------------------------
prob = pulp.LpProblem("BinPacking_Gondolas", pulp.LpMinimize)

# y[j] = 1 se a gôndola j é usada
y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(max_gondolas)}
# x[i, j] = 1 se o item i está na gôndola j
x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
     for i in range(n_itens) for j in range(max_gondolas)}
# t[j] = 1 se a gôndola j é de alimentos, 0 se é de limpeza
t = {j: pulp.LpVariable(f"t_{j}", cat="Binary") for j in range(max_gondolas)}

# (a) Função objetivo: minimizar o número de gôndolas
prob += pulp.lpSum(y[j] for j in range(max_gondolas))

# (b) Cada item em exatamente 1 gôndola
for i in range(n_itens):
    prob += pulp.lpSum(x[i, j] for j in range(max_gondolas)) == 1

# (c) Capacidade da gôndola
for j in range(max_gondolas):
    prob += (pulp.lpSum(itens[i]['largura'] * x[i, j] for i in range(n_itens))
             <= CAPACIDADE_GONDOLA * y[j])

# (d) Incompatibilidade sanitária (Alimento vs. Limpeza)
alimentos_idx = [i for i in range(n_itens) if itens[i]['categoria'] == 'Alimento']
limpeza_idx = [i for i in range(n_itens) if itens[i]['categoria'] == 'Limpeza']

for j in range(max_gondolas):
    for i in alimentos_idx:
        prob += x[i, j] <= t[j]
    for i in limpeza_idx:
        prob += x[i, j] <= 1 - t[j]

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
for j in range(max_gondolas):
    if pulp.value(y[j]) > 0.5:
        itens_na_gondola = [itens[i] for i in range(n_itens)
                            if pulp.value(x[i, j]) > 0.5]
        if not itens_na_gondola:
            continue
        gondolas_usadas += 1
        largura_ocupada = sum(item['largura'] for item in itens_na_gondola)
        cat_gondola = itens_na_gondola[0]['categoria']

        print(f"\nGôndola #{gondolas_usadas} [{cat_gondola}] - "
              f"Ocupação: {largura_ocupada:.1f} / {CAPACIDADE_GONDOLA} cm")
        for item in itens_na_gondola:
            print(f"  └─ {item['nome']} (ID {item['id']}): {item['largura']:.1f} cm")