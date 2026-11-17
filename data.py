import pandas as pd

# 1. Carregar o CSV
df = pd.read_csv('Product Information.csv')

# 2. Listas de palavras-chave para classificação
palavras_limpeza = [
    'cleaning', 'laundry', 'detergent', 'household', 'soap', 
    'paper goods', 'dish', 'trash', 'cleaner', 'air fresheners', 'dishwashing'
]

palavras_alimentos = [
    'fruit', 'vegetable', 'snack', 'beverage', 'cereal', 'frozen', 
    'bakery', 'canned', 'dairy', 'meat', 'pantry', 'pasta', 'juice', 
    'coffee', 'tea', 'sauce', 'spices', 'breakfast', 'soups', 'cheese', 
    'granola', 'energy', 'candy', 'chocolate'
]

palavras_higiene = [
    'oral hygiene', 'hair care', 'shave', 'diapers', 'facial care',
    'deodorants', 'body care', 'skin care', 'feminine care', 'beauty'
]

def categorizar_produto(nome_familia):
    nome_lower = str(nome_familia).lower()
    
    if any(p in nome_lower for p in palavras_limpeza):
        return 'Limpeza'
    elif any(p in nome_lower for p in palavras_alimentos):
        return 'Alimento'
    elif any(p in nome_lower for p in palavras_higiene):
        return 'Higiene Pessoal'
    else:
        return 'Outros'

# 3. Aplicar categorização usando o nome exato da coluna 'Product family'
df['Categoria'] = df['Product family'].apply(categorizar_produto)

# 4. Converter largura média de polegadas (in) para centímetros (cm)
# Nota: 'Avg width  (in)' tem dois espaços no cabeçalho do CSV original
coluna_largura = [col for col in df.columns if 'Avg width' in col][0]
df['Largura_cm'] = df[coluna_largura] * 2.54

# 5. Filtrar Alimentos, Limpeza e Higiene Pessoal
df_filtrado = df[df['Categoria'].isin(['Alimento', 'Limpeza', 'Higiene Pessoal'])].copy()

print("--- RESUMO DOS DADOS PROCESSADOS ---")
print(f"Total de famílias de Alimentos: {len(df_filtrado[df_filtrado['Categoria'] == 'Alimento'])}")
print(f"Total de famílias de Limpeza: {len(df_filtrado[df_filtrado['Categoria'] == 'Limpeza'])}")
print(f"Total de famílias de Higiene Pessoal: {len(df_filtrado[df_filtrado['Categoria'] == 'Higiene Pessoal'])}")
print("\nPrimeiros 5 produtos categorizados:")
print(df_filtrado[['Aisle ID', 'Product family', 'Categoria', 'Largura_cm']].head())