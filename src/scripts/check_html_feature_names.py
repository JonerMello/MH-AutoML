import os
import re

# Encontrar o arquivo HTML mais recente
html_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.html') and 'shap_force_plot' in file:
            html_files.append(os.path.join(root, file))

if html_files:
    # Pegar o arquivo mais recente
    latest_html = max(html_files, key=os.path.getctime)
    print(f"📄 Verificando arquivo: {latest_html}")
    
    # Ler o conteúdo do arquivo HTML
    with open(latest_html, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\n🔍 Análise do arquivo HTML:")
    print(f"   - Tamanho do arquivo: {len(content)} caracteres")
    
    # Procurar por diferentes padrões de feature names
    patterns_to_check = [
        (r'Feature\s+\d+', 'Feature X genérica'),
        (r'feature_\d+', 'feature_X real'),
        (r'feature\d+', 'featureX sem underscore'),
        (r'"feature_0"', 'feature_0 com aspas'),
        (r'"feature_1"', 'feature_1 com aspas'),
        (r'"feature_2"', 'feature_2 com aspas'),
        (r'"feature_3"', 'feature_3 com aspas'),
        (r'"feature_4"', 'feature_4 com aspas'),
    ]
    
    found_patterns = []
    for pattern, description in patterns_to_check:
        matches = re.findall(pattern, content)
        if matches:
            found_patterns.append((description, len(matches), matches[:3]))
    
    if found_patterns:
        print(f"   - Padrões encontrados:")
        for desc, count, examples in found_patterns:
            print(f"     • {desc}: {count} ocorrências - Exemplos: {examples}")
    else:
        print(f"   - Nenhum padrão de feature name encontrado")
    
    # Procurar por dados JSON no HTML que possam conter feature names
    json_patterns = [
        r'\{[^}]*"feature[^}]*\}',
        r'\[[^\]]*"feature[^\]]*\]',
    ]
    
    print(f"\n🔍 Procurando por dados JSON:")
    for pattern in json_patterns:
        matches = re.findall(pattern, content)
        if matches:
            print(f"   - Encontrado JSON com features: {matches[:2]}")
    
    # Verificar se há algum texto legível no HTML
    # Remover tags HTML e procurar por texto
    text_content = re.sub(r'<[^>]+>', ' ', content)
    text_content = re.sub(r'\s+', ' ', text_content)
    
    # Procurar por feature names no texto limpo
    feature_in_text = re.findall(r'feature_\d+', text_content)
    if feature_in_text:
        print(f"   - Features encontradas no texto: {feature_in_text[:5]}")
    else:
        print(f"   - Nenhuma feature encontrada no texto limpo")
        
else:
    print("❌ Nenhum arquivo HTML de SHAP force plot encontrado") 