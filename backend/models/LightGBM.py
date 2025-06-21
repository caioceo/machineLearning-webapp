import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelagem e Pré-processamento
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc)
from lightgbm import LGBMClassifier

# Bibliotecas para balanceamento
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Pipeline que suporta o SMOTE

# --- 1. Configurações e Carregamento dos Dados ---
pd.set_option('display.max_columns', None)
# Usando um estilo de gráfico mais sóbrio
plt.style.use('seaborn-v0_8-whitegrid')

# NOTA: O arquivo foi ajustado para 'dados_tratados_combinados.csv'.
# Altere se o nome do seu arquivo for 'dados_tratados_combinados (2).csv'.
try:
    df = pd.read_csv('dados_tratados_combinados.csv')
    print("Arquivo 'dados_tratados_combinados.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Arquivo 'dados_tratados_combinados.csv' não encontrado.")
    exit()
    
# --- 2. Definição da Variável Alvo: Satisfação Profissional ---
target_col_original = 'você_está_satisfeito_na_sua_empresa_atual?'
target_col_final = 'target_satisfeito'
df.dropna(subset=[target_col_original], inplace=True)
df[target_col_final] = df[target_col_original].astype(int)
print("\nDistribuição da variável alvo (Satisfação):")
print(df[target_col_final].value_counts(normalize=True))

# --- 3. Engenharia e Seleção de Features ---
categorical_features = [
    'nivel', 'faixa_salarial', 'setor', 'numero_de_funcionarios',
    'atualmente_qual_a_sua_forma_de_trabalho?'
]
experience_col = 'quanto_tempo_de_experiência_na_área_de_dados_você_tem?'
exp_map = {
    'Não tenho experiência na área de dados': 0, 'Menos de 1 ano': 0.5, 'de 1 a 2 anos': 1.5,
    'de 3 a 4 anos': 3.5, 'de 5 a 6 anos': 5.5, 'de 7 a 10 anos': 8.5, 'Mais de 10 anos': 12
}
df['experiencia_anos'] = df[experience_col].map(exp_map).fillna(0)
motivos_features = [
    'falta_de_oportunidade_de_crescimento_no_emprego_atual',
    'salário_atual_não_corresponde_ao_mercado',
    'não_tenho_uma_boa_relação_com_meu_líder_gestor',
    'gostaria_de_receber_mais_benefícios',
    'o_clima_de_trabalho_ambiente_não_é_bom',
    'falta_de_maturidade_analítica_na_empresa'
]
numeric_features = ['experiencia_anos'] + motivos_features
for col in numeric_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
final_cat_features = [col for col in categorical_features if col in df.columns]
final_num_features = [col for col in numeric_features if col in df.columns]
final_features_list = final_cat_features + final_num_features
print("\nFeatures selecionadas para o modelo:", final_features_list)

# --- 4. Pré-processamento e Pipeline ---
for col in final_cat_features:
    df[col] = df[col].fillna('Não Informado')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), final_cat_features),
        ('num', StandardScaler(), final_num_features)
    ],
    remainder='drop'
)
X = df[final_features_list]
y = df[target_col_final]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LGBMClassifier(random_state=42, n_jobs=-1))
])

# --- 5. Treinamento e Otimização do Modelo ---
print("\nIniciando treinamento com LightGBM, SMOTE e busca de hiperparâmetros...")
params = {
    'classifier__n_estimators': [100, 200, 400],
    'classifier__learning_rate': [0.02, 0.05, 0.1],
    'classifier__num_leaves': [31, 40, 50],
    'classifier__max_depth': [7, 10, 15],
    'classifier__reg_alpha': [0.1, 0.5],
    'classifier__reg_lambda': [0.1, 0.5]
}
search = RandomizedSearchCV(
    pipeline, params, n_iter=30, cv=5,
    scoring='f1_macro', verbose=1, n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

# --- 6. Avaliação e Visualização dos Resultados ---

# Predições para ambos os conjuntos
y_pred_train = best_model.predict(X_train)
y_proba_train = best_model.predict_proba(X_train)[:, 1]
y_pred_test = best_model.predict(X_test)
y_proba_test = best_model.predict_proba(X_test)[:, 1]

print("\n" + "="*60)
print("AVALIAÇÃO COMPARATIVA: TREINO VS. TESTE")
print("="*60)
print("\nMelhores parâmetros encontrados:")
print(search.best_params_)

# Relatórios de Classificação
print("\n--- Relatório de Classificação (TREINO) ---")
print(f"AUC-ROC no Treino: {roc_auc_score(y_train, y_proba_train):.3f}")
print(classification_report(y_train, y_pred_train, target_names=['Insatisfeito', 'Satisfeito']))

print("\n--- Relatório de Classificação (TESTE) ---")
print(f"AUC-ROC no Teste: {roc_auc_score(y_test, y_proba_test):.3f}")
print(classification_report(y_test, y_pred_test, target_names=['Insatisfeito', 'Satisfeito']))


# --- 6.1 Matrizes de Confusão Comparativas (Treino vs. Teste) ---
# Criando uma figura com dois subplots, lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Matrizes de Confusão Comparativas', fontsize=16)

# Matriz de Confusão (Treino)
ConfusionMatrixDisplay.from_estimator(best_model, X_train, y_train,
                                      cmap='Greens', ax=axes[0],
                                      display_labels=['Insatisfeito', 'Satisfeito'])
axes[0].set_title('Conjunto de Treino')

# Matriz de Confusão (Teste)
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test,
                                      cmap='Blues', ax=axes[1],
                                      display_labels=['Insatisfeito', 'Satisfeito'])
axes[1].set_title('Conjunto de Teste')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajusta o layout para o supertítulo
plt.show()


# --- 6.2 Curva de Aprendizagem ---
print("\n" + "="*60)
print("GERANDO CURVA DE APRENDIZAGEM")
print("="*60)
train_sizes, train_scores, test_scores = learning_curve(
    estimator=best_model, X=X, y=y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_macro'
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(12, 7))
plt.title("Curva de Aprendizagem do Modelo")
plt.xlabel("Tamanho do Conjunto de Treinamento")
plt.ylabel("Pontuação F1-Macro")
# Usando cores mais sóbrias
plt.plot(train_sizes, train_scores_mean, 'o-', color="darkorange", label="Score de Treino")
plt.plot(train_sizes, test_scores_mean, 'o-', color="steelblue", label="Score de Validação Cruzada")
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="steelblue")
plt.legend(loc="best")
plt.show()


# --- 6.3 Curva ROC Comparativa ---
print("\n" + "="*60)
print("GERANDO CURVA ROC COMPARATIVA")
print("="*60)
fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
roc_auc_test = auc(fpr_test, tpr_test)
fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
roc_auc_train = auc(fpr_train, tpr_train)

plt.figure(figsize=(11, 8))
plt.title('Curva ROC Comparativa: Treino vs. Teste', fontsize=14)
# Usando cores mais sóbrias
plt.plot(fpr_train, tpr_train, color='steelblue', lw=2, linestyle='--', label=f'Curva ROC Treino (AUC = {roc_auc_train:.3f})')
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Curva ROC Teste (AUC = {roc_auc_test:.3f})')
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle=':', label='Classificador Aleatório')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
plt.legend(loc="lower right", fontsize=11)
plt.show()


# --- 7. Importância das Features ---
print("\n" + "="*60)
print("GERANDO GRÁFICO DE IMPORTÂNCIA DAS FEATURES")
print("="*60)
try:
    cat_features_out = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(final_cat_features)
    all_feature_names = np.concatenate([cat_features_out, final_num_features])
    importances = best_model.named_steps['classifier'].feature_importances_
    feature_importances = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 10))
    # Usando uma paleta de cores mais sóbria (cividis)
    sns.barplot(data=feature_importances.head(20), x='Importance', y='Feature', palette='cividis')
    plt.title('Top 20 Features Mais Importantes (Modelo Otimizado)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"\nNão foi possível gerar o gráfico de importância de features: {e}")