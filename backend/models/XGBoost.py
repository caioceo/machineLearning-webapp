import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Modelagem e Pré-processamento
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay, 
                            roc_auc_score, roc_curve, auc, accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_categorical_model(df, target, columns=None, predict_nan=False):
    """
    Treina um modelo XGBoost para classificação binária com dados categóricos.
    
    Parâmetros:
    df (pandas.DataFrame): DataFrame com os dados
    target (str): Nome da coluna do target binário (0 ou 1)
    columns (list): Lista de colunas para usar como features. Se None, usa todas exceto o target
    predict_nan (bool): Se True, tenta prever valores NaN no target ao invés de removê-los
    
    Retorna:
    dict: Dicionário contendo o modelo treinado e métricas de avaliação
    """
    # Configurações iniciais
    print("=="*30)
    print("INICIANDO PIPELINE DE MODELAGEM COM XGBOOST")
    print("=="*30)
    
    # Fazer uma cópia do DataFrame para evitar modificar o original
    df_work = df.copy()
    
    # Verificar que target é uma string (nome da coluna)
    if not isinstance(target, str):
        raise TypeError(f"O parâmetro 'target' deve ser uma string com o nome da coluna, não {type(target)}")
    
    # Verificar se a coluna target existe no DataFrame
    if target not in df_work.columns:
        raise ValueError(f"A coluna target '{target}' não foi encontrada no DataFrame")
    
    # Preparação dos dados - garantir que columns é uma lista de colunas
    if columns is None:
        columns = [col for col in df_work.columns if col != target]
    else:
        # Remover o target da lista de colunas se estiver presente
        columns = [col for col in columns if col != target]
    
    # Verificar se todas as colunas existem
    missing_columns = [col for col in columns if col not in df_work.columns]
    if missing_columns:
        raise ValueError(f"As seguintes colunas não foram encontradas no DataFrame: {missing_columns}")
        
    print(f"Target: {target}")
    print(f"Número de features: {len(columns)}")
    
    # Lidar com valores NaN no target
    nan_mask = df_work[target].isna()
    nan_count = nan_mask.sum()
    
    if nan_count > 0:
        print(f"\nEncontrados {nan_count} valores NaN na coluna target ({100*nan_count/len(df_work):.2f}% dos dados).")
        
        if predict_nan:
            print("Tentando prever valores NaN no target...")
            
            # Separar dados com target conhecido e desconhecido
            df_known = df_work[~nan_mask].copy()
            df_unknown = df_work[nan_mask].copy()
            
            # Verificar se temos dados suficientes para treinar um modelo de imputação
            if len(df_known) < 20:  # número arbitrário, ajuste conforme necessário
                print("AVISO: Poucos dados com target conhecido. A predição de NaN pode ser imprecisa.")
            
            # Verificar se o target conhecido é binário
            unique_values = df_known[target].unique()
            if len(unique_values) != 2:
                print(f"AVISO: O target deve ser binário. Valores encontrados: {unique_values}")
                print("Os valores NaN serão removidos em vez de previstos.")
                df_work = df_work[~nan_mask].copy()
            else:
                # Treinar modelo auxiliar para prever o target
                X_known = df_known[columns]
                y_known = df_known[target]
                
                # Preprocessamento das features
                categorical_features = []
                numeric_features = []
                
                for col in columns:
                    # Se o tipo da coluna for objeto, string ou categoria, consideramos categórica
                    if X_known[col].dtype == 'object' or X_known[col].dtype == 'string' or X_known[col].dtype.name == 'category':
                        categorical_features.append(col)
                    # Se for numérica, consideramos numérica
                    elif np.issubdtype(X_known[col].dtype, np.number):
                        numeric_features.append(col)
                    # Para outros casos, convertemos para categórica
                    else:
                        categorical_features.append(col)
                        X_known[col] = X_known[col].astype(str)
                
                # Tratamento de valores ausentes nas features
                for col in categorical_features:
                    X_known[col] = X_known[col].fillna('Não Informado')
                
                for col in numeric_features:
                    X_known[col] = pd.to_numeric(X_known[col], errors='coerce').fillna(X_known[col].median())
                
                # Preparar preprocessador para o modelo auxiliar
                transformers = []
                if categorical_features:
                    transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
                                        categorical_features))
                if numeric_features:
                    transformers.append(('num', StandardScaler(), numeric_features))
                
                preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
                
                # Modelo auxiliar para imputação
                imputation_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('imputer_model', XGBClassifier(
                        random_state=42,
                        n_jobs=-1,
                        objective='binary:logistic',
                        eval_metric='logloss',
                        use_label_encoder=False
                    ))
                ])
                
                # Treinar modelo auxiliar com validação cruzada para avaliar qualidade
                cv_results = []
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, val_idx in skf.split(X_known, y_known):
                    X_train_cv, X_val_cv = X_known.iloc[train_idx], X_known.iloc[val_idx]
                    y_train_cv, y_val_cv = y_known.iloc[train_idx], y_known.iloc[val_idx]
                    
                    imputation_pipeline.fit(X_train_cv, y_train_cv)
                    y_pred_cv = imputation_pipeline.predict(X_val_cv)
                    cv_accuracy = accuracy_score(y_val_cv, y_pred_cv)
                    cv_results.append(cv_accuracy)
                
                mean_accuracy = np.mean(cv_results)
                print(f"Qualidade do modelo auxiliar (média da acurácia em 5-fold CV): {mean_accuracy:.4f}")
                
                if mean_accuracy < 0.65:  # limiar arbitrário, ajuste conforme necessário
                    print("AVISO: Modelo auxiliar tem baixa precisão. Os valores NaN serão removidos.")
                    df_work = df_work[~nan_mask].copy()
                else:
                    # Treinar modelo de imputação com todos os dados conhecidos
                    imputation_pipeline.fit(X_known, y_known)
                    
                    # Preparar dados desconhecidos
                    X_unknown = df_unknown[columns]
                    
                    # Aplicar o mesmo preprocessamento
                    for col in categorical_features:
                        X_unknown[col] = X_unknown[col].fillna('Não Informado')
                    
                    for col in numeric_features:
                        X_unknown[col] = pd.to_numeric(X_unknown[col], errors='coerce').fillna(X_unknown[col].median())
                    
                    # Prever valores para o target
                    y_pred_unknown = imputation_pipeline.predict(X_unknown)
                    
                    # Atribuir valores previstos
                    df_unknown[target] = y_pred_unknown
                    
                    # Combinar dados conhecidos e previstos
                    df_work = pd.concat([df_known, df_unknown])
                    
                    print(f"Valores NaN previstos: {sum(y_pred_unknown == 1)} positivos, {sum(y_pred_unknown == 0)} negativos")
        else:
            # Se não estamos prevendo NaN, simplesmente remova as linhas
            print("Removendo linhas com valores NaN no target...")
            df_work = df_work[~nan_mask].copy()
            print(f"Restaram {len(df_work)} linhas após a remoção.")
    
    # Verificar novamente se o target é binário após a imputação/remoção de NaN
    unique_values = df_work[target].unique()
    if len(unique_values) != 2:
        raise ValueError(f"O target deve ser binário. Valores encontrados: {unique_values}")
    
    # Distribuição do target
    target_counts = df_work[target].value_counts()
    print("\nDistribuição do target:")
    for value, count in target_counts.items():
        percentage = 100 * count / len(df_work)
        print(f"- Valor {value}: {count} registros ({percentage:.2f}%)")
    
    # Identificando tipos de colunas automaticamente
    X = df_work[columns].copy()
    y = df_work[target].copy()
    
    # Identificar colunas categóricas e numéricas novamente para o conjunto completo
    categorical_features = []
    numeric_features = []
    
    for col in columns:
        # Se o tipo da coluna for objeto, string ou categoria, consideramos categórica
        if X[col].dtype == 'object' or X[col].dtype == 'string' or X[col].dtype.name == 'category':
            categorical_features.append(col)
        # Se for numérica, consideramos numérica
        elif np.issubdtype(X[col].dtype, np.number):
            numeric_features.append(col)
        # Para outros casos, convertemos para categórica
        else:
            categorical_features.append(col)
            X[col] = X[col].astype(str)
    
    print(f"\nFeatures categóricas ({len(categorical_features)}): {categorical_features}")
    print(f"\nFeatures numéricas ({len(numeric_features)}): {numeric_features}")
    
    # Tratamento de valores ausentes
    for col in categorical_features:
        X[col] = X[col].fillna('Não Informado')
    
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median())
    
    # Divisão em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    print(f"\nConjunto de treino: {X_train.shape[0]} exemplos")
    print(f"Conjunto de teste: {X_test.shape[0]} exemplos")
    
    # Cálculo do balanceamento de classes para o XGBoost
    class_counts = y_train.value_counts()
    if len(class_counts) == 2:
        scale_pos_weight_value = class_counts[0] / class_counts[1]
        print(f"\nFator de balanceamento (scale_pos_weight): {scale_pos_weight_value:.2f}")
    else:
        scale_pos_weight_value = 1.0
    
    # Configuração do preprocessador
    transformers = []
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))
    if numeric_features:
        transformers.append(('num', StandardScaler(), numeric_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Configuração do pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            random_state=42,
            n_jobs=-1,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight_value
        ))
    ])
    
    # Determine o número adequado de folds para CV com base no tamanho das classes
    min_class_samples = min(class_counts)
    n_splits = min(5, int(min_class_samples))  # Usar no máximo 5 folds, mas ajustar para dados pequenos
    
    if n_splits <= 1:
        print(f"\nAVISO: Classe minoritária tem apenas {min_class_samples} amostra(s). A validação cruzada será desabilitada.")
        # Se não houver amostras suficientes para validação cruzada, use o conjunto de teste simples
        cv = None
    else:
        print(f"\nUsando validação cruzada com {n_splits} divisões devido ao tamanho dos dados.")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Busca de hiperparâmetros
    print("\nIniciando busca de hiperparâmetros...")
    params = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [1, 1.5, 2]
    }
    
    # Se o conjunto de dados for muito pequeno, reduza o número de iterações
    n_iter = min(20, X_train.shape[0] // 2) if X_train.shape[0] < 100 else 20
    
    # Configurar a busca de hiperparâmetros com o CV ajustado
    search = RandomizedSearchCV(
        pipeline, params, n_iter=n_iter,
        cv=cv, scoring='f1_macro',
        verbose=1, n_jobs=-1, random_state=42
    ) if cv else pipeline  # Se não há CV suficiente, use o pipeline padrão
    
    if cv:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print("\nMelhores parâmetros encontrados:")
        print(search.best_params_)
    else:
        # No caso de dados muito pequenos, treine o modelo sem otimização
        search.fit(X_train, y_train)
        best_model = search
        print("\nModelo treinado sem otimização de hiperparâmetros devido ao tamanho dos dados.")
    
    # Avaliação do modelo
    y_pred_train = best_model.predict(X_train)
    y_proba_train = best_model.predict_proba(X_train)[:, 1]
    y_pred_test = best_model.predict(X_test)
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*60)
    print("AVALIAÇÃO COMPARATIVA: TREINO VS. TESTE")
    print("="*60)
    
    print("\n--- Relatório de Classificação (TREINO) ---")
    print(f"AUC-ROC no Treino: {roc_auc_score(y_train, y_proba_train):.3f}")
    print(classification_report(y_train, y_pred_train))
    
    print("\n--- Relatório de Classificação (TESTE) ---")
    print(f"AUC-ROC no Teste: {roc_auc_score(y_test, y_proba_test):.3f}")
    print(classification_report(y_test, y_pred_test))
    
    # Matriz de Confusão
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Matrizes de Confusão Comparativas (XGBoost)', fontsize=16)
    ConfusionMatrixDisplay.from_estimator(best_model, X_train, y_train, cmap='Greens', ax=axes[0])
    axes[0].set_title('Conjunto de Treino')
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, cmap='Blues', ax=axes[1])
    axes[1].set_title('Conjunto de Teste')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # Curva ROC
    print("\n" + "="*60)
    print("GERANDO CURVA ROC COMPARATIVA")
    print("="*60)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    plt.figure(figsize=(11, 8))
    plt.title('Curva ROC Comparativa: Treino vs. Teste', fontsize=14)
    plt.plot(fpr_train, tpr_train, color='steelblue', lw=2, linestyle='--', 
             label=f'Curva ROC Treino (AUC = {roc_auc_train:.3f})')
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, 
             label=f'Curva ROC Teste (AUC = {roc_auc_test:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle=':', label='Classificador Aleatório')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.show()
    
    # Verificar se temos dados suficientes para a curva de aprendizagem
    if X.shape[0] >= 20:  # pelo menos 20 amostras para fazer análise de curva de aprendizagem
        # Curva de Aprendizagem
        print("\n" + "="*60)
        print("GERANDO CURVA DE APRENDIZAGEM")
        print("="*60)
        
        # Determinar número de pontos na curva baseado no tamanho dos dados
        n_points = min(10, X.shape[0] // 3)
        if n_points < 3:
            n_points = 3  # pelo menos 3 pontos
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=best_model, X=X, y=y, 
            cv=min(3, n_splits) if n_splits > 1 else 2,  # reduzir para dados pequenos
            n_jobs=-1,
            train_sizes=np.linspace(0.3, 1.0, n_points),  # começar com 30% dos dados
            scoring='f1_macro'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(12, 7))
        plt.title("Curva de Aprendizagem do Modelo")
        plt.xlabel("Tamanho do Conjunto de Treinamento")
        plt.ylabel("Pontuação F1-Macro")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="darkorange", label="Score de Treino")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="steelblue", label="Score de Validação Cruzada")
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="darkorange")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="steelblue")
        plt.legend(loc="best")
        plt.show()
    else:
        print("\nConjunto de dados muito pequeno para gerar curva de aprendizagem.")
    
    # Importância das Features
    print("\n" + "="*60)
    print("GERANDO GRÁFICO DE IMPORTÂNCIA DAS FEATURES")
    print("="*60)
    
    try:
        feature_names = []
        # Obter nomes de colunas após one-hot encoding
        for name, transformer, cols in best_model.named_steps['preprocessor'].transformers_:
            if name == 'cat':
                if hasattr(transformer, 'get_feature_names_out'):
                    cat_features_out = transformer.get_feature_names_out(cols)
                    feature_names.extend(cat_features_out)
            elif name == 'num':
                feature_names.extend(cols)
        
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({
            'Feature': feature_names, 
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 10))
        top_n = min(20, len(feature_importances))  # Mostrar no máximo 20 features
        sns.barplot(data=feature_importances.head(top_n), x='Importance', y='Feature', palette='cividis')
        plt.title(f'Top {top_n} Features Mais Importantes')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nNão foi possível gerar o gráfico de importância de features: {e}")
    
    # Retornar resultados
    results = {
        'model': best_model,
        'best_params': search.best_params_ if hasattr(search, 'best_params_') else None,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'train_f1': f1_score(y_train, y_pred_train, average='weighted'),
        'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
        'train_auc': roc_auc_score(y_train, y_proba_train),
        'test_auc': roc_auc_score(y_test, y_proba_test)
    }
    
    # Se estamos prevendo NaN, adicionar o modelo de imputação aos resultados
    if predict_nan and nan_count > 0 and 'imputation_pipeline' in locals():
        results['imputation_model'] = imputation_pipeline
        results['imputation_accuracy'] = mean_accuracy
    
    return results