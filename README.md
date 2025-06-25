# Aplicação Web de Modelo Preditivo Binário

## Visão Geral

O projeto consiste em uma aplicação web cujo objetivo é disponibilizar, de forma dinâmica e flexível, o uso de modelos preditivos para prever um target binário e identificar as principais features determinantes para a previsão. O sistema foi pensado para ser facilmente adaptável a diferentes conjuntos de dados, desde que respeitem os requisitos de ter um target binário e features categóricas.

## Arquitetura e Tecnologias Utilizadas

- **Backend (API):**
  - **Framework:** Flask (Python)
  - **Funcionalidade:** A API foi desenvolvida para receber, via requisição, o modelo preditivo, o nome do target e a lista de features. Isso possibilita ao usuário enviar qualquer tabela, desde que siga as regras estabelecidas:
    - O target deve ser binário.
    - As features devem ser categóricas.
  - **Diferencial:** A abordagem dinâmica permite o uso flexível de diversos datasets sem necessidade de ajustes no código-fonte, bastando respeitar os formatos definidos.
  - **Saída Avançada:** Além das previsões, a API retorna gráficos e estatísticas detalhadas sobre o desempenho do modelo e sobre as features utilizadas, possibilitando a visualização de métricas como acurácia, matriz de confusão, importância das variáveis, entre outros. Esses dados são enviados prontos para exibição no frontend, facilitando a análise pelo usuário.

- **Frontend:**
  - **Framework:** Svelte
  - **Funcionalidade:** O frontend foi desenvolvido em Svelte, proporcionando uma interface moderna e responsiva para interação com a API. O usuário pode:
    - Enviar seus próprios dados (upload de tabelas).
    - Selecionar o target e as features.
    - Visualizar as previsões do modelo.
    - Acompanhar gráficos e estatísticas retornadas pela API, como métricas preditivas, gráficos de importância de features, matriz de confusão, etc.
    - Ver os principais fatores determinantes para as decisões do modelo.

## Funcionalidades Principais

1. **Upload de Dados:** O usuário pode enviar uma tabela de dados para análise.
2. **Seleção Dinâmica:** O sistema identifica automaticamente as colunas disponíveis para seleção do target e das features.
3. **Previsão:** O sistema executa o modelo preditivo sobre os dados enviados, retornando as previsões do target binário.
4. **Interpretabilidade:** O sistema apresenta as features mais determinantes na previsão, auxiliando o usuário a entender os fatores que impactam o resultado.
5. **Visualização de Resultados:** O frontend exibe gráficos e estatísticas gerados pela API, permitindo análise visual e detalhada dos resultados e do desempenho do modelo.


## Como Utilizar

Siga o passo a passo abaixo para rodar o projeto localmente:


Tutorial no youtube https://youtu.be/6p2G6xTxX0M

### 1. Clone o repositório

```bash
git clone https://github.com/caioceo/machineLearning-webapp
```

---

### 2. Configuração e execução do backend (servidor)

```bash
cd machineLearning-webapp/backend
pip install -r requirements.txt
flask --app main.py run
```

---

### 3. Configuração e execução do frontend (interface web)

Abra um novo terminal e execute:

```bash
cd machineLearning-webapp/frontend
npm install
npm run dev
```

---

Pronto! Agora o backend estará rodando em um terminal e o frontend em outro. Acesse a aplicação web pelo endereço informado no terminal do frontend (normalmente http://localhost:5173).

### Regras e Restrições

- O dataset enviado pelo usuário deve conter:
  - Um target binário (apenas dois valores possíveis).
  - Features categóricas (sem valores numéricos contínuos).
- O modelo preditivo deve ser compatível com as regras acima.

## Conclusão

O projeto proporciona uma solução prática e eficiente para a aplicação de modelos preditivos binários em diferentes conjuntos de dados categóricos, focando em flexibilidade, facilidade de uso, interpretabilidade e visualização intuitiva dos resultados. A separação clara entre backend (Flask) e frontend (Svelte) garante escalabilidade e manutenção facilitada do sistema.

---

**Autor:** Caio César de Oliveira  
