# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Mateus Junior de Macedo Cavalcanti

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em [train_model.py](train_model.py) usa três blocos convolucionais com `Conv2D`, `BatchNormalization`, `ReLU` e `MaxPooling2D`, seguida de uma camada de `Dropout` e de uma saída densa com 10 classes e `softmax`. O treinamento foi feito com validação implícita via `validation_split=0.1`, com `EarlyStopping` monitorando `val_loss` para evitar overfitting e preservar o melhor peso do modelo. O `learning_rate=5e-4` foi escolhido para estabilizar o aprendizado em CPU, enquanto o `dropout=0.4` e o `batch_size=128` ajudaram a reduzir overfitting sem tornar o treino excessivamente lento.

### 2️⃣ Bibliotecas Utilizadas

- TensorFlow 2.18.0
- NumPy

### 3️⃣ Técnica de Otimização do Modelo

O modelo foi convertido para TensorFlow Lite em [optimize_model.py](optimize_model.py) usando `tf.lite.TFLiteConverter` com `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, o que aplica quantização por intervalo dinâmico para reduzir o tamanho do artefato de edge.

### 4️⃣ Resultados Obtidos

- Após a execução do treinamento, a saída mostrou uma acurácia de validação de 0.9710.
- Após o treino, o arquivo [model.h5](model.h5) foi medido no sistema e apresentou tamanho de 1.065.752 bytes, cerca de 1,02 MB.
- Após a conversão para TFLite, o arquivo [model.tflite](model.tflite) foi medido no sistema e apresentou tamanho de 89.800 bytes, cerca de 87,70 KB.
- Ao comparar os dois arquivos, a redução de tamanho foi de 975.952 bytes, o que representa cerca de 91,57% em relação ao modelo original.

### 5️⃣ Comentários Adicionais (Opcional)

Durante o treinamento, a validação inicial mostrou instabilidade, então a combinação de `EarlyStopping`, `dropout` e `learning_rate` mais controlado foi fundamental para manter um treinamento estável em CPU. A principal limitação observada foi que, embora o modelo tenha se comportado bem no MNIST, ele ainda depende de uma arquitetura simples e de um dataset bem delimitado, o que pode limitar a generalização para imagens mais complexas.

### 6️⃣ Exemplo de Inferência

Saída observada ao rodar [run_inference.py](run_inference.py):

```text
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

No exemplo executado, as 5 amostras foram classificadas corretamente, já que a saída do [run_inference.py](run_inference.py) mostrou o valor predito e o valor real para cada uma delas e eles coincidiram.
