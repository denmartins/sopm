# SOPM — Interface (execução via Docker)

Interface interativa do trabalho **Self-Organizing Prompt Maps for Lightweight
Prompt Adaptation**: você digita um prompt, o sistema o projeta sobre um mapa
auto-organizável (SOM) de prompts, recupera exemplos semelhantes e contrastantes
e pede a um LLM local que reescreva o prompt com base neles.

Tudo roda em containers: **não é preciso instalar Python, PyTorch nem Ollama na
máquina**, apenas o Docker.

---

## 1. Pré-requisitos

- **Docker Desktop** (Windows/macOS) ou **Docker Engine + plugin Compose** (Linux)
  — <https://docs.docker.com/get-started/get-docker/>
- ~15 GB livres em disco (imagem da aplicação + modelo do LLM)
- 8 GB de RAM (16 GB recomendado se usar o modelo `qwen3` completo)

Não é necessária GPU: a imagem usa PyTorch em CPU.

## 2. Executar

Na pasta do projeto:

```bash
docker compose up --build
```

Quando o log mostrar `You can now view your Streamlit app in your browser`, abra:

**<http://localhost:8501>**

Para encerrar: `Ctrl+C` e, se quiser liberar os containers, `docker compose down`.

### O que acontece no primeiro `up`

| Etapa | Duração aproximada | Observação |
|---|---|---|
| Build da imagem da aplicação | 5–10 min | baixa PyTorch (CPU) e dependências |
| Download do modelo no Ollama | 5–20 min | depende da internet e do modelo escolhido |
| Treino do SOM na primeira consulta | poucos segundos | os embeddings já vêm pré-computados na imagem |

Execuções seguintes sobem em segundos: imagem, modelo do LLM e artefatos ficam
em cache/volumes.

## 3. Modelo do LLM (importante)

O padrão é `qwen3` (~5 GB), o mesmo usado nos experimentos. Para uma demonstração
mais leve, copie `.env.example` para `.env` e troque o modelo:

```bash
cp .env.example .env      # no PowerShell: copy .env.example .env
```

```ini
OLLAMA_MODEL=qwen3:1.7b   # ~1.4 GB, baixa e responde bem mais rápido
# OLLAMA_MODEL=qwen3:0.6b # ~500 MB, o mais leve só para validar a instalação
```

Depois, `docker compose up` novamente. Também é possível trocar o modelo
diretamente na barra lateral da interface, desde que ele já tenha sido baixado:

```bash
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama list      # modelos disponíveis
```

## 4. Usando a interface

1. **Painel esquerdo** — cole um prompt e clique em **Improve prompt**. O sistema
   calcula o embedding, encontra o neurônio vencedor (BMU) no SOM, seleciona
   exemplos *similares* (vizinhança próxima) e *diferentes* (vizinhança distante)
   e pede ao LLM uma versão reescrita do prompt.
2. **Tabelas 3 e 4** — os prompts positivos e negativos que embasaram a reescrita.
3. **Seção 5** — envia qualquer prompt (o original ou o melhorado) ao LLM para
   comparar as respostas.
4. **Painel direito** — a U-Matrix do SOM. Regiões claras são fronteiras entre
   clusters; escuras são áreas homogêneas. O `x` dourado marca onde seu prompt caiu.
   Abaixo do mapa é possível navegar pelos prompts de cada neurônio.

**Barra lateral** — dataset, diretório de artefatos, modelo de embedding, endereço
do Ollama, número/raio de vizinhos e hiperparâmetros do SOM (dimensões, iterações,
`sigma`, taxa de aprendizado, semente). Mudar os parâmetros do SOM dispara um novo
treino, que é então cacheado.

### Outros datasets

O campo *Dataset CSV* aceita qualquer arquivo em `bench_data/`, já incluído na
imagem — por exemplo `bench_data/natural_instructions_n150.csv`,
`bench_data/dolly_n150.csv`, `bench_data/samsum_n150.csv`. O CSV precisa ter as
colunas `id`, `prompt_name`, `input` e `target`.

## 5. Estrutura do que foi containerizado

```
Dockerfile            imagem da aplicação (Python 3.9 + PyTorch CPU + Streamlit)
docker-compose.yml    orquestra os serviços app + ollama + ollama-pull
docker/entrypoint.sh  instala o cache de embeddings no volume no primeiro boot
requirements.txt      dependências fixadas nas versões validadas do projeto
.dockerignore         mantém venv, notebooks e resultados fora da imagem
.env.example          modelo, portas e dataset configuráveis
```

Serviços:

- **`app`** — interface Streamlit na porta 8501.
- **`ollama`** — servidor LLM local, API compatível com OpenAI em `/v1`.
- **`ollama-pull`** — serviço de uso único que baixa o modelo e encerra
  (é normal vê-lo com status `Exited (0)`).

Volumes (persistem entre execuções):

- `ollama-models` — modelos baixados do LLM.
- `sopm-artifacts` — cache de embeddings (`embeddings.sqlite`) e SOMs treinados.

Para começar do zero: `docker compose down -v` (apaga também os volumes).

## 6. Problemas comuns

**A porta 8501 já está em uso.** Defina `APP_PORT=8600` em `.env` e suba de novo.

**`Error calling Ollama` na interface.** O modelo ainda está sendo baixado.
Acompanhe com `docker compose logs -f ollama-pull` e tente novamente ao terminar.

**Respostas do LLM muito lentas.** Esperado em CPU com o `qwen3` de 8B. Use
`OLLAMA_MODEL=qwen3:1.7b`, ou habilite a GPU descomentando o bloco `deploy:` do
serviço `ollama` em `docker-compose.yml` (requer NVIDIA Container Toolkit).

**Build falha ao baixar pacotes.** Rede/proxy. Repita `docker compose build`; as
camadas já concluídas são reaproveitadas.

**Ver os logs de um serviço:** `docker compose logs -f app`

## 7. Execução sem Docker (referência)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run sopmInterface.py
```

Nesse caso é preciso ter o Ollama instalado localmente em `http://localhost:11434`.
