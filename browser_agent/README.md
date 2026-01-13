# browser_agent (open-source)

This is an **open-source browser_agent service** (pure FastAPI + MCP SSE) that provides:
- **Browser automation**: `/browser_use` (session-based)
- **Agent execution**: `/browser_use_agent` (SSE streaming)
- **Web search**: `/web_search` (optional Serper key, with fallback engines)
- **URL to Markdown**: `/url_to_markdown` (via `r.jina.ai`)
- **Browser sessions**: `/create_browser_session`, `/destroy_browser_session`, `/get_current_browser_state`

## Project layout

```bash
.
├── aiground/
│   ├── browser_agent_service/    # browser_agent core implementation (UI-free)
│   ├── open_source_server.py     # entrypoint (prefixless HTTP + /mcp)
│   └── sample_mcp_server.py      # legacy filename, delegates to open_source_server
├── config/
│   ├── browser_agent.yaml          # local config (no secrets committed)
│   └── browser_agent.example.yaml  # config template
├── requirements.txt
└── start_local_mcp_server.sh
```

## Create conda environment (recommended)

```bash
conda create -n aiground python=3.11 -y
conda activate aiground
```

## Install & run

### 1) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```


### 2) Configuration

Copy the config template:

```bash
cp config/browser_agent.example.yaml config/browser_agent.yaml
```

Recommended: store secrets in environment variables (do not commit them):

- `SERPER_KEY_ID`: optional, used for Google Search (Serper)
- `JINA_API_KEY`: optional, used for `url_to_markdown`
- `BROWSER_AGENT_CONFIG`: optional, config file path (default: `config/browser_agent.yaml`)

### 3) Start the service

```bash
export PYTHONPATH=.
python -u aiground/open_source_server.py
```

Or use the script:

```bash
bash start_local_mcp_server.sh
```

Default listen address: `0.0.0.0:9091`

## HTTP API (prefixless)

- `POST /ping`
- `POST /browser_use`
- `POST /web_search`
- `POST /url_to_markdown`
- `POST /create_browser_session`
- `POST /destroy_browser_session`
- `GET /get_current_browser_state`
- `POST /browser_use_agent` (`text/event-stream`)

## MCP SSE

- `GET /mcp`
- `POST /mcp/messages/`


