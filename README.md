<div align="center">
  <picture>
      <img src="./assets/logo.png" width="85%">
  </picture>
</div>

<p align="center">
<p align="center">
🤗 <a href="https://huggingface.co/TencentBAC" target="_blank">HuggingFace</a> |
🤖 <a href="https://huggingface.co/TencentBAC" target="_blank">ModelScope</a>  | 📑 <a href="https://huggingface.co/TencentBAC">Tech Report</a>


<!-- # Yunque DeepResearch -->

# Introduction

Deep research has emerged as a transformative capability for autonomous agents, empowering Large Language Models to navigate complex, open-ended tasks. However, realizing its full potential is hindered by critical limitations, including escalating contextual noise in long-horizon tasks, fragility leading to cascading errors, and a lack of modular extensibility. To address these challenges, we introduce **Yunque DeepResearch**, a hierarchical, modular, and robust framework. The architecture is characterized by three key components: (1) a centralized *Multi-Agent Orchestration System* that routes subtasks to an *Atomic Capability Pool* of tools and specialized sub-agents; (2) a *Dynamic Context Management* mechanism that structures completed sub-goals into semantic summaries to mitigate information overload; and (3) a proactive *Supervisor Module* that ensures resilience through active anomaly detection and context pruning. Yunque DeepResearch achieves state-of-the-art performance across a range of agentic deep research benchmarks, including GAIA, BrowseComp, BrowseComp-ZH, and Humanity’s Last Exam. We open-source the framework, reproducible implementations, and application cases to empower the community.

More details can be found in our  📑 [Tech Report](https://huggingface.co/TencentBAC).

<p align="center">
  <img width="75%" src="./assets/benchmark.png">
</p>

# 🚀 Application Demos

Experience Yunque DeepResearch in action. The demonstration below highlights the system's observability and workflow via our interactive visualization interface. The source code for this UI is available in the [DeepResearchUI repository](https://github.com/fzd9752/DeepResearchUI).

<div align="center">
  <video src="https://github.com/user-attachments/assets/5ed7b768-c038-4283-a5f6-a0f593e41a51" controls width="90%"></video>
</div>

<br>

We deployed Yunque DeepResearch in the challenging domain of **Intelligent Content Moderation**, where it functions as an AI Copilot to significantly enhance review efficiency. 
<br> 
*我们将云雀 DeepResearch 部署于智能内容审核领域。系统作为 AI 审核助手，能够精准识别暴力、色情、恐怖、营销广告及违法违规等复杂内容，显著提升审核效率。*

<!-- <div style="display: flex; justify-content: space-between; gap: 10px;">
  <div style="width: 49%;">
    <video src="https://github.com/user-attachments/assets/f4ae278a-726c-41e8-aece-6cc979c1fd70" controls width="100%"></video>
    <p align="center">
      <b>Case 1: Automated Pre-labeling</b><br>
      <i>案例 1：自动化预标注</i>
    </p>
  </div>
  <div style="width: 49%;">
    <video src="https://github.com/user-attachments/assets/285a5153-1759-4c18-a830-560eb0ec21ce" controls width="100%"></video>
    <p align="center">
      <b>Case 2: Human Decision Verification</b><br>
      <i>案例 2：校验人审误标</i>
    </p>
  </div>
</div> -->
  
<table>
  <tr>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/f4ae278a-726c-41e8-aece-6cc979c1fd70" controls width="100%"></video>
      <br>
      <p align="center">
        <b>Case 1: Automated Pre-labeling</b><br>
        <i>案例 1：自动化预标注</i>
      </p>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/285a5153-1759-4c18-a830-560eb0ec21ce" controls width="100%"></video>
      <br>
      <p align="center">
        <b>Case 2: Human Decision Verification</b><br>
        <i>案例 2：校验人审误标</i>
      </p>
    </td>
  </tr>
</table>

<br>

**📧 Contact & Collaboration**
<br>
For customized development or enterprise solutions tailored to your specific scenarios, please contact us at: **[{yuyangyin, yukiyxcai}@tencent.com](mailto:yuyangyin@tencent.com,yukiyxcai@tencent.com)**
<br>
*如需定制化开发或特定业务场景落地合作，欢迎联系我们。*

# Features

- ⚙️ **Effective Orchestration System**: We implement a centralized orchestration framework anchored by a Main Agent that serves as the strategic core. Utilizing a flexible dispatch mechanism, the planner dynamically routes tasks to the most appropriate resource within the Atomic Capability Pool: it directly invokes basic tools for low-latency atomic operations while delegating complex, long-horizon objectives to specialized sub-agents.
- 🗂️ **Dynamic Context Management**: We propose a **sub-goal-driven memory mechanism** to resolve the tension between context length and information density. By treating sub-goals as the fundamental unit of trajectory segmentation, our system dynamically partitions the research process: completed sub-goals are folded into concise structured summaries to maintain global planning awareness, while the active sub-goal retains fine-grained ReAct traces for precise execution. This hybrid approach transforms linear history into structured semantic milestones.
- 🧩 **Modularity and Extensibility**: We ensure adaptability through a modular **"Atomic Capability Pool"** that separates strategic planning from action execution. By standardizing basic tools and specialized sub-agents as functional units, our architecture attains high composability. This separation creates an extensible ecosystem where new capabilities—ranging from atomic utility functions to expert-level solvers—can be dynamically registered and deployed, ensuring the framework remains resilient to evolving requirements.
- 🛡️ **Stability and Robustness**: We incorporate a dedicated **Supervisor module** to ensure system stability and mitigate the fragility often seen in long-horizon tasks. Unlike rigid reflection schedules, this mechanism performs active anomaly detection on the agent's trajectory. Upon identifying failures, it triggers a self-correction protocol, explicitly prunes invalid context to prevent memory pollution, guiding the agent to autonomously recover and synthesize a viable alternative response.

<p align="center">
  <img width="90%" src="./assets/framework.png">
</p>


# Quick Start

This guide provides instructions for setting up the environment and running inference scripts.

## 1. Environment Setup

```bash
conda create -n yunque-dr python=3.10.0
conda activate yunque-dr
pip install -r requirements.txt
```

## 2. Configuration & Data Preparation

### Environment Setup

Initialize your configuration by copying the example environment file:

```bash
# Copy the example environment file
cp .env.example .env
```

Edit the `.env` file and provide your actual API keys and configuration values:

- **AGENT_API_KEY/AGENT_API_BASE**: OpenAI-compatible API credentials for the main agent.
- **SUMMARY_API_KEY/SUMMARY_API_BASE**: OpenAI-compatible API credentials for the summary model.
- **LLM_NAME/SUMMARY_MODEL_NAME**: The specific model names you wish to deploy.
- **SERPER_KEY_ID**: Your API key from [Serper.dev](https://serper.dev/) (used for web search and Google Scholar).
- **JINA_API_KEYS**: Your API key from [Jina.ai](https://jina.ai/) (used for parsing web pages).
- **SANDBOX_FUSION_ENDPOINT**: The endpoint for the Python interpreter sandbox (refer to [SandboxFusion](https://github.com/bytedance/SandboxFusion)).
- **DATASET**: Path to your evaluation dataset.
- **OUTPUT_PATH**: Directory for saving results.

**Note:** For detailed descriptions of each variable, please refer to the comments inside `.env.example`.

### Input Data Format

The system accepts data in JSONL format. Prepare your dataset (e.g., `questions.jsonl`) using the following structure:

```json
{"question": "What is the capital of France?", "answer": "Paris"}
{"question": "Explain quantum computing", "answer": ""}
```

**Note:** The `answer` field serves as the ground truth for automated evaluation. The system will generate its own response and compare it against this reference to calculate performance metrics.

## 3. Prepare API to call the model

The framework supports any **OpenAI-compatible API** (e.g., OpenAI, DeepSeek, vLLM, SGLang).

- **External Providers**: Get your API Key and Base URL (e.g., `https://api.openai.com/v1`).
- **Self-Hosted (vLLM/SGLang)**: Start your inference server.

  ```bash
  # Example: Launch vLLM server
  python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --dtype auto \
    --api-key token-abc123
  ```

## 4. Run Inference

Execute the inference script:

```bash
bash run.sh
```

## 📧 Contact Information

For customized development or enterprise solutions tailored to your specific scenarios, please contact us at: **[{yuyangyin, yukiyxcai}@tencent.com](mailto:yuyangyin@tencent.com,yukiyxcai@tencent.com)**
<br>
*如需定制化开发或特定业务场景落地合作，欢迎联系我们。*

## Acknowledgement

We thank the open-source community for their contributions, especially the authors of the following projects:

- [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- [browser-use](https://github.com/browser-use/browser-use)
- [OpenManus](https://github.com/FoundationAgents/OpenManus)
