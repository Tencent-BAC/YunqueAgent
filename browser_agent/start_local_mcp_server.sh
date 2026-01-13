#!/bin/bash
#

export PYTHONPATH=.
#nohup python3 -u aiground/sample_mcp_server.py 1>local_mcp_server.log 2>&1 &
python3 -u aiground/sample_mcp_server.py

# conda activate aiground
# chcp 65001
# cd C:\Users\Administrator\Desktop\deepresearch\aiground_agent_1_0
# $env:PYTHONUTF8="1"
# $env:PYTHONIOENCODING="utf-8"
# $env:PYTHONPATH="."
# python -u aiground\sample_mcp_server.py

# 9888：gaia
# 9889: simpleQA