# BugTraceAgent

基于 LangGraph + GitHub MCP 的证据驱动 GitHub 代码分析 Agent。

BugTraceAgent 可以读取真实 GitHub Repository、Issue 与源码，
通过代码搜索和文件读取建立证据链，并生成中文问题分析、
根因假设和修改建议。

## Architecture

User Query
    ↓
FastAPI
    ↓
LangGraph Agent
    ↓
DeepSeek
    ↓
GitHub MCP
    ↓
Issue / Code Search / File Read
    ↓
Evidence-Grounded Analysis

## Features

- GitHub Repository 目录与源码读取
- GitHub Code Search
- GitHub Issue 读取
- 历史 Issue 搜索
- 中文代码问题分析
- PostgreSQL Checkpoint 多轮上下文
- FastAPI REST API
- SSE 流式响应
- GitHub MCP Read-only 模式
- Pytest 单元测试

## Tech Stack

Python 3.12 / FastAPI / LangGraph / MCP / DeepSeek /
PostgreSQL / Docker / Pytest

## Quick Start

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"