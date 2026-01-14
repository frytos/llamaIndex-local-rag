# Grafana MCP Server Setup Guide

This guide explains how to set up the Grafana MCP (Model Context Protocol) server integration with Claude Code to enable AI-assisted Grafana operations.

## Overview

The Grafana MCP server allows Claude to:
- Query and search Grafana dashboards
- Execute PromQL queries against Prometheus datasources
- Run LogQL queries against Loki datasources
- Create and manage annotations
- View alert rules and notification settings
- Create Grafana Incident items
- Generate dashboard summaries and extract specific data

## Prerequisites

- Grafana instance running (included in docker-compose.yml)
- Docker installed (for running the MCP server)
- Claude Code configured

## Quick Start

### 1. Start Grafana

Your project already includes Grafana in docker-compose.yml:

```bash
# Start all monitoring services
docker-compose up -d

# Or start specific services
docker-compose up -d grafana prometheus
```

Grafana will be available at: http://localhost:3000

Default credentials:
- Username: `admin`
- Password: `admin` (or set via GRAFANA_ADMIN_PASSWORD)

### 2. Create a Grafana Service Account Token

1. Log in to Grafana at http://localhost:3000
2. Navigate to **Administration** → **Service Accounts**
3. Click **Add service account**
4. Set name: `claude-mcp-integration`
5. Set role: **Viewer** (or **Editor** if you want write access)
6. Click **Create**
7. Click **Add service account token**
8. Set name: `claude-code-token`
9. Click **Generate token**
10. **Copy the token immediately** (it won't be shown again)

### 3. Configure Environment Variables

Add your token to `.env`:

```bash
# Grafana Configuration
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
GRAFANA_SERVICE_ACCOUNT_TOKEN=your-token-here
```

**Security Note**: Never commit your `.env` file with real tokens to version control!

### 4. Verify MCP Configuration

The MCP server is configured in `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "grafana": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "--network", "llamaindex-local-rag_rag_network",
        "-e", "GRAFANA_URL",
        "-e", "GRAFANA_SERVICE_ACCOUNT_TOKEN",
        "mcp/grafana",
        "-t", "stdio"
      ],
      "env": {
        "GRAFANA_URL": "http://rag_grafana:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "${GRAFANA_SERVICE_ACCOUNT_TOKEN}"
      }
    }
  }
}
```

### 5. Restart Claude Code

After configuration, restart Claude Code to load the MCP server:

```bash
# Exit and restart Claude Code
exit
claude-code
```

## Usage Examples

Once configured, you can ask Claude to interact with Grafana:

### Dashboard Operations
```
"Show me all available Grafana dashboards"
"Summarize the RAG Overview dashboard"
"What metrics are tracked in the system_overview dashboard?"
```

### Query Operations
```
"Query Prometheus for current CPU usage"
"Show me recent error logs from Loki"
"What's the query response time over the last hour?"
```

### Alert Management
```
"List all active alerts"
"Show me the alert rules for database performance"
"What notification channels are configured?"
```

### Annotations
```
"Create an annotation marking the deployment at 2pm"
"Show me recent annotations on the performance dashboard"
```

## Configuration Details

### Docker Network

The MCP server runs in the same Docker network (`llamaindex-local-rag_rag_network`) as your Grafana instance, allowing it to access Grafana using the container name `rag_grafana:3000`.

### Service Account Permissions

Recommended scopes for the service account token:

| Scope | Permission | Purpose |
|-------|------------|---------|
| `dashboards:read` | Read | View and query dashboards |
| `datasources:read` | Read | Access datasource configurations |
| `datasources:query` | Execute | Run PromQL/LogQL queries |
| `annotations:read` | Read | View annotations |
| `annotations:write` | Write | Create annotations (optional) |
| `alerting:read` | Read | View alert rules and status |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GRAFANA_URL` | Grafana instance URL | `http://rag_grafana:3000` |
| `GRAFANA_SERVICE_ACCOUNT_TOKEN` | API token | (required) |
| `GRAFANA_ADMIN_USER` | Admin username | `admin` |
| `GRAFANA_ADMIN_PASSWORD` | Admin password | `admin` |

## Troubleshooting

### MCP Server Not Loading

**Symptom**: Claude doesn't show Grafana tools

**Solutions**:
1. Check that `.claude/mcp.json` exists and is valid JSON
2. Verify `GRAFANA_SERVICE_ACCOUNT_TOKEN` is set in `.env`
3. Restart Claude Code
4. Check Claude Code logs for MCP errors

### Connection Refused

**Symptom**: "Connection refused" errors when querying Grafana

**Solutions**:
1. Verify Grafana is running: `docker ps | grep grafana`
2. Check network: `docker network ls | grep rag_network`
3. Test connection from MCP container:
   ```bash
   docker run --rm --network llamaindex-local-rag_rag_network curlimages/curl \
     curl -v http://rag_grafana:3000/api/health
   ```

### Invalid Token

**Symptom**: "Invalid API token" or 401 errors

**Solutions**:
1. Verify token is correctly copied (no extra spaces/newlines)
2. Check service account is active in Grafana
3. Regenerate token if needed
4. Ensure token has required permissions

### Docker Image Not Found

**Symptom**: "Unable to find image 'mcp/grafana'"

**Solution**: Pull the image manually:
```bash
docker pull mcp/grafana:latest
```

## Advanced Configuration

### Custom Grafana URL

If using a different Grafana instance:

```json
{
  "mcpServers": {
    "grafana": {
      "env": {
        "GRAFANA_URL": "https://your-grafana.example.com",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "${GRAFANA_SERVICE_ACCOUNT_TOKEN}"
      }
    }
  }
}
```

### Multiple Grafana Instances

Configure multiple MCP servers:

```json
{
  "mcpServers": {
    "grafana-local": {
      "command": "docker",
      "args": [...],
      "env": {
        "GRAFANA_URL": "http://localhost:3000",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "${GRAFANA_LOCAL_TOKEN}"
      }
    },
    "grafana-prod": {
      "command": "docker",
      "args": [...],
      "env": {
        "GRAFANA_URL": "https://grafana.prod.example.com",
        "GRAFANA_SERVICE_ACCOUNT_TOKEN": "${GRAFANA_PROD_TOKEN}"
      }
    }
  }
}
```

### Read-Only Mode

For safety, run in read-only mode:

```json
{
  "mcpServers": {
    "grafana": {
      "args": [
        "run", "--rm", "-i",
        "--network", "llamaindex-local-rag_rag_network",
        "-e", "GRAFANA_URL",
        "-e", "GRAFANA_SERVICE_ACCOUNT_TOKEN",
        "mcp/grafana",
        "-t", "stdio",
        "--disable-write"
      ]
    }
  }
}
```

## Security Best Practices

1. **Use Service Account Tokens**: Never use personal user credentials
2. **Principle of Least Privilege**: Grant only required permissions
3. **Token Rotation**: Regularly rotate service account tokens
4. **Environment Variables**: Store tokens in `.env`, never hardcode
5. **Audit Logs**: Review Grafana audit logs for MCP access
6. **Read-Only by Default**: Use `--disable-write` unless needed

## Features Overview

### Available Tools

The MCP server provides these capabilities:

- **Dashboard Management**
  - Search and list dashboards
  - Get dashboard JSON and summaries
  - Create/update dashboards (if write enabled)

- **Data Queries**
  - Execute PromQL instant and range queries
  - Run LogQL queries
  - Access metric metadata

- **Monitoring**
  - View alert rules and status
  - List notification policies
  - Check OnCall schedules

- **Investigation**
  - Create incidents
  - Manage Sift investigations
  - Analyze error patterns

- **Annotations**
  - Create deployment markers
  - Query historical annotations
  - Link annotations to dashboards

## Next Steps

- Explore your dashboards: `config/grafana/dashboards/`
- Learn Prometheus queries: [PromQL docs](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- Configure alerts: [Grafana Alerting docs](https://grafana.com/docs/grafana/latest/alerting/)
- Set up Loki: [Loki setup guide](https://grafana.com/docs/loki/latest/setup/)

## References

- [Grafana MCP Server GitHub](https://github.com/grafana/mcp-grafana)
- [Model Context Protocol Docs](https://modelcontextprotocol.io/)
- [Grafana API Documentation](https://grafana.com/docs/grafana/latest/developers/http_api/)
- [Service Account Guide](https://grafana.com/docs/grafana/latest/administration/service-accounts/)
