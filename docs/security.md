

# Security & Privacy

## Data locality

- Documents are indexed locally
- No document content is uploaded
- Only retrieved excerpts are sent to LLMs

## Optional access key

Set:
```bash
APP_ACCESS_KEY=your-secret

Clients must send:
X-ACCESS-KEY: your-secret


Threat model

Protected against:
Accidental document leaks
Prompt-based exfiltration
Memory poisoning
Not designed for:
Multi-tenant hostile environments