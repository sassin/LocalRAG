

# Troubleshooting

## “Not found in indexed documents”

- Retrieval returned NO_HITS
- Check `records.json`
- Use `rag_get_page` for tables

## Table exists but not retrieved

- Ask explicitly for the table
- Use page-level retrieval

## Slow responses

- Large prompt (many tables)
- Reduce chunk size
- Limit RETURN_EVIDENCE

## Model errors

- Verify API keys
- Check provider selection
