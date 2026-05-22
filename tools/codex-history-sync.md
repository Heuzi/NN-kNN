# Codex History Sync

This helper copies only local Codex conversation history between computers.

It includes:
- `.codex\sessions`
- `.codex\session_index.jsonl`

It does not copy:
- `.codex\auth.json`
- `.codex\state_*.sqlite`
- `.codex\logs_*.sqlite`
- `.codex\memories`

## Export on computer A

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\Sync-CodexHistory.ps1 -Mode export -Path "D:\CodexHistory"
```

You can point `-Path` to a USB drive or a cloud-synced folder such as OneDrive or Dropbox.

## Import on computer B

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\Sync-CodexHistory.ps1 -Mode import -Path "D:\CodexHistory"
```

The import merges `session_index.jsonl` by session `id`, so it is safer than replacing the file outright.

## Notes

- Close Codex before exporting or importing for the cleanest results.
- If your `.codex` folder lives somewhere else, pass `-CodexRoot "C:\path\to\.codex"`.
