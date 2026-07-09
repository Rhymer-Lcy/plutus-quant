# Unattended daily wrapper for the net-payout FREE-DATA forward record (Windows Task Scheduler).
# Mirrors the sibling hermes-quant/scripts/paper_live.ps1. Captures stdout+stderr to a timestamped
# log (Task Scheduler discards them otherwise) and propagates the Python exit code. Register with:
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 register
# Do NOT register with a bare `schtasks /ST 09:30`: that stores a FLOATING local time, which silently
# drifts away from 09:30 Beijing whenever the machine's timezone changes. schedule_tasks.ps1 anchors
# the trigger to a +08:00 instant instead, so it fires at 09:30 Beijing on any timezone and across DST.
# 09:30 Beijing reliably has the prior US session's finalized daily bar (US close is ~04:00-05:00
# Beijing). The forward-record INCEPTION is fixed (live.strategy.PAPER_INCEPTION); this only refreshes
# "today" -- and because every run recomputes the whole curve from inception, a missed day (PC off /
# holiday) loses nothing: the next run catches up.
#
# RETRY-WITH-BACKOFF: paper_forward.py exits 75 (EX_TEMPFAIL) on a TRANSIENT failure (yfinance
# unreachable / empty pull) and 1 on a fatal error. On 75 this wrapper waits and retries up to
# PLUTUS_RETRY_MAX times spaced PLUTUS_RETRY_DELAY_SEC apart (default 12 x 300 s ~= 1 h). Each run is
# idempotent (recompute-from-inception), so retrying is safe.
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$py = if ($env:PLUTUS_PYTHON) { $env:PLUTUS_PYTHON }
      elseif (Test-Path "D:\Anaconda3\envs\plutus\python.exe") { "D:\Anaconda3\envs\plutus\python.exe" }
      else { "python" }
$logdir = Join-Path $repo "results\paper\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$log = Join-Path $logdir ("paper_forward_{0:yyyyMMdd}.log" -f (Get-Date))

$maxAttempts = if ($env:PLUTUS_RETRY_MAX) { [int]$env:PLUTUS_RETRY_MAX } else { 12 }
$delaySec    = if ($env:PLUTUS_RETRY_DELAY_SEC) { [int]$env:PLUTUS_RETRY_DELAY_SEC } else { 300 }
$EX_TEMPFAIL = 75

# PYTHONIOENCODING=utf-8 so Python emits UTF-8; Start-Process writes raw child output to temp files
# (no PowerShell re-encoding, no stderr-as-error wrapping); both are appended to the log in order.
$env:PYTHONIOENCODING = "utf-8"
$script = Join-Path $repo 'scripts\paper_forward.py'

# Pin Yahoo's hostnames to IPs so the pull survives a corporate VPN that breaks the SYSTEM DNS
# resolver for *.finance.yahoo.com (observed: curl "(6) Could not resolve host"). yfinance fetches
# over curl/libcurl, which resolves DNS itself, so the VPN's broken resolver sinks every request even
# though the IPs are reachable. Resolve via a PUBLIC DNS server (VPNs rarely block UDP/53 to these)
# and hand Python curl `host:port:ip` resolve entries via PLUTUS_YF_RESOLVE (read by
# yfinance_source._pinned_session). Best-effort: a host no resolver can reach is skipped; if none
# resolve, the variable stays empty and Python falls back to its default session (system DNS) -- a
# transient failure the retry loop / next run rides out (recompute-from-inception).
$yfHosts = 'query1.finance.yahoo.com', 'query2.finance.yahoo.com', 'finance.yahoo.com'
$resolveEntries = @()
foreach ($h in $yfHosts) {
  foreach ($dns in '1.1.1.1', '8.8.8.8', '223.5.5.5') {
    try {
      $a = Resolve-DnsName -Server $dns -Name $h -Type A -DnsOnly -ErrorAction Stop |
        Where-Object { $_.Type -eq 'A' -and $_.IPAddress } | Select-Object -First 1
      if ($a) { $resolveEntries += "$($h):443:$($a.IPAddress)"; break }
    } catch { }
  }
}
if ($resolveEntries.Count -gt 0) {
  $env:PLUTUS_YF_RESOLVE = ($resolveEntries -join ',')
  "pinned Yahoo DNS -> $env:PLUTUS_YF_RESOLVE" | Out-File -FilePath $log -Append -Encoding utf8
}

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
  "=== run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (attempt $attempt/$maxAttempts) ===" |
    Out-File -FilePath $log -Append -Encoding utf8
  $out = [System.IO.Path]::GetTempFileName(); $err = [System.IO.Path]::GetTempFileName()
  $proc = Start-Process -FilePath $py -ArgumentList "`"$script`"" `
    -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
  Get-Content -LiteralPath $out, $err -Encoding UTF8 | Out-File -FilePath $log -Append -Encoding utf8
  Remove-Item -LiteralPath $out, $err -ErrorAction SilentlyContinue

  if ($proc.ExitCode -ne $EX_TEMPFAIL) { exit $proc.ExitCode }   # success (0) or fatal (!=75) -> done
  if ($attempt -lt $maxAttempts) {
    "transient data failure (exit 75); retrying in $delaySec s (attempt $attempt/$maxAttempts) ..." |
      Out-File -FilePath $log -Append -Encoding utf8
    Start-Sleep -Seconds $delaySec
  }
}
"retries exhausted ($maxAttempts attempts); yfinance never became reachable. The next scheduled run will catch up (recompute-from-inception)." |
  Out-File -FilePath $log -Append -Encoding utf8
exit $EX_TEMPFAIL
