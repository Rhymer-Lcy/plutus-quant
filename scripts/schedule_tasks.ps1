# Manage the Plutus scheduled task (version-controlled definition so it is reproducible & portable).
# One daily job:
#   plutus-paper-forward  09:30 Beijing  -> paper_forward.ps1  (net-payout free-data forward record)
#
# WHY 09:30 BEIJING (not local): the operator keeps Beijing hours, and 09:30 Beijing reliably has the
# prior US session's FINALIZED daily bar (US close ~04:00-05:00 Beijing), while today's US bar has not
# begun -- so yfinance returns a clean, complete close (running at US-open would risk grabbing today's
# still-forming bar). The trigger is therefore ANCHORED to +08:00: it fires at the same absolute instant
# regardless of the machine's timezone or US DST. (Windows preserves an offset-qualified StartBoundary as
# an absolute instant; a floating local time -- what a bare `schtasks /ST 09:30` stores -- would instead
# drift when the OS timezone changed, which is exactly the bug this replaces.)
# StartWhenAvailable: a run missed because the PC was off/asleep fires once on the next boot/wake (one
# catch-up suffices -- paper_forward recomputes the whole curve from inception, so nothing is lost).
#
# Interpreter: a scheduled task runs with no activated conda env, so paper_forward.ps1 must be told which
# python runs `plutus`. `register` resolves it (explicit PLUTUS_PYTHON, else the active conda env, else
# PATH), VERIFIES it can `import plutus`, and persists it as the user-scope PLUTUS_PYTHON env var the
# wrapper reads -- so no interpreter path is hardcoded in source. Run `register` from an activated plutus
# env, or set PLUTUS_PYTHON first.
#
# Usage (paths derived from this script's location):
#   powershell -ExecutionPolicy Bypass -File scripts\schedule_tasks.ps1 register   # (re)create (idempotent)
#   powershell ... schedule_tasks.ps1 status     # state + last result + next run
#   powershell ... schedule_tasks.ps1 disable    # PAUSE (keep definition)
#   powershell ... schedule_tasks.ps1 enable     # RESUME
#   powershell ... schedule_tasks.ps1 remove     # DELETE
# Run once now:  Start-ScheduledTask -TaskName plutus-paper-forward
param([ValidateSet('register', 'status', 'disable', 'enable', 'remove')][string]$action = 'status')
$ErrorActionPreference = 'Stop'

$TaskName = 'plutus-paper-forward'
$Anchor = '2025-01-01T09:30:00+08:00'   # 09:30 Beijing, offset-qualified => fires at that absolute instant

function Resolve-PlutusPython {
  # Locate + VERIFY the interpreter that runs the plutus package (a wrong one fails every run). Order:
  # explicit PLUTUS_PYTHON, then the active conda env, then PATH. Returns the verified path; throws if none.
  $candidates = @()
  if ($env:PLUTUS_PYTHON) { $candidates += $env:PLUTUS_PYTHON }
  if ($env:CONDA_PREFIX) { $candidates += (Join-Path $env:CONDA_PREFIX 'python.exe') }
  $onPath = (Get-Command python -ErrorAction SilentlyContinue).Source
  if ($onPath) { $candidates += $onPath }
  $candidates = $candidates | Select-Object -Unique
  foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) {
      & $c -c 'import plutus' 2>$null
      if ($LASTEXITCODE -eq 0) { return (Resolve-Path $c).Path }
    }
  }
  throw ("No Python interpreter that can 'import plutus' was found (tried: " + ($candidates -join ', ') +
    "). Activate the plutus env or set PLUTUS_PYTHON to its python.exe, then re-run register.")
}

$wrapper = Join-Path $PSScriptRoot 'paper_forward.ps1'
switch ($action) {
  'register' {
    $py = Resolve-PlutusPython
    [Environment]::SetEnvironmentVariable('PLUTUS_PYTHON', $py, 'User')
    $env:PLUTUS_PYTHON = $py
    "configured PLUTUS_PYTHON = $py (user scope; paper_forward.ps1 reads this -- no interpreter path hardcoded)"
    $a = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper`""
    $trig = New-ScheduledTaskTrigger -Daily -At '09:30'
    $trig.StartBoundary = $Anchor          # re-anchor to 09:30 Beijing (absolute instant, DST-proof)
    $p = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
    $s = New-ScheduledTaskSettingsSet -StartWhenAvailable
    Register-ScheduledTask -TaskName $TaskName -Action $a -Trigger $trig -Principal $p -Settings $s `
      -Description 'Plutus daily forward record (fires 09:30 Beijing; StartWhenAvailable catch-up)' -Force | Out-Null
    $next = (Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime
    "registered $TaskName @ 09:30 Beijing (anchored $Anchor); next run (local) = $next"
  }
  'disable' { Disable-ScheduledTask -TaskName $TaskName | Out-Null; "disabled (paused) $TaskName" }
  'enable'  { Enable-ScheduledTask -TaskName $TaskName | Out-Null; "enabled $TaskName" }
  'remove'  { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; "removed $TaskName" }
  'status'  {
    $cfg = [Environment]::GetEnvironmentVariable('PLUTUS_PYTHON', 'User')
    "PLUTUS_PYTHON (user) = $(if ($cfg) { $cfg } else { '(unset -- run register)' })"
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($t) {
      $i = Get-ScheduledTaskInfo -TaskName $TaskName
      "{0}: state {1}, startBoundary {2}, last {3} (result {4}), next {5}" -f `
        $TaskName, $t.State, $t.Triggers[0].StartBoundary, $i.LastRunTime, $i.LastTaskResult, $i.NextRunTime
    } else { "$TaskName not registered -- run: schedule_tasks.ps1 register" }
  }
}
