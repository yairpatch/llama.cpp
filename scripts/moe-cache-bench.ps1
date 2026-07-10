#Requires -Version 5.1
<#
Interleaved A/B benchmark for the MoE VRAM expert cache (--moe-cache).

Compares two llama-server configurations with everything else held equal:
  static : auto-fit static expert offload only (no extra flags)
  hybrid : auto-fit + --moe-cache <MiB> managing the host-resident layers

Configs alternate every round (A B A B ...) so thermal / power drift affects
both equally. Each server start gets a warmup request (fills the cache and
the thermal state) and one measured request; only the measured request
counts. Requests use temperature 0 and a fixed length, so the workload is
identical across runs.

Usage (from the directory containing llama-server.exe, or pass -Exe):
  powershell -ExecutionPolicy Bypass -File moe-cache-bench.ps1 -Model "C:\path\to\model.gguf"

Optional: -Rounds 3 -NPredict 2000 -Ctx 28000 -MoeCacheMiB 1500 -Port 8090

Before running: plug in, set the Windows power plan to best performance,
close background GPU users. Expect ~5-6 minutes per round at ~50 t/s.
If the script is interrupted, kill any leftover server with:
  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process

Notes on the output:
  - 'hash' is an MD5 prefix of the generated text. Within the static config
    it must be identical across rounds (temp 0). The hybrid config computes
    cached experts on the GPU, so its text can differ from static (and can
    vary slightly between rounds as the cached set evolves) - expected.
  - results.csv and per-run server logs land in a timestamped directory.
#>
param(
    [Parameter(Mandatory = $true)] [string] $Model,
    [string] $Exe         = ".\llama-server.exe",
    [int]    $Rounds      = 3,
    [int]    $NPredict    = 2000,
    [int]    $Ctx         = 28000,
    [int]    $Port        = 8090,
    [int]    $CooldownSec = 20,
    [int]    $MoeCacheMiB = 1500
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Model)) { throw "model not found: $Model" }
if (-not (Test-Path $Exe))   { throw "llama-server not found: $Exe (pass -Exe or run from build\bin\Release)" }

$configs = @(
    @{ Name = "static"; Args = @();                              Env = @{} },
    @{ Name = "hybrid"; Args = @("--moe-cache", "$MoeCacheMiB"); Env = @{ MOE_CACHE_DUP_IDS = "1"; MOE_CACHE_VERBOSE = "1" } }
)

$prompt = "Write a detailed, multi-chapter story about a lighthouse keeper who discovers a hidden library beneath the sea. Include dialogue, rich descriptions, and several plot twists."

$resultsDir = Join-Path (Get-Location) ("moe-bench-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $resultsDir | Out-Null

$moeEnvNames = @("MOE_CACHE_DUP_IDS","MOE_CACHE_VERBOSE","MOE_CACHE_INTERVAL","MOE_CACHE_DECAY","MOE_CACHE_PROMOTE_MB","MOE_CACHE_MARGIN")

function Clear-MoeEnv {
    foreach ($n in $moeEnvNames) {
        Remove-Item -Path "Env:$n" -ErrorAction SilentlyContinue
    }
}

function Wait-Server {
    param([int] $TimeoutSec = 300)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    throw "server did not become healthy within $TimeoutSec s"
}

function Invoke-Gen {
    $body = @{ prompt = $prompt; n_predict = $NPredict; temperature = 0; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800
    $md5  = [System.Security.Cryptography.MD5]::Create()
    $hash = [System.BitConverter]::ToString($md5.ComputeHash([System.Text.Encoding]::UTF8.GetBytes([string]$r.content))).Replace("-","").Substring(0,8)
    [pscustomobject]@{
        tps  = [math]::Round([double]$r.timings.predicted_per_second, 2)
        n    = [int]$r.timings.predicted_n
        hash = $hash
    }
}

$rows = @()

for ($round = 1; $round -le $Rounds; $round++) {
    foreach ($cfg in $configs) {
        Clear-MoeEnv
        foreach ($kv in $cfg.Env.GetEnumerator()) { Set-Item -Path "Env:$($kv.Key)" -Value $kv.Value }

        $tag = "$($cfg.Name)-r$round"
        $srvArgs = @("-m", "`"$Model`"", "--no-mmap", "-c", "$Ctx", "--port", "$Port") + $cfg.Args
        Write-Host ""
        Write-Host ("[{0}] starting: {1} {2}" -f $tag, $Exe, ($srvArgs -join " "))

        $proc = Start-Process -FilePath $Exe -ArgumentList $srvArgs -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $resultsDir "$tag.out.log") `
            -RedirectStandardError  (Join-Path $resultsDir "$tag.err.log")
        try {
            Wait-Server
            Write-Host "[$tag] warmup request ($NPredict tokens)..."
            $warm = Invoke-Gen
            Write-Host ("[$tag] warmup:   {0} t/s" -f $warm.tps)
            Write-Host "[$tag] measured request..."
            $meas = Invoke-Gen
            Write-Host ("[$tag] measured: {0} t/s ({1} tokens, hash {2})" -f $meas.tps, $meas.n, $meas.hash)
            $rows += [pscustomobject]@{
                round = $round; config = $cfg.Name
                warm_tps = $warm.tps; tps = $meas.tps; n = $meas.n; hash = $meas.hash
            }
        } finally {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Wait-Process -Id $proc.Id -Timeout 30 -ErrorAction SilentlyContinue
        }
        Clear-MoeEnv
        Write-Host "[$tag] cooldown ${CooldownSec}s..."
        Start-Sleep -Seconds $CooldownSec
    }
}

$rows | Export-Csv -Path (Join-Path $resultsDir "results.csv") -NoTypeInformation

Write-Host ""
Write-Host "=== raw results (measured request only) ==="
$rows | Format-Table round, config, warm_tps, tps, n, hash -AutoSize | Out-String | Write-Host

Write-Host "=== summary ==="
$summary = $rows | Group-Object config | ForEach-Object {
    $t = $_.Group.tps | Measure-Object -Average -Minimum -Maximum
    [pscustomobject]@{
        config = $_.Name; runs = $_.Count
        mean = [math]::Round($t.Average, 2); min = $t.Minimum; max = $t.Maximum
    }
}
$summary | Format-Table -AutoSize | Out-String | Write-Host

$a = $summary | Where-Object { $_.config -eq "static" }
$b = $summary | Where-Object { $_.config -eq "hybrid" }
if ($a -and $b -and $a.mean -gt 0) {
    $delta = [math]::Round(100.0 * ($b.mean - $a.mean) / $a.mean, 1)
    Write-Host ("hybrid vs static mean: {0}%" -f $delta)
    if     ($b.min -gt $a.max) { Write-Host "verdict: hybrid clearly faster (ranges do not overlap)" }
    elseif ($a.min -gt $b.max) { Write-Host "verdict: static clearly faster (ranges do not overlap)" }
    else                       { Write-Host "verdict: ranges overlap - difference is within run-to-run noise" }
}

Write-Host ""
Write-Host "server logs and results.csv: $resultsDir"
