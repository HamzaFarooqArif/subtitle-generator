# Build a synthetic test video with known ground-truth speech.
#
# Uses the Windows SAPI voices so the repo needs no checked-in media and no
# personal footage to validate the pipeline end to end. The fixture
# deliberately includes a quiet passage and a noise-only passage, which are the
# two cases the gating stage exists to handle.

param(
    [string]$OutDir = "$PSScriptRoot\fixtures"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Add-Type -AssemblyName System.Speech

$script = @'
Good morning. Today we are going to walk down to the river and look at the old bridge.
The weather forecast said it would rain later, so we should probably bring a jacket with us.
My brother Thomas is going to meet us there at about four o clock in the afternoon.
'@

$speechWav = Join-Path $OutDir "speech.wav"
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
Write-Host "Installed SAPI voices: $($voices -join ', ')"
$synth.SetOutputToWaveFile($speechWav)
$synth.Rate = 0
$synth.Speak($script)
$synth.SetOutputToNull()
$synth.Dispose()
Write-Host "wrote $speechWav"

# Resolve ffmpeg the same way the package does.
$ffmpeg = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Source
if (-not $ffmpeg) {
    $ffmpeg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ffmpeg) { throw "ffmpeg not found" }

$fixture = Join-Path $OutDir "sample.mp4"

# Layout: 1 s lead-in silence, the speech at full level, 3 s of low-level
# noise (the hallucination trap), the speech again at -22 dB (the "whispered"
# case), then 1 s of silence.
$graph = "[0:a]aformat=sample_rates=16000:channel_layouts=mono,asplit=2[s1][s2];" +
         "[s1]adelay=1000|1000[a0];" +
         "[s2]volume=-22dB[quiet];" +
         "[1:a]aformat=sample_rates=16000:channel_layouts=mono[noise];" +
         "[a0][noise][quiet]concat=n=3:v=0:a=1,apad=pad_dur=1[aout]"

& $ffmpeg -hide_banner -loglevel error -y `
    -i $speechWav `
    -f lavfi -i "anoisesrc=color=pink:amplitude=0.02:duration=3" `
    -f lavfi -i "color=c=black:s=320x240:r=5" `
    -filter_complex $graph `
    -map "[aout]" -map "2:v" -shortest `
    -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 128k `
    $fixture
if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed with exit code $LASTEXITCODE" }

Write-Host "wrote $fixture"

# Ground truth for the assertion in test_pipeline_e2e.py
$script | Set-Content -Path (Join-Path $OutDir "sample.groundtruth.txt") -Encoding utf8
Write-Host "wrote ground truth"
