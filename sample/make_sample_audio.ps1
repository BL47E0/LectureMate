# Creates sample/sample_lecture.wav from sample/sample_lecture.txt using the
# text-to-speech voice that is built into Windows. No internet or install needed.
#
# Run from the LectureMate folder:
#   powershell -ExecutionPolicy Bypass -File sample\make_sample_audio.ps1

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$textFile = Join-Path $root "sample_lecture.txt"
$wavFile = Join-Path $root "sample_lecture.wav"

$text = Get-Content -Path $textFile -Raw -Encoding UTF8
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.SetOutputToWaveFile($wavFile)
$synth.Speak($text)
$synth.Dispose()

Write-Host "Created: $wavFile"
