param(
    [string]$Scenes = "scripts/video/privacy_demo_scenes.json",
    [string]$Output = "artifacts/video/yinbian-privacy-demo/audio"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ".").Path
$scenePath = (Resolve-Path $Scenes).Path
$outputPath = Join-Path $root $Output
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

Add-Type -AssemblyName System.Speech
$items = Get-Content $scenePath -Raw -Encoding UTF8 | ConvertFrom-Json
$durations = @()
foreach ($item in $items) {
    $wav = Join-Path $outputPath ($item.id + ".wav")
    $speaker = [System.Speech.Synthesis.SpeechSynthesizer]::new()
    try {
        $voices = @($speaker.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name })
        $voice = $null
        foreach ($candidate in @("Microsoft Huihui Desktop", "Microsoft Huihui", "Microsoft Yaoyao", "Microsoft Kangkang")) {
            if ($voices -contains $candidate) { $voice = $candidate; break }
        }
        if (-not $voice) { throw "No Chinese SAPI voice is available" }
        $speaker.SelectVoice($voice)
        $speaker.Rate = 2
        $speaker.Volume = 100
        $speaker.SetOutputToWaveFile($wav)
        $speaker.Speak([string]$item.narration)
    }
    finally {
        $speaker.Dispose()
    }
    $duration = [double](& .venv\Scripts\python.exe -c "import sys,wave; f=wave.open(sys.argv[1],'rb'); print(f.getnframes()/f.getframerate()); f.close()" $wav)
    $duration = [Math]::Round($duration, 3)
    $durations += [pscustomobject]@{
        id = [string]$item.id
        title = [string]$item.title
        caption = [string]$item.caption
        narration = [string]$item.narration
        audio = $wav
        duration_seconds = $duration
    }
}
$durations | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $outputPath "scenes.generated.json") -Encoding UTF8
Write-Output ("Generated " + $durations.Count + " narration clips")
Write-Output ("Narration seconds: " + [Math]::Round(($durations | Measure-Object duration_seconds -Sum).Sum, 2))
