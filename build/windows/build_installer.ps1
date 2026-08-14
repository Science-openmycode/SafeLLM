param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$Iscc = ""
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$webview = Join-Path $PSScriptRoot "vendor\WebView2"
if (-not (Test-Path -LiteralPath $webview)) {
    throw "WebView2 Fixed Runtime is missing. Place the unpacked x64 runtime in $webview"
}
if (-not $Iscc) {
    $localIscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    $machineIscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    $Iscc = if (Test-Path -LiteralPath $localIscc) { $localIscc } else { $machineIscc }
}
if (-not (Test-Path -LiteralPath $Iscc)) {
    throw "Inno Setup 6 compiler was not found: $Iscc"
}
Push-Location $repo
try {
    & $Python -m PyInstaller --noconfirm --clean "build\windows\yinbian.spec"
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
    $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"
    if (-not (Test-Path -LiteralPath $csc)) {
        $csc = Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe"
    }
    if (-not (Test-Path -LiteralPath $csc)) { throw ".NET Framework C# compiler was not found" }
    New-Item -ItemType Directory -Force "build\windows\bin" | Out-Null
    & $csc /nologo /target:exe /define:CONSOLE /reference:System.Web.Extensions.dll /out:"build\windows\bin\YinbianLauncherConsole.exe" "build\windows\YinbianLauncher.cs"
    if ($LASTEXITCODE -ne 0) { throw "console launcher build failed" }
    & $csc /nologo /target:winexe /reference:System.Web.Extensions.dll /reference:System.Windows.Forms.dll /out:"build\windows\bin\YinbianLauncherGui.exe" "build\windows\YinbianLauncher.cs"
    if ($LASTEXITCODE -ne 0) { throw "GUI launcher build failed" }
    & $Iscc "build\windows\YinbianZhimo.iss"
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
    $installer = Get-ChildItem "dist\installer\YinbianZhimo-1.0.0-Windows-x64-Offline.exe"
    if ($env:YINBIAN_SIGNTOOL -and $env:YINBIAN_SIGN_CERT_SHA1) {
        & $env:YINBIAN_SIGNTOOL sign /sha1 $env:YINBIAN_SIGN_CERT_SHA1 /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $installer.FullName
        if ($LASTEXITCODE -ne 0) { throw "Windows code signing failed" }
    }
    $hash = (Get-FileHash -LiteralPath $installer.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    Set-Content -LiteralPath ($installer.FullName + ".sha256") -Value "$hash  $($installer.Name)" -Encoding ascii
    Write-Output $installer.FullName
    Write-Output $hash
}
finally {
    Pop-Location
}
