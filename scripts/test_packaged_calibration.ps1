param(
    [string]$Executable = "dist/MultiCamera/MultiCamera.exe",
    [string]$Report = "installer_output/calibration_self_test.json",
    [string]$Baseline = "installer_output/calibration_source_test.json",
    [int]$TimeoutSeconds = 120
)
$ErrorActionPreference = "Stop"
$exePath = (Resolve-Path $Executable).Path
$reportPath = [System.IO.Path]::GetFullPath($Report)
$baselinePath = (Resolve-Path $Baseline).Path
if ($reportPath -eq $baselinePath) { throw "Report and baseline paths must differ" }
if (Test-Path $reportPath) { Remove-Item $reportPath -Force }
$workDir = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
New-Item -ItemType Directory -Path $workDir | Out-Null
$pythonPath = $env:PYTHONPATH
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    # A windowed EXE does not reliably block PowerShell's direct invocation.
    $process = Start-Process -FilePath $exePath -WorkingDirectory $workDir -PassThru `
        -ArgumentList @("--self-test-calibration", "--report", "`"$reportPath`"")
    if (!$process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill()
        $process.WaitForExit()
        throw "Packaged calibration self-test timed out after $TimeoutSeconds seconds"
    }
    $process.WaitForExit()
    if (!(Test-Path $reportPath)) { throw "EXE did not write a self-test report (exit $($process.ExitCode))" }
    $result = Get-Content $reportPath -Raw | ConvertFrom-Json
    if ($process.ExitCode -ne 0 -or $result.status -ne "passed" -or !$result.frozen) {
        Get-Content $reportPath -Raw | Write-Output
        throw "Packaged calibration failed (exit $($process.ExitCode))"
    }
    $source = Get-Content $baselinePath -Raw | ConvertFrom-Json
    if ($source.status -ne "passed" -or $source.frozen) { throw "Invalid source baseline" }
    $expectedVersion = python scripts/check_release.py --installed --print-version
    if ($LASTEXITCODE -ne 0) { throw "Release version validation failed" }
    if ($result.build_identity.version -ne $expectedVersion -or
        $source.build_identity.version -ne $expectedVersion -or
        !$source.build_identity.commit -or $source.build_identity.commit -eq "未知" -or
        $result.build_identity.commit -ne $source.build_identity.commit) {
        throw "Packaged version or commit differs from the tested source"
    }
    if ($env:GITHUB_ACTIONS -eq "true") {
        if ($result.build_identity.dirty -ne $false -or
            $source.build_identity.dirty -ne $false) {
            throw "Release identity must describe a clean checkout"
        }
        if ($env:GITHUB_REF_TYPE -eq "tag" -and
            $result.build_identity.tag -ne $env:GITHUB_REF_NAME) {
            throw "Packaged tag differs from the requested release tag"
        }
    }
    if ($result.opencv.version -ne $source.opencv.version -or
        $result.opencv.native_sha256 -ne $source.opencv.native_sha256) {
        throw "Packaged OpenCV version or native library differs from the tested build environment"
    }
    foreach ($api in $source.opencv.apis.PSObject.Properties) {
        if ($result.opencv.apis.($api.Name) -ne $api.Value) {
            throw "OpenCV interface changed while packaging: $($api.Name)"
        }
    }
    if ($result.opencv.distributions.'opencv-contrib-python' -ne
        $source.opencv.distributions.'opencv-contrib-python') {
        throw "OpenCV distribution metadata changed while packaging"
    }
    Write-Output "Frozen EXE calibration passed; OpenCV $($result.opencv.version)"
    Write-Output "Native library: $($result.opencv.native_path)"
    $result.opencv.apis | ConvertTo-Json | Write-Output
    $result.calibration | ConvertTo-Json | Write-Output
} finally {
    $env:PYTHONPATH = $pythonPath
    Remove-Item $workDir -Recurse -Force
}
