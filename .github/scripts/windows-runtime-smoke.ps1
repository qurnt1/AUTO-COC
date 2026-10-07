param(
    [Parameter(Mandatory = $true)]
    [string]$BundleDirectory
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Start-SmokeProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Arguments = '',
        [string]$WorkingDirectory = (Get-Location).Path,
        [switch]$Hidden,
        [switch]$CaptureOutput,
        [hashtable]$Environment = @{}
    )

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Path
    $startInfo.Arguments = $Arguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $Hidden.IsPresent
    $startInfo.WindowStyle = if ($Hidden) {
        [System.Diagnostics.ProcessWindowStyle]::Hidden
    }
    else {
        [System.Diagnostics.ProcessWindowStyle]::Normal
    }
    if ($CaptureOutput) {
        $startInfo.RedirectStandardOutput = $true
        $startInfo.StandardOutputEncoding = [System.Text.UTF8Encoding]::new()
        $startInfo.RedirectStandardError = $true
        $startInfo.StandardErrorEncoding = [System.Text.UTF8Encoding]::new()
    }
    foreach ($entry in $Environment.GetEnumerator()) {
        $startInfo.EnvironmentVariables[$entry.Key] = $entry.Value
    }

    $process = [System.Diagnostics.Process]::Start($startInfo)
    return $process
}

function Get-AppProcessIds {
    param([Parameter(Mandatory = $true)][string]$ExecutablePath)

    $processName = [System.IO.Path]::GetFileNameWithoutExtension($ExecutablePath)
    return @(
        Get-Process -Name $processName -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -and $_.Path -ieq $ExecutablePath } |
            Select-Object -ExpandProperty Id
    )
}

function Get-ExternalBrowserProcesses {
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'chrome.exe' OR Name = 'msedge.exe'" |
            Select-Object @{ Name = 'Id'; Expression = { [int]$_.ProcessId } }, Name
    )
}

function Get-NormalizedFullPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
}

function Write-CapturedProcessOutput {
    param(
        [Parameter(Mandatory = $true)]$StandardOutputTask,
        [Parameter(Mandatory = $true)]$StandardErrorTask
    )

    $standardOutput = $StandardOutputTask.GetAwaiter().GetResult()
    $standardError = $StandardErrorTask.GetAwaiter().GetResult()
    if (-not [string]::IsNullOrWhiteSpace($standardOutput)) {
        Write-Host $standardOutput.TrimEnd()
    }
    if (-not [string]::IsNullOrWhiteSpace($standardError)) {
        Write-Warning $standardError.TrimEnd()
    }
}

function Stop-SmokeProcess {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$Label
    )

    try {
        $Process.Refresh()
        if ($Process.HasExited) {
            return $true
        }

        if ($Process.MainWindowHandle -ne [IntPtr]::Zero) {
            [void]$Process.CloseMainWindow()
        }
        if ($Process.WaitForExit(5000)) {
            return $true
        }

        Write-Warning "$Label did not exit after a close request; terminating its process tree."
        $Process.Kill($true)
        if ($Process.WaitForExit(5000)) {
            Write-Warning "$Label required forced termination."
            return $true
        }

        Write-Warning "$Label is still running after forced termination and the 5-second wait."
        return $false
    }
    catch {
        Write-Warning "Could not confirm that $Label stopped: $($_.Exception.Message)"
        return $false
    }
}

function Wait-ForMainWindow {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $lastHandle = [IntPtr]::Zero
    $lastTitle = ''
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "AUTO-COC exited before its main window appeared (exit $($Process.ExitCode))."
        }

        try {
            $lastHandle = $Process.MainWindowHandle
            $lastTitle = $Process.MainWindowTitle
        }
        catch [System.InvalidOperationException] {
            Start-Sleep -Milliseconds 250
            continue
        }

        if ($lastHandle -ne [IntPtr]::Zero -and $lastTitle -eq 'AUTO-COC') {
            return $lastHandle
        }
        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for AUTO-COC HWND/title; handle=$lastHandle title='$lastTitle'."
}

$runnerTemp = $env:RUNNER_TEMP
if ([string]::IsNullOrWhiteSpace($runnerTemp)) {
    throw 'RUNNER_TEMP is required for the isolated desktop smoke test.'
}

$bundlePath = (Resolve-Path -LiteralPath $BundleDirectory).Path
$installers = @(Get-ChildItem -LiteralPath $bundlePath -Filter '*-setup.exe' -File)
if ($installers.Count -ne 1) {
    throw "Expected one NSIS setup executable in '$bundlePath'; found $($installers.Count)."
}

$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$tauriConfig = Get-Content -LiteralPath (Join-Path $repositoryRoot 'frontend/src-tauri/tauri.conf.json') -Raw |
    ConvertFrom-Json
$tauriIdentifier = [string]$tauriConfig.identifier
if ($tauriIdentifier -notmatch '^[A-Za-z0-9.-]+$') {
    throw "Unexpected Tauri identifier '$tauriIdentifier'; refusing to derive a filesystem path."
}

$knownLocalAppData = [System.Environment]::GetFolderPath(
    [System.Environment+SpecialFolder]::LocalApplicationData
)
if ([string]::IsNullOrWhiteSpace($knownLocalAppData) -or -not [System.IO.Directory]::Exists($knownLocalAppData)) {
    throw 'Windows Known Folder LocalApplicationData is unavailable.'
}
$tauriAppLocalDataPath = Join-Path $knownLocalAppData $tauriIdentifier
$existingTauriData = [System.IO.Directory]::GetFileSystemEntries($knownLocalAppData, $tauriIdentifier)
if ($existingTauriData.Count -gt 0) {
    throw "Tauri app-local-data path already exists; refusing to touch it: '$tauriAppLocalDataPath'."
}

$smokeFolderName = "auto-coc-runtime-smoke-$([Guid]::NewGuid().ToString('N'))"
$existingSmokeRoot = [System.IO.Directory]::GetFileSystemEntries($runnerTemp, $smokeFolderName)
if ($existingSmokeRoot.Count -gt 0) {
    throw "Smoke scratch path already exists; refusing to touch it: '$(Join-Path $runnerTemp $smokeFolderName)'."
}

$smokeRoot = Join-Path $runnerTemp $smokeFolderName
$installDir = Join-Path $smokeRoot 'install'
$localAppData = Join-Path $smokeRoot 'local-app-data'
$webViewData = Join-Path $smokeRoot 'webview2'
$tauriAppLocalDataTarget = Join-Path $smokeRoot 'tauri-local-data'
$mainProcess = $null
$secondProcess = $null
$installerProcess = $null
$uninstallerProcess = $null
$uiAutomationProcess = $null
$junctionCreated = $false
$allSmokeProcessesStopped = $true
$junctionSafeForTargetCleanup = $true

try {
    New-Item -ItemType Directory -Path $smokeRoot | Out-Null
    New-Item -ItemType Directory -Path $localAppData, $webViewData, $tauriAppLocalDataTarget | Out-Null
    $junction = New-Item -ItemType Junction -Path $tauriAppLocalDataPath -Target $tauriAppLocalDataTarget
    $junctionCreated = $true
    $junctionSafeForTargetCleanup = $false
    $junctionTarget = [System.IO.Path]::GetFullPath([string]$junction.Target)
    if ($junction.LinkType -ne 'Junction' -or $junctionTarget -ine [System.IO.Path]::GetFullPath($tauriAppLocalDataTarget)) {
        throw "Could not verify Tauri app-data junction at '$tauriAppLocalDataPath'."
    }
    Write-Host "Redirected Tauri app-local-data to RUNNER_TEMP: $tauriAppLocalDataTarget."
    Write-Host "Installing $($installers[0].Name) into RUNNER_TEMP."
    $appEnvironment = @{
        LOCALAPPDATA = $localAppData
        WEBVIEW2_USER_DATA_FOLDER = $webViewData
    }

    # NSIS requires /D= to be the final, unquoted argument.
    $installerProcess = Start-SmokeProcess -Path $installers[0].FullName `
        -Arguments "/S /D=$installDir" -Hidden -Environment $appEnvironment
    if (-not $installerProcess.WaitForExit(180000)) {
        $installerProcess.Kill($true)
        if (-not $installerProcess.WaitForExit(5000)) {
            Write-Warning 'NSIS installer remained active 5 seconds after forced termination.'
        }
        throw 'NSIS installation timed out after 180 seconds.'
    }
    if ($installerProcess.ExitCode -ne 0) {
        throw "NSIS installation failed with exit code $($installerProcess.ExitCode)."
    }

    $appExecutables = @(
        Get-ChildItem -LiteralPath $installDir -Filter '*.exe' -File -Recurse |
            Where-Object { $_.Name -notmatch '(?i)uninstall' }
    )
    if ($appExecutables.Count -ne 1) {
        throw "Expected one installed app executable; found $($appExecutables.Count)."
    }
    $appPath = $appExecutables[0].FullName
    $browserPidsBefore = @(Get-ExternalBrowserProcesses | Select-Object -ExpandProperty Id)

    Write-Host "Starting installed app: $($appExecutables[0].Name)."
    $mainProcess = Start-SmokeProcess -Path $appPath -WorkingDirectory $installDir `
        -Environment $appEnvironment
    $mainWindow = Wait-ForMainWindow -Process $mainProcess -TimeoutSeconds 60
    Write-Host "Main window found: PID $($mainProcess.Id), HWND $mainWindow, title AUTO-COC."

    $uiAutomationScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'windows-ui-smoke.ps1')).Path
    $windowsPowerShell = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
    $macroSuffix = [Guid]::NewGuid().ToString('N').Substring(0, 12)
    $macroName = "CI-SMOKE-$macroSuffix"
    $renamedMacroName = "CI-RENAMED-$macroSuffix"
    $uiAutomationArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$uiAutomationScript`" -WindowHandle $($mainWindow.ToInt64()) -AppProcessId $($mainProcess.Id) -MacroName $macroName -RenamedMacroName $renamedMacroName"
    Write-Host 'Exercising macro creation and rename through Windows UI Automation.'
    $uiAutomationProcess = Start-SmokeProcess -Path $windowsPowerShell -Arguments $uiAutomationArguments `
        -WorkingDirectory $smokeRoot -Hidden -CaptureOutput
    $uiOutputTask = $uiAutomationProcess.StandardOutput.ReadToEndAsync()
    $uiErrorTask = $uiAutomationProcess.StandardError.ReadToEndAsync()
    if (-not $uiAutomationProcess.WaitForExit(120000)) {
        Write-Warning 'Windows UI Automation smoke timed out after 120 seconds; terminating its process tree.'
        $uiAutomationProcess.Refresh()
        if (-not $uiAutomationProcess.HasExited) {
            try {
                $uiAutomationProcess.Kill($true)
            }
            catch [System.InvalidOperationException] {
                $uiAutomationProcess.Refresh()
                if (-not $uiAutomationProcess.HasExited) {
                    Write-Warning 'The UI Automation helper could not be terminated after its timeout.'
                }
            }
        }
        if ($uiAutomationProcess.WaitForExit(5000)) {
            Write-CapturedProcessOutput -StandardOutputTask $uiOutputTask -StandardErrorTask $uiErrorTask
        }
        else {
            Write-Warning 'Windows UI Automation helper remained active 5 seconds after forced termination.'
        }
        throw 'Windows UI Automation smoke timed out after 120 seconds.'
    }
    Write-CapturedProcessOutput -StandardOutputTask $uiOutputTask -StandardErrorTask $uiErrorTask
    if ($uiAutomationProcess.ExitCode -ne 0) {
        throw "Windows UI Automation smoke failed with exit code $($uiAutomationProcess.ExitCode)."
    }

    $macroDirectory = Join-Path (Join-Path $localAppData 'AUTO-COC') 'macros'
    $oldMacroFile = Join-Path $macroDirectory "$macroName.json"
    $renamedMacroFile = Join-Path $macroDirectory "$renamedMacroName.json"
    if ((Test-Path -LiteralPath $oldMacroFile) -or -not (Test-Path -LiteralPath $renamedMacroFile)) {
        throw 'The Rust backend did not persist the expected macro rename in the isolated profile.'
    }
    $persistedMacro = Get-Content -LiteralPath $renamedMacroFile -Raw | ConvertFrom-Json
    if ([string]$persistedMacro.name -cne $renamedMacroName) {
        throw 'The isolated macro file does not contain the renamed macro name.'
    }
    Write-Host 'Rust backend persistence confirmed: only the renamed macro file exists in the isolated profile.'

    Write-Host 'Starting a second instance.'
    $secondProcess = Start-SmokeProcess -Path $appPath -WorkingDirectory $installDir `
        -Environment $appEnvironment
    if (-not $secondProcess.WaitForExit(15000)) {
        throw 'Second instance did not exit within 15 seconds.'
    }
    if ($secondProcess.ExitCode -ne 0) {
        throw "Second instance exited with code $($secondProcess.ExitCode)."
    }
    $mainProcess.Refresh()
    if ($mainProcess.HasExited) {
        throw 'The first instance exited after the second launch.'
    }

    $appPids = @(Get-AppProcessIds -ExecutablePath $appPath)
    if ($appPids.Count -ne 1 -or $appPids[0] -ne $mainProcess.Id) {
        throw "Expected one AUTO-COC process ($($mainProcess.Id)); found: $($appPids -join ', ')."
    }
    $windowAfterSecondLaunch = Wait-ForMainWindow -Process $mainProcess -TimeoutSeconds 5
    if ($windowAfterSecondLaunch -ne $mainWindow) {
        throw "The first instance changed its main HWND after the second launch."
    }

    $newBrowsers = @(
        Get-ExternalBrowserProcesses |
            Where-Object { $_.Id -notin $browserPidsBefore }
    )
    if ($newBrowsers.Count -gt 0) {
        throw "External browser process(es) started: $(($newBrowsers | ForEach-Object { "$($_.Name) PID $($_.Id)" }) -join ', ')."
    }
    Write-Host 'No new Chrome or Edge browser process detected (WebView2 is excluded).'

    if (-not $mainProcess.CloseMainWindow()) {
        throw 'CloseMainWindow did not send a close request to AUTO-COC.'
    }
    if (-not $mainProcess.WaitForExit(30000)) {
        throw 'AUTO-COC did not exit within 30 seconds after CloseMainWindow.'
    }
    if ($mainProcess.ExitCode -ne 0) {
        throw "AUTO-COC exited with code $($mainProcess.ExitCode) after graceful close."
    }
    Write-Host 'AUTO-COC closed cleanly.'
}
finally {
    foreach ($entry in @(
        @{ Label = 'Windows UI Automation helper'; Process = $uiAutomationProcess },
        @{ Label = 'AUTO-COC second instance'; Process = $secondProcess },
        @{ Label = 'AUTO-COC main instance'; Process = $mainProcess },
        @{ Label = 'NSIS installer'; Process = $installerProcess }
    )) {
        if ($null -ne $entry.Process) {
            if (-not (Stop-SmokeProcess -Process $entry.Process -Label $entry.Label)) {
                $allSmokeProcessesStopped = $false
            }
            $entry.Process.Dispose()
        }
    }

    if ($allSmokeProcessesStopped -and (Test-Path -LiteralPath $installDir)) {
        $uninstallers = @(
            Get-ChildItem -LiteralPath $installDir -Filter '*uninstall*.exe' -File -Recurse
        )
        if ($uninstallers.Count -eq 1) {
            try {
                $uninstallerProcess = Start-SmokeProcess -Path $uninstallers[0].FullName `
                    -Arguments "/S _?=$installDir" -WorkingDirectory $installDir -Hidden
                if (-not $uninstallerProcess.WaitForExit(60000)) {
                    Write-Warning 'NSIS uninstaller timed out after 60 seconds.'
                    if (-not (Stop-SmokeProcess -Process $uninstallerProcess -Label 'NSIS uninstaller')) {
                        $allSmokeProcessesStopped = $false
                    }
                }
                elseif ($uninstallerProcess.ExitCode -ne 0) {
                    Write-Warning "NSIS uninstaller exited with code $($uninstallerProcess.ExitCode)."
                }
                else {
                    Write-Host 'NSIS uninstaller completed.'
                }
            }
            catch {
                Write-Warning "NSIS uninstall cleanup failed: $($_.Exception.Message)"
                if ($null -ne $uninstallerProcess) {
                    $uninstallerProcess.Refresh()
                    if (-not $uninstallerProcess.HasExited -and
                        -not (Stop-SmokeProcess -Process $uninstallerProcess -Label 'NSIS uninstaller')) {
                        $allSmokeProcessesStopped = $false
                    }
                }
            }
            finally {
                if ($null -ne $uninstallerProcess) {
                    if (-not $uninstallerProcess.HasExited -and
                        -not (Stop-SmokeProcess -Process $uninstallerProcess -Label 'NSIS uninstaller')) {
                        $allSmokeProcessesStopped = $false
                    }
                    $uninstallerProcess.Dispose()
                }
            }
        }
        elseif ($uninstallers.Count -eq 0) {
            Write-Warning 'No NSIS uninstaller was found; the installed app could not be uninstalled.'
        }
        else {
            Write-Warning "Found $($uninstallers.Count) NSIS uninstallers; uninstall was skipped because the target is ambiguous."
        }
    }
    elseif (-not $allSmokeProcessesStopped) {
        Write-Warning 'Skipping NSIS uninstall and scratch cleanup because a smoke process may still be running.'
    }
    else {
        Write-Warning 'No install directory exists; no NSIS uninstall could be run.'
    }

    if ($allSmokeProcessesStopped -and $junctionCreated) {
        try {
            $junctionItem = Get-Item -Force -LiteralPath $tauriAppLocalDataPath -ErrorAction SilentlyContinue
            if ($null -eq $junctionItem) {
                Write-Warning 'The temporary Tauri app-data junction is already absent.'
                $junctionSafeForTargetCleanup = $true
            }
            else {
                $junctionTarget = Get-NormalizedFullPath -Path ([string]$junctionItem.Target)
                if ($junctionItem.LinkType -ne 'Junction' -or
                    $junctionTarget -ine (Get-NormalizedFullPath -Path $tauriAppLocalDataTarget)) {
                    throw "Refusing to remove a path that is no longer the smoke junction: '$tauriAppLocalDataPath'."
                }

                # Directory.Delete removes the junction itself; it does not traverse its target.
                [System.IO.Directory]::Delete($tauriAppLocalDataPath, $false)
                $junctionSafeForTargetCleanup = $true
                Write-Host 'Removed the temporary Tauri app-data junction.'
            }
        }
        catch {
            Write-Warning "Could not remove the temporary Tauri app-data junction: $($_.Exception.Message)"
        }
    }

    if ($allSmokeProcessesStopped -and $junctionSafeForTargetCleanup) {
        $tempRoot = Get-NormalizedFullPath -Path $runnerTemp
        $smokeRootFull = Get-NormalizedFullPath -Path $smokeRoot
        $tauriTargetFull = Get-NormalizedFullPath -Path $tauriAppLocalDataTarget
        if (-not $smokeRootFull.StartsWith($tempRoot.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
            -not $tauriTargetFull.StartsWith($smokeRootFull.TrimEnd('\') + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing cleanup outside the smoke scratch root: '$smokeRootFull'."
        }

        $targetSafeForRootCleanup = -not (Test-Path -LiteralPath $tauriTargetFull)
        if (Test-Path -LiteralPath $tauriTargetFull) {
            try {
                $targetAttributes = [System.IO.File]::GetAttributes($tauriTargetFull)
                if (($targetAttributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "Refusing recursive cleanup of a reparse-point target: '$tauriTargetFull'."
                }
                Remove-Item -LiteralPath $tauriTargetFull -Recurse -Force
                $targetSafeForRootCleanup = $true
                Write-Host 'Removed the isolated Tauri app-local-data target.'
            }
            catch {
                Write-Warning "Could not remove isolated Tauri app-local-data: $($_.Exception.Message)"
            }
        }

        if ($targetSafeForRootCleanup -and (Test-Path -LiteralPath $smokeRootFull)) {
            try {
                Remove-Item -LiteralPath $smokeRootFull -Recurse -Force
            }
            catch {
                Write-Warning "Could not remove smoke files under RUNNER_TEMP: $($_.Exception.Message)"
            }
        }
        elseif (-not $targetSafeForRootCleanup) {
            Write-Warning 'Leaving the scratch root in RUNNER_TEMP because the isolated app-data target could not be safely removed.'
        }
    }
    elseif (-not $allSmokeProcessesStopped) {
        Write-Warning "Leaving smoke scratch and Tauri app-data junction in place because a process may still use '$tauriAppLocalDataTarget'; the hosted runner will be discarded."
    }
    else {
        Write-Warning "Leaving smoke scratch in RUNNER_TEMP because the Tauri junction could not be safely removed."
    }
}
