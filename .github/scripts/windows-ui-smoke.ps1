param(
    [Parameter(Mandatory = $true)]
    [long]$WindowHandle,
    [Parameter(Mandatory = $true)]
    [int]$AppProcessId,
    [Parameter(Mandatory = $true)]
    [string]$MacroName,
    [Parameter(Mandatory = $true)]
    [string]$RenamedMacroName,
    [Parameter(Mandatory = $true)]
    [string]$MacroFilePath,
    [ValidateSet('setup', 'start-close-recording', 'verify-restart')]
    [string]$Mode = 'setup',
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes

function Get-Descendants {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root)

    try {
        return @($Root.FindAll(
            [System.Windows.Automation.TreeScope]::Descendants,
            [System.Windows.Automation.Condition]::TrueCondition
        ))
    }
    catch {
        return @()
    }
}

function Find-VisibleElement {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][System.Windows.Automation.ControlType]$ControlType,
        [switch]$RequireEnabled,
        [switch]$IncludeOffscreen
    )

    foreach ($element in @(Get-Descendants -Root $Root)) {
        try {
            $current = $element.Current
            if ($current.Name -ceq $Name -and $current.ControlType -eq $ControlType -and
                ($IncludeOffscreen -or -not $current.IsOffscreen)) {
                if ($RequireEnabled -and -not $current.IsEnabled) {
                    continue
                }
                return $element
            }
        }
        catch {
            # WebView2 can invalidate elements during a React rerender; the next query reacquires them.
        }
    }

    return $null
}

function Find-VisibleElementContainingName {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][System.Windows.Automation.ControlType]$ControlType
    )

    foreach ($element in @(Get-Descendants -Root $Root)) {
        try {
            $current = $element.Current
            if ($current.ControlType -eq $ControlType -and -not $current.IsOffscreen -and
                $current.Name.IndexOf($Name, [System.StringComparison]::Ordinal) -ge 0) {
                return $element
            }
        }
        catch {
            # Retry with a fresh tree after provider invalidation.
        }
    }

    return $null
}

function Wait-ForElement {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][System.Windows.Automation.ControlType]$ControlType,
        [Parameter(Mandatory = $true)][int]$Seconds,
        [switch]$RequireEnabled,
        [switch]$IncludeOffscreen
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
        $element = Find-VisibleElement -Root $Root -Name $Name -ControlType $ControlType `
            -RequireEnabled:$RequireEnabled -IncludeOffscreen:$IncludeOffscreen
        if ($null -ne $element) {
            return $element
        }
        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for UI Automation $($ControlType.ProgrammaticName) named '$Name'."
}

function Wait-ForElementContainingName {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][System.Windows.Automation.ControlType]$ControlType,
        [Parameter(Mandatory = $true)][int]$Seconds
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
        $element = Find-VisibleElementContainingName -Root $Root -Name $Name -ControlType $ControlType
        if ($null -ne $element) {
            return $element
        }
        Start-Sleep -Milliseconds 250
    }

    throw "Timed out waiting for a visible UI Automation $($ControlType.ProgrammaticName) containing '$Name'."
}

function Wait-ForElementContainingNameToDisappear {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][System.Windows.Automation.ControlType]$ControlType,
        [Parameter(Mandatory = $true)][int]$Seconds
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
        if ($null -eq (Find-VisibleElementContainingName -Root $Root -Name $Name -ControlType $ControlType)) {
            return
        }
        Start-Sleep -Milliseconds 250
    }

    throw "The previous UI Automation $($ControlType.ProgrammaticName) still contains '$Name'."
}

function Invoke-Element {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Element)

    $pattern = $Element.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke()
}

function Set-ElementValue {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Element,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $pattern = $Element.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
    $pattern.SetValue($Value)
}

function Read-MacroFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    $macro = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    $updatedAtText = [string]$macro.updated_at
    if ([string]::IsNullOrWhiteSpace($updatedAtText)) {
        throw "Macro file '$Path' has no updated_at value."
    }
    try {
        $updatedAt = [DateTimeOffset]::Parse(
            $updatedAtText,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::AssumeUniversal
        )
    }
    catch {
        throw "Macro file '$Path' has an invalid updated_at value '$updatedAtText'."
    }

    return [pscustomobject]@{ Data = $macro; UpdatedAt = $updatedAt }
}

function Get-NameDiagnostics {
    param([Parameter(Mandatory = $true)][System.Windows.Automation.AutomationElement]$Root)

    $namesByType = @{
        'ControlType.Button' = @()
        'ControlType.Edit' = @()
        'ControlType.ListItem' = @()
        'ControlType.Text' = @()
    }
    foreach ($element in @(Get-Descendants -Root $Root)) {
        try {
            $current = $element.Current
            $typeName = $current.ControlType.ProgrammaticName
            if ($namesByType.ContainsKey($typeName) -and
                -not [string]::IsNullOrWhiteSpace($current.Name)) {
                $visibility = if ($current.IsOffscreen) { ' [offscreen]' } else { '' }
                $namesByType[$typeName] += "$($current.Name)$visibility"
            }
        }
        catch {
            # Diagnostics are best-effort while the page is rerendering.
        }
    }

    $parts = foreach ($typeName in @('ControlType.Button', 'ControlType.Edit', 'ControlType.ListItem', 'ControlType.Text')) {
        $names = @($namesByType[$typeName] | Select-Object -Unique | Select-Object -First 12)
        "$typeName=[$($names -join ' | ')]"
    }
    return "Visible UI Automation names: $($parts -join '; ')."
}

$window = $null
try {
    $accentedE = [char]0x00E9
    $accentedA = [char]0x00E0
    $accentedCircumflexE = [char]0x00EA
    $curlyApostrophe = [char]0x2019
    $createLabel = "Cr$($accentedE)er une macro"
    $createSubmitLabel = "Cr$($accentedE)er la macro"
    $readyLabel = 'Pr' + $accentedCircumflexE + 't ' + $accentedA + ' lancer'
    $eventCountLabel = '0 ' + $accentedE + 'v' + $accentedE + 'nements'
    $renamedMacroEventCountLabel = $RenamedMacroName + ' ' + $eventCountLabel
    $stopRecordingLabel = 'Arr' + $accentedCircumflexE + 'ter l' + $curlyApostrophe + 'enregistrement'
    $window = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$WindowHandle)
    if ($window.Current.ProcessId -ne $AppProcessId) {
        throw "The UI Automation window belongs to process $($window.Current.ProcessId), expected AUTO-COC process $AppProcessId."
    }

    if ($Mode -eq 'verify-restart') {
        $null = Wait-ForElement -Root $window -Name $RenamedMacroName `
            -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
        $null = Wait-ForElementContainingName -Root $window -Name $RenamedMacroName `
            -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
        Wait-ForElementContainingNameToDisappear -Root $window -Name $MacroName `
            -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
        $null = Wait-ForElement -Root $window -Name $readyLabel `
            -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
        $null = Wait-ForElementContainingName -Root $window -Name $renamedMacroEventCountLabel `
            -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
        Write-Output "UI Automation confirmed '$RenamedMacroName' reloaded after a full restart with an empty sequence."
        return
    }

    if ($Mode -eq 'start-close-recording') {
        $macroBeforeRecording = Read-MacroFile -Path $MacroFilePath
        if ([string]$macroBeforeRecording.Data.name -cne $RenamedMacroName -or
            @($macroBeforeRecording.Data.steps).Count -ne 0) {
            throw 'The renamed macro must be empty before starting the close-during-recording scenario.'
        }

        $record = Wait-ForElement -Root $window -Name 'Enregistrer' `
            -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
        Invoke-Element -Element $record
        $null = Wait-ForElementContainingName -Root $window -Name 'Capture des actions en cours' `
            -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds ($TimeoutSeconds + 10)
        Start-Sleep -Seconds 4
        $null = Wait-ForElement -Root $window -Name $stopRecordingLabel `
            -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds `
            -RequireEnabled -IncludeOffscreen
        Write-Output 'UI Automation left the empty recording active for the graceful-close scenario.'
        return
    }

    $create = Wait-ForElement -Root $window -Name $createLabel `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    Invoke-Element -Element $create

    $nameInput = Wait-ForElement -Root $window -Name 'Nom de la macro' `
        -ControlType ([System.Windows.Automation.ControlType]::Edit) -Seconds $TimeoutSeconds -RequireEnabled
    Set-ElementValue -Element $nameInput -Value $MacroName
    $createSubmit = Wait-ForElement -Root $window -Name $createSubmitLabel `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    Invoke-Element -Element $createSubmit

    $null = Wait-ForElement -Root $window -Name $MacroName `
        -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
    $null = Wait-ForElementContainingName -Root $window -Name $MacroName `
        -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
    Write-Output "UI Automation confirmed the new macro '$MacroName' in the atelier and macro list."

    $rename = Wait-ForElement -Root $window -Name 'Renommer' `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    Invoke-Element -Element $rename
    $renameInput = Wait-ForElement -Root $window -Name 'Nouveau nom' `
        -ControlType ([System.Windows.Automation.ControlType]::Edit) -Seconds $TimeoutSeconds -RequireEnabled
    Set-ElementValue -Element $renameInput -Value $RenamedMacroName
    $renameSubmit = Wait-ForElement -Root $window -Name 'Enregistrer le nom' `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    Invoke-Element -Element $renameSubmit

    $null = Wait-ForElement -Root $window -Name $RenamedMacroName `
        -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
    $null = Wait-ForElementContainingName -Root $window -Name $RenamedMacroName `
        -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
    Wait-ForElementContainingNameToDisappear -Root $window -Name $MacroName `
        -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
    Write-Output "UI Automation confirmed the renamed macro '$RenamedMacroName' and its removal under the old name."

    $record = Wait-ForElement -Root $window -Name 'Enregistrer' `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    $macroBeforeRecording = Read-MacroFile -Path $MacroFilePath
    if ([string]$macroBeforeRecording.Data.name -cne $RenamedMacroName) {
        throw 'The macro file name does not match the renamed macro before recording.'
    }
    Invoke-Element -Element $record
    $null = Wait-ForElementContainingName -Root $window -Name 'Capture des actions en cours' `
        -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds ($TimeoutSeconds + 10)
    Start-Sleep -Seconds 4
    $stopRecording = Wait-ForElement -Root $window -Name $stopRecordingLabel `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds `
        -RequireEnabled -IncludeOffscreen
    Invoke-Element -Element $stopRecording
    $null = Wait-ForElement -Root $window -Name $readyLabel `
        -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
    $null = Wait-ForElementContainingName -Root $window -Name $renamedMacroEventCountLabel `
        -ControlType ([System.Windows.Automation.ControlType]::ListItem) -Seconds $TimeoutSeconds
    $macroAfterRecording = Read-MacroFile -Path $MacroFilePath
    if ($macroAfterRecording.UpdatedAt -le $macroBeforeRecording.UpdatedAt) {
        throw 'The macro updated_at timestamp did not advance after stopping the empty recording.'
    }
    if (@($macroAfterRecording.Data.steps).Count -ne 0) {
        throw 'The isolated macro file should contain no events after the recording without injected input.'
    }
    $play = Wait-ForElement -Root $window -Name 'Lire la macro' `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds
    if ($play.Current.IsEnabled) {
        throw 'The empty recording unexpectedly enabled macro playback.'
    }
    Write-Output 'UI Automation confirmed recording start and stop; updated_at advanced and the persisted sequence is empty.'
}
catch {
    [System.Console]::WriteLine("Windows UI Automation smoke failed: $($_.Exception.Message)")
    if ($null -ne $window) {
        [System.Console]::WriteLine((Get-NameDiagnostics -Root $window))
    }
    exit 1
}
