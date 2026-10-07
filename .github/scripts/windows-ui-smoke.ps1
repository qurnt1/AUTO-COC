param(
    [Parameter(Mandatory = $true)]
    [long]$WindowHandle,
    [Parameter(Mandatory = $true)]
    [int]$AppProcessId,
    [Parameter(Mandatory = $true)]
    [string]$MacroName,
    [Parameter(Mandatory = $true)]
    [string]$RenamedMacroName,
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

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
        [switch]$RequireEnabled
    )

    foreach ($element in @(Get-Descendants -Root $Root)) {
        try {
            $current = $element.Current
            if ($current.Name -ceq $Name -and $current.ControlType -eq $ControlType -and -not $current.IsOffscreen) {
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
        [switch]$RequireEnabled
    )

    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $Seconds) {
        $element = Find-VisibleElement -Root $Root -Name $Name -ControlType $ControlType -RequireEnabled:$RequireEnabled
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
            if (-not $current.IsOffscreen -and $namesByType.ContainsKey($typeName) -and
                -not [string]::IsNullOrWhiteSpace($current.Name)) {
                $namesByType[$typeName] += $current.Name
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
    $window = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$WindowHandle)
    if ($window.Current.ProcessId -ne $AppProcessId) {
        throw "The UI Automation window belongs to process $($window.Current.ProcessId), expected AUTO-COC process $AppProcessId."
    }

    $null = Wait-ForElement -Root $window -Name 'Vos macros' `
        -ControlType ([System.Windows.Automation.ControlType]::Text) -Seconds $TimeoutSeconds
    Write-Output "UI Automation found the accessible view label 'Vos macros'."

    $create = Wait-ForElement -Root $window -Name 'Créer une macro' `
        -ControlType ([System.Windows.Automation.ControlType]::Button) -Seconds $TimeoutSeconds -RequireEnabled
    Invoke-Element -Element $create

    $nameInput = Wait-ForElement -Root $window -Name 'Nom de la macro' `
        -ControlType ([System.Windows.Automation.ControlType]::Edit) -Seconds $TimeoutSeconds -RequireEnabled
    Set-ElementValue -Element $nameInput -Value $MacroName
    $createSubmit = Wait-ForElement -Root $window -Name 'Créer la macro' `
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
}
catch {
    [System.Console]::Error.WriteLine("Windows UI Automation smoke failed: $($_.Exception.Message)")
    if ($null -ne $window) {
        [System.Console]::Error.WriteLine((Get-NameDiagnostics -Root $window))
    }
    exit 1
}
