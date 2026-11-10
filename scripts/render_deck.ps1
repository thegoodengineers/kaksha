# Export a .pptx to PDF and one PNG per slide with PowerPoint. Usage: render_deck.ps1 <pptx> <outdir>
param([string]$pptx, [string]$out)
$pptx = (Resolve-Path $pptx).Path
New-Item -ItemType Directory -Force $out | Out-Null
$out = (Resolve-Path $out).Path
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($pptx, $true, $false, $false)
$p.SaveAs([IO.Path]::ChangeExtension($pptx, ".pdf"), 32)
$i = 1
foreach ($s in $p.Slides) { $s.Export((Join-Path $out ("slide{0}.png" -f $i)), "PNG", 1600, 900); $i++ }
$p.Close(); $app.Quit()
