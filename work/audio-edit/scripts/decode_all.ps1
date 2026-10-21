# 解码 6 个 m4a -> 16k mono s16le wav
$ErrorActionPreference = 'Stop'
$root = 'C:\projects\local\audio_processer'
$wavdir = Join-Path $root 'work\audio-edit\wav'
New-Item -ItemType Directory -Force -Path $wavdir | Out-Null
$map = @(
  @{ short = 'xfzx'; name = '0_20140227-丁耘 西方哲学(20140227-丁耘 西方哲学)_normal..m4a' },
  @{ short = 'lxg1'; name = '20_丁耘理想国.古希腊哲学(20141219-丁耘  古希腊哲学)_normal..m4a' },
  @{ short = 'xes';  name = '24_【复旦哲学系丁耘】 亚里士多德《形而上学》(01)_normal..m4a' },
  @{ short = 'lxg4'; name = '4_丁耘理想国.古希腊哲学(第四卷)_normal..m4a' },
  @{ short = 'wlx';  name = '【复旦哲学系丁耘】 亚里士多德《物理学》(05-20100401)_5_normal..m4a' },
  @{ short = 'ly';   name = '【复旦哲学系丁耘】孔子《论语》(06)_19_normal..m4a' }
)
$map | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $root 'work\audio-edit\wav\filemap.json') -Encoding utf8
foreach ($m in $map) {
  $src = Join-Path $root ('data\' + $m.name)
  $dst = Join-Path $wavdir ($m.short + '.wav')
  if (Test-Path $dst) { Write-Host ("[skip] " + $m.short); continue }
  $t0 = Get-Date
  & ffmpeg -hide_banner -loglevel error -y -i $src -vn -ac 1 -ar 16000 -c:a pcm_s16le $dst
  $dt = ((Get-Date) - $t0).TotalSeconds
  $sz = (Get-Item $dst).Length / 1MB
  Write-Host ("[done] {0} {1:N1}s -> {2:N0} MB (ffmpeg {3:N1}s)" -f $m.short, ((Get-Item $dst).Length/32000), $sz, $dt)
}
Write-Host 'ALL DECODE DONE'
