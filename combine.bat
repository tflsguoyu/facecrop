mkdir out
for /f "delims=" %%p in ('dir /b/ad') do move %%p\*.* out

pause