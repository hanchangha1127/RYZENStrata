@echo off
rem Build and package the Windows x64 RYZENStrata engine for gfx1151.
rem Requires CMake, Ninja, Python 3.10+, Git, and Visual Studio C++ Build Tools.
rem The ROCm 10.2 toolchain is installed in a private environment (ROCM_VENV).
rem Run with 'tests' to build the HIP unit tests as well.
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..\..") do set "SRC=%%~fI"
set "STRATA_HIP_ARCHS=gfx1151"
if not defined STRATA_ROCM_VERSION set "STRATA_ROCM_VERSION=10.2.0a20260930"
if not defined STRATA_ROCM_INDEX set "STRATA_ROCM_INDEX=https://nightly.repo.amd.com/rocm/whl-next/"
if not defined ROCM_VENV set "ROCM_VENV=%SRC%\.rocm-win"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-hip-win"
if not defined DIST_DIR set "DIST_DIR=%SRC%\dist"
set "TESTS=OFF"
if /i "%~1"=="tests" set "TESTS=ON"

rem ---- 1. MSVC: the linker, the C runtime and the Windows SDK (ROCm's clang uses them)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS="
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do set "VS=%%V"
if not defined VS (echo Visual Studio Build Tools with the C++ workload are needed & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
where cmake >nul 2>nul || (echo CMake is needed & exit /b 1)
where ninja >nul 2>nul || (echo Ninja is needed & exit /b 1)

rem ---- 2. ROCm (TheRock wheels: the compiler, the HIP runtime, hipBLAS/hipBLASLt/rocBLAS, a device package per arch)
set "PY="
py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY (echo Python 3.10+ is needed & exit /b 1)
if not exist "%ROCM_VENV%\Scripts\python.exe" %PY% -m venv "%ROCM_VENV%" || exit /b 1
set "EXTRAS=libraries,devel"
for %%A in (%STRATA_HIP_ARCHS:;= %) do set "EXTRAS=!EXTRAS!,device-%%A"
set "STAMP=%ROCM_VENV%\strata-rocm.txt"
set "WANT=%STRATA_ROCM_VERSION% %EXTRAS%"
set "HAVE="
if exist "%STAMP%" set /p HAVE=<"%STAMP%"
if not "!HAVE!"=="!WANT!" if not "%STRATA_ROCM_KEEP%"=="1" (
  echo Installing ROCm %STRATA_ROCM_VERSION% [%EXTRAS%] into %ROCM_VENV% ...
  "%ROCM_VENV%\Scripts\python.exe" -m pip install --disable-pip-version-check --index-url "%STRATA_ROCM_INDEX%" "rocm[%EXTRAS%]==%STRATA_ROCM_VERSION%" || exit /b 1
  "%ROCM_VENV%\Scripts\rocm-sdk.exe" init || exit /b 1
  >"%STAMP%" echo !WANT!
)
for /f "delims=" %%R in ('"%ROCM_VENV%\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not exist "%ROCM%\lib\llvm\bin\clang++.exe" (echo ROCm has no compiler in "%ROCM%" & exit /b 1)
set "ROCM_F=%ROCM:\=/%"
set "BITCODE=%ROCM_F%/lib/llvm/amdgcn/bitcode"

set "HIP_PLATFORM=amd"
set "HIP_PATH=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%PATH%"

rem ---- 3. configure and build the gfx1151 engine
  cmake -G Ninja -S "%SRC%\engine" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release ^
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=%TESTS% ^
    -DSTRATA_BUILD_SPLIT=OFF ^
    -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_HIP_ARCHITECTURES=%STRATA_HIP_ARCHS%" ^
    "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" ^
    "-DCMAKE_HIP_COMPILER=%ROCM_F%/lib/llvm/bin/clang++.exe" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=%ROCM_F%" ^
    "-DCMAKE_PREFIX_PATH=%ROCM_F%" "-DCMAKE_HIP_FLAGS=--rocm-path=%ROCM_F% --rocm-device-lib-path=%BITCODE%" || exit /b 1
if "%TESTS%"=="ON" (
  cmake --build "%BUILD_DIR%" || exit /b 1
) else (
  cmake --build "%BUILD_DIR%" --target strata strata-device || exit /b 1
)

rem ---- 4. the zip: the programs, the ROCm DLLs they load (+ rocBLAS/hipBLASLt kernels for these archs), licenses
"%ROCM_VENV%\Scripts\python.exe" "%SRC%\terminal\tools\hip\package_windows.py" --build "%BUILD_DIR%" --rocm "%ROCM%" ^
  --archs "%STRATA_HIP_ARCHS%" --rocm-version "%STRATA_ROCM_VERSION%" --out "%DIST_DIR%" || exit /b 1
echo.
echo Done: %DIST_DIR%\strata-windows-x64-hip.zip
endlocal
