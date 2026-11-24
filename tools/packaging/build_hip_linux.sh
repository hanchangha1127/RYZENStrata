#!/bin/bash
# Package the Linux x86-64 RYZENStrata HIP engine and its gfx1151 runtime.
# Set HP to the ROCm root. CORE and LIBW can point at split TheRock roots.
# Set OUT to a work directory, or NO_PACKAGE=1 to stop after staging.
set -u
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
HP=${HP:?HP=<ROCm devel root>}
CORE=${CORE:-$HP}
LIBW=${LIBW:-$HP}
OUT=${OUT:-$PWD/work-halo}
PKGVER=${PKGVER:-0.1.67}
COMMIT=${COMMIT:-unknown}
JOBS=${JOBS:-16}
V=halo
ARCHS=gfx1151
mkdir -p "$OUT"; S=$OUT/build.status
st() { echo "$* $(date +%T)" >> "$S"; }
# a step that failed stops the build with its non-zero exit (a lost exit code let stale binaries through before): chk $? "what"
chk() { local rc=$1; shift; st "$* rc=$rc"; [ "$rc" = 0 ] || { st "ABORT $* failed (rc=$rc)"; exit 1; }; }
# a tree without .git (a tarball) has no commit for the build id: the packer writes it to BUILD_ID (CMakeLists.txt reads it), first 9 characters
ID9=${COMMIT:0:9}
if [ "$COMMIT" != unknown ] && [ ! -e "$SRC/.git" ] && [ ! -f "$SRC/BUILD_ID" ]; then printf '%s\n' "$ID9" > "$SRC/BUILD_ID"; fi
: > "$S"
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
# TheRock 7.14.x dists (the Strix Halo / gfx1151 package) keep libnuma, libz, libdrm ... in lib/rocm_sysdeps; older wheels have no such folder
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
CLANG=$HP/lib/llvm/bin/clang++
[ -x "$CLANG" ] || CLANG=$HP/llvm/bin/clang++
HIPDEFS=("-DCMAKE_HIP_COMPILER=$CLANG" "-DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP" "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")
st "start $V ($ARCHS) commit $COMMIT"
cd "$SRC" || exit 1
cmake -G Ninja -S engine -B "$OUT/b" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_SPLIT=OFF "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg.log" 2>&1
chk $? "engine configured"
nice -n 10 cmake --build "$OUT/b" --target strata -j "$JOBS" > "$OUT/build.log" 2>&1
chk $? "engine built"
[ -x "$OUT/b/strata" ] || { st "ABORT no strata"; exit 1; }

STAGE=$OUT/stage
rm -rf "$STAGE"
mkdir -p "$STAGE"
cat > "$STAGE/BUILD.json" <<EOJ
{"version": "$PKGVER", "source": "RYZENStrata $COMMIT", "variant": "$V", "archs": ["$ARCHS"], "portable": true, "vision": "none"}
EOJ
DATA=(--data "$LIBW/lib/rocblas/library=rocblas/library" --data "$LIBW/lib/hipblaslt/library=hipblaslt/library" --data-arch "$ARCHS")
LLVMTARGET="$CORE/lib/llvm/lib/$(uname -m)-unknown-linux-gnu"
LLVMSEARCH=(--search "$CORE/lib/llvm/lib")
[ -d "$LLVMTARGET" ] && LLVMSEARCH+=(--search "$LLVMTARGET")   # ROCm 10.2+ LLVM 24 per-target runtime directory
SEARCH=(--search "$LIBW/lib" --search "$HP/lib" --search "$CORE/lib" "${LLVMSEARCH[@]}")
[ -z "$SYSDEPS" ] || SEARCH+=(--search "$HP/lib/rocm_sysdeps/lib")
python3 tools/packaging/collect_libs.py --binary "$OUT/b/strata" "${SEARCH[@]}" \
  --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --out "$STAGE" "${DATA[@]}" > "$OUT/collect.log" 2>&1
chk $? "collected engine libraries"
# #5060: a ROCm whose rocBLAS / hipBLASLt keep their device code in kpack archives (ROCm 10.x: <rocm>/.kpack, found by the libraries at ../.kpack
# from their own folder, which is outside the package): the blas_lib archives of these archs go to kpack/ in the package, and the engine and the GPU
# reader name them in ROCM_KPACK_PATH (engine/include/strata/core/kpack_path.hpp).  rocSOLVER's solver_lib is left out (Strata launches none of its kernels;
# terminal/tools/hip/package_windows.py does the same).  TheRock's wheels (the release builds) embed the device code: no .kpack, nothing copied.
KPDIRS=()
for r in "$(dirname "$(readlink -f "$LIBW/lib/librocblas.so.5" 2>/dev/null || echo "$LIBW/lib/x")")/.." "$HP" "$CORE" "$LIBW"; do
  [ -d "$r/.kpack" ] && KPDIRS+=("$(readlink -f "$r/.kpack")")
done
if [ ${#KPDIRS[@]} -gt 0 ]; then
  mkdir -p "$STAGE/kpack"; nkp=0
  for a in $ARCHS; do
    f=""; for d in "${KPDIRS[@]}"; do [ -f "$d/blas_lib_$a.kpack" ] && { f="$d/blas_lib_$a.kpack"; break; }; done
    if [ -n "$f" ]; then cp "$f" "$STAGE/kpack/"; nkp=$((nkp+1)); else st "kpack: no blas_lib_$a.kpack in ${KPDIRS[0]}"; fi
  done
  st "kpack: $nkp blas_lib archives copied to kpack/ from ${KPDIRS[0]}"
fi
cp "$OUT/b/strata" "$STAGE/" || exit 1
for f in "$STAGE"/*.so "$STAGE"/*.so.*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done
st "RUNPATH strata: $(readelf -d "$STAGE/strata" | grep -E 'RPATH|RUNPATH' | cut -d'[' -f2 | tr -d ']' | paste -sd' ')"
# the hipBLASLt version of this ROCm into BUILD.json, and the tuning tables of terminal/tools/hip that match it (architecture and version) into tuning/
python3 tools/packaging/stage_tuning.py --stage "$STAGE" --archs "$ARCHS" --rocm "$HP" >> "$OUT/stage_tuning.log" 2>&1
trc=$?; st "tuning tables rc=$trc: $(tail -n 1 "$OUT/stage_tuning.log")"; [ "$trc" = 0 ] || { st "ABORT tuning tables"; exit 1; }
[ -n "${NO_PACKAGE:-}" ] && { st "STAGE-DONE"; exit 0; }
rm -rf "$OUT/components"; mkdir -p "$OUT/components/engine"
python3 tools/packaging/package_engine.py --src "$STAGE" --backend hip --version "$PKGVER" --out "$OUT/components/engine" \
  --arch "$ARCHS" --variant "$V" --build-id "$COMMIT+rocm" --tree --notes "RYZENStrata gfx1151 engine" > "$OUT/package.log" 2>&1
chk $? "packaged"
cat "$OUT/package.log" >> "$S"
# every library a bundled file needs is in the package or on binary_deps.py's short base-system list (glibc, libgcc_s, libstdc++);
# collect_libs.py bundles libatomic.so.1 (librocm_sysdeps_numa.so.1 needs it) with its notice in licenses/
for d in "$OUT/components/engine"/*/; do python3 tools/packaging/binary_deps.py "$d" >> "$S" 2>&1; chk $? "imports covered in $(basename "$d")"; done
st "BUILD-DONE"
