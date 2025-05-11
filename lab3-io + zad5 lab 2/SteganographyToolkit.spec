# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('zad1.py', '.'), ('zad2.py', '.'), ('zad3.py', '.'), ('zad4.py', '.'), ('zad5.py', '.'), ('iminim.py', '.')],
    hiddenimports=['numpy', 'cv2', 'matplotlib', 'matplotlib.pyplot', 'scipy', 'scipy.fftpack', 'PIL', 'binascii', 'math'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SteganographyToolkit',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
