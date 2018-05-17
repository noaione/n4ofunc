### N4O Function for Vapoursynth

How to install:
```py
1. Download/Clone this repo
2. Extract the files
3. Copy n4ofunc.py to \Python36\Lib\site-packages
4. Load it at vapoursynth, ex: import n4ofunc as nao
```

(stolen) Feature:
- Deband (mSdeband, Sdeband, Mdeband, Hdeband)
- KNLM (Mknlm, Hknlm) #Not Working Properly
- BM3D (Nbm3d, Hbm3d)
- SMDegrain (Nsmd, Hsmd)
- AA, using vsTAAmbk (Nnedi3taa, eedi3taa, Ntaa, Htaa)
- deblock
- dehardsub
- IVTC, Decimate, Resize to 720p (animu)
- hdr2sdr (experimental)
- benchmark (random shit)
- i444 meme (i444)
- descale_rescale
- Hybrid denoise (S_hybriddenoise, M_hybriddenoise, H_hybriddenoise)

Problem:
- Everything is a problem
- KNLM not working properly
- Brokes after 1 preview (Kinda fixed)

Version rev:
- Version 1.0: first commit
- Version 1.1: Fix autistCode, added a little bit option for High SMDegrain
- Version 1.2: Some changes for High Deband, cleaned up script. Changed argument for i444
- Version 1.3: Added descale_rescale, reworked autistCode and renamed it to benchmark