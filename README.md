### N4O Function for Vapoursynth

How to install:
```py
1. Download/Clone this repo
2. Extract the files
3. Copy n4ofunc.py to \Python36\Lib\site-packages
4. Load it, ex: import n4ofunc as nao
```

(stolen) Feature:
- Deband (mSdeband, Sdeband, Mdeband, Hdeband)
- SMDegrain (Nsmd, Hsmd)
- AA, using vsTAAmbk (Nnedi3taa, eedi3taa, Ntaa, Htaa)
- dehardsub
- IVTC, Decimate, Resize to 720p (animu) [aliases: animu, ivtc]
- hdr2sdr (experimental) [Hable Tonemapping Algorithm, should now provide better result] [Aliases: tonemap, hdr2sdr]
- benchmark (random things, just put n4ofunc.benchmark(src) and you're done) [Aliases: autistCode, benchmark]
- i444 meme (i444)
- descale_rescale [Aliases: rescale]
- Hybrid denoise (S_hybriddenoise, M_hybriddenoise, H_hybriddenoise)

Problem:
- Everything is a problem
- KNLM still broken
- Brokes after 1 preview (Kinda fixed)

Version rev:
- Version 1.0: first commit
- Version 1.1: Fix autistCode, added a little bit option for High SMDegrain
- Version 1.2: Some changes for High Deband, cleaned up script. Changed argument for i444
- Version 1.3: Added descale_rescale, reworked autistCode and renamed it to benchmark
- Version 1.3.1: Fixed KNLMeansCL, some tweak and clean up
- Version 1.4: Reworked hdr2sdr, now using hable tonemap algorithm that stolen from age@Doom9 with some additional command. Added some aliases
