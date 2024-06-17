[//]: # (Project: Assera)
[//]: # (Version: v1.2)

# Assera v1.2 Reference

## `assera.Target.Model`

Defines constants for some well-known CPU models.

<!--
NOTE: This table should be in sync with assera/python/assera/Targets.py.
To generate the table entries:
1. Re-build Assera
2. cd build/lib.xxx
3. Run this code snippet and replace the entries in the tables:

from assera.Targets import KNOWN_DEVICES, _MODEL_TRANSLATION_DICT, Target

devices = KNOWN_DEVICES[Target.Category.CPU]
for d in sorted(devices): 
    print(f"`assera.Target.Model.{d.upper().translate(_MODEL_TRANSLATION_DICT)}` | {d}")

print("\n\n")
devices = KNOWN_DEVICES[Target.Category.GPU]
for d in sorted(devices): 
    print(f"`assera.Target.Model.{d.upper().translate(_MODEL_TRANSLATION_DICT)}` | {d}")
-->

type | description
--- | ---
`assera.Target.Model.AMD_1200` | AMD 1200
`assera.Target.Model.AMD_1300X` | AMD 1300X
`assera.Target.Model.AMD_1400` | AMD 1400
`assera.Target.Model.AMD_1500X` | AMD 1500X
`assera.Target.Model.AMD_1600` | AMD 1600
`assera.Target.Model.AMD_1600X` | AMD 1600X
`assera.Target.Model.AMD_1700` | AMD 1700
`assera.Target.Model.AMD_1700X` | AMD 1700X
`assera.Target.Model.AMD_1800X` | AMD 1800X
`assera.Target.Model.AMD_1900X` | AMD 1900X
`assera.Target.Model.AMD_1920X` | AMD 1920X
`assera.Target.Model.AMD_1950X` | AMD 1950X
`assera.Target.Model.AMD_200GE` | AMD 200GE
`assera.Target.Model.AMD_2200G` | AMD 2200G
`assera.Target.Model.AMD_2200GE` | AMD 2200GE
`assera.Target.Model.AMD_2200U` | AMD 2200U
`assera.Target.Model.AMD_220GE` | AMD 220GE
`assera.Target.Model.AMD_2300U` | AMD 2300U
`assera.Target.Model.AMD_2300X` | AMD 2300X
`assera.Target.Model.AMD_2400G` | AMD 2400G
`assera.Target.Model.AMD_2400GE` | AMD 2400GE
`assera.Target.Model.AMD_240GE` | AMD 240GE
`assera.Target.Model.AMD_2500U` | AMD 2500U
`assera.Target.Model.AMD_2500X` | AMD 2500X
`assera.Target.Model.AMD_2600` | AMD 2600
`assera.Target.Model.AMD_2600E` | AMD 2600E
`assera.Target.Model.AMD_2600H` | AMD 2600H
`assera.Target.Model.AMD_2600X` | AMD 2600X
`assera.Target.Model.AMD_2700` | AMD 2700
`assera.Target.Model.AMD_2700E` | AMD 2700E
`assera.Target.Model.AMD_2700U` | AMD 2700U
`assera.Target.Model.AMD_2700X` | AMD 2700X
`assera.Target.Model.AMD_2700X_GOLD_EDITION` | AMD 2700X Gold Edition
`assera.Target.Model.AMD_2800H` | AMD 2800H
`assera.Target.Model.AMD_2920X` | AMD 2920X
`assera.Target.Model.AMD_2950X` | AMD 2950X
`assera.Target.Model.AMD_2970WX` | AMD 2970WX
`assera.Target.Model.AMD_2990WX` | AMD 2990WX
`assera.Target.Model.AMD_3000G` | AMD 3000G
`assera.Target.Model.AMD_300U` | AMD 300U
`assera.Target.Model.AMD_3050U` | AMD 3050U
`assera.Target.Model.AMD_3101` | AMD 3101
`assera.Target.Model.AMD_3150U` | AMD 3150U
`assera.Target.Model.AMD_3151` | AMD 3151
`assera.Target.Model.AMD_3200G` | AMD 3200G
`assera.Target.Model.AMD_3200U` | AMD 3200U
`assera.Target.Model.AMD_3201` | AMD 3201
`assera.Target.Model.AMD_3250U` | AMD 3250U
`assera.Target.Model.AMD_3251` | AMD 3251
`assera.Target.Model.AMD_3255` | AMD 3255
`assera.Target.Model.AMD_3300U` | AMD 3300U
`assera.Target.Model.AMD_3301` | AMD 3301
`assera.Target.Model.AMD_3351` | AMD 3351
`assera.Target.Model.AMD_3400G` | AMD 3400G
`assera.Target.Model.AMD_3401` | AMD 3401
`assera.Target.Model.AMD_3451` | AMD 3451
`assera.Target.Model.AMD_3500` | AMD 3500
`assera.Target.Model.AMD_3500U` | AMD 3500U
`assera.Target.Model.AMD_3500X` | AMD 3500X
`assera.Target.Model.AMD_3550H` | AMD 3550H
`assera.Target.Model.AMD_3580U` | AMD 3580U
`assera.Target.Model.AMD_3600` | AMD 3600
`assera.Target.Model.AMD_3600X` | AMD 3600X
`assera.Target.Model.AMD_3600XT` | AMD 3600XT
`assera.Target.Model.AMD_3700U` | AMD 3700U
`assera.Target.Model.AMD_3700X` | AMD 3700X
`assera.Target.Model.AMD_3750H` | AMD 3750H
`assera.Target.Model.AMD_3780U` | AMD 3780U
`assera.Target.Model.AMD_3800X` | AMD 3800X
`assera.Target.Model.AMD_3800XT` | AMD 3800XT
`assera.Target.Model.AMD_3900` | AMD 3900
`assera.Target.Model.AMD_3900X` | AMD 3900X
`assera.Target.Model.AMD_3900XT` | AMD 3900XT
`assera.Target.Model.AMD_3950X` | AMD 3950X
`assera.Target.Model.AMD_3960X` | AMD 3960X
`assera.Target.Model.AMD_3970X` | AMD 3970X
`assera.Target.Model.AMD_3980X` | AMD 3980X
`assera.Target.Model.AMD_3990X` | AMD 3990X
`assera.Target.Model.AMD_4300G` | AMD 4300G
`assera.Target.Model.AMD_4300GE` | AMD 4300GE
`assera.Target.Model.AMD_4300U` | AMD 4300U
`assera.Target.Model.AMD_4500U` | AMD 4500U
`assera.Target.Model.AMD_4600G` | AMD 4600G
`assera.Target.Model.AMD_4600GE` | AMD 4600GE
`assera.Target.Model.AMD_4600H` | AMD 4600H
`assera.Target.Model.AMD_4600HS` | AMD 4600HS
`assera.Target.Model.AMD_4600U` | AMD 4600U
`assera.Target.Model.AMD_4680U` | AMD 4680U
`assera.Target.Model.AMD_4700G` | AMD 4700G
`assera.Target.Model.AMD_4700GE` | AMD 4700GE
`assera.Target.Model.AMD_4700U` | AMD 4700U
`assera.Target.Model.AMD_4800H` | AMD 4800H
`assera.Target.Model.AMD_4800HS` | AMD 4800HS
`assera.Target.Model.AMD_4800U` | AMD 4800U
`assera.Target.Model.AMD_4900H` | AMD 4900H
`assera.Target.Model.AMD_4900HS` | AMD 4900HS
`assera.Target.Model.AMD_4980U` | AMD 4980U
`assera.Target.Model.AMD_5300G` | AMD 5300G
`assera.Target.Model.AMD_5300GE` | AMD 5300GE
`assera.Target.Model.AMD_5300U` | AMD 5300U
`assera.Target.Model.AMD_5400U` | AMD 5400U
`assera.Target.Model.AMD_5500U` | AMD 5500U
`assera.Target.Model.AMD_5600G` | AMD 5600G
`assera.Target.Model.AMD_5600GE` | AMD 5600GE
`assera.Target.Model.AMD_5600H` | AMD 5600H
`assera.Target.Model.AMD_5600HS` | AMD 5600HS
`assera.Target.Model.AMD_5600U` | AMD 5600U
`assera.Target.Model.AMD_5600X` | AMD 5600X
`assera.Target.Model.AMD_5700G` | AMD 5700G
`assera.Target.Model.AMD_5700GE` | AMD 5700GE
`assera.Target.Model.AMD_5700U` | AMD 5700U
`assera.Target.Model.AMD_5800` | AMD 5800
`assera.Target.Model.AMD_5800H` | AMD 5800H
`assera.Target.Model.AMD_5800HS` | AMD 5800HS
`assera.Target.Model.AMD_5800U` | AMD 5800U
`assera.Target.Model.AMD_5800X` | AMD 5800X
`assera.Target.Model.AMD_5900` | AMD 5900
`assera.Target.Model.AMD_5900HS` | AMD 5900HS
`assera.Target.Model.AMD_5900HX` | AMD 5900HX
`assera.Target.Model.AMD_5900X` | AMD 5900X
`assera.Target.Model.AMD_5950X` | AMD 5950X
`assera.Target.Model.AMD_5980HS` | AMD 5980HS
`assera.Target.Model.AMD_5980HX` | AMD 5980HX
`assera.Target.Model.AMD_7232P` | AMD 7232P
`assera.Target.Model.AMD_7251` | AMD 7251
`assera.Target.Model.AMD_7252` | AMD 7252
`assera.Target.Model.AMD_7261` | AMD 7261
`assera.Target.Model.AMD_7262` | AMD 7262
`assera.Target.Model.AMD_7272` | AMD 7272
`assera.Target.Model.AMD_7281` | AMD 7281
`assera.Target.Model.AMD_7282` | AMD 7282
`assera.Target.Model.AMD_72F3` | AMD 72F3
`assera.Target.Model.AMD_7301` | AMD 7301
`assera.Target.Model.AMD_7302` | AMD 7302
`assera.Target.Model.AMD_7302P` | AMD 7302P
`assera.Target.Model.AMD_7313` | AMD 7313
`assera.Target.Model.AMD_7313P` | AMD 7313P
`assera.Target.Model.AMD_7343` | AMD 7343
`assera.Target.Model.AMD_7351` | AMD 7351
`assera.Target.Model.AMD_7351P` | AMD 7351P
`assera.Target.Model.AMD_7352` | AMD 7352
`assera.Target.Model.AMD_7371` | AMD 7371
`assera.Target.Model.AMD_73F3` | AMD 73F3
`assera.Target.Model.AMD_7401` | AMD 7401
`assera.Target.Model.AMD_7401P` | AMD 7401P
`assera.Target.Model.AMD_7402` | AMD 7402
`assera.Target.Model.AMD_7402P` | AMD 7402P
`assera.Target.Model.AMD_7413` | AMD 7413
`assera.Target.Model.AMD_7443` | AMD 7443
`assera.Target.Model.AMD_7443P` | AMD 7443P
`assera.Target.Model.AMD_7451` | AMD 7451
`assera.Target.Model.AMD_7452` | AMD 7452
`assera.Target.Model.AMD_7453` | AMD 7453
`assera.Target.Model.AMD_74F3` | AMD 74F3
`assera.Target.Model.AMD_7501` | AMD 7501
`assera.Target.Model.AMD_7502` | AMD 7502
`assera.Target.Model.AMD_7502P` | AMD 7502P
`assera.Target.Model.AMD_7513` | AMD 7513
`assera.Target.Model.AMD_7532` | AMD 7532
`assera.Target.Model.AMD_7542` | AMD 7542
`assera.Target.Model.AMD_7543` | AMD 7543
`assera.Target.Model.AMD_7543P` | AMD 7543P
`assera.Target.Model.AMD_7551` | AMD 7551
`assera.Target.Model.AMD_7551P` | AMD 7551P
`assera.Target.Model.AMD_7552` | AMD 7552
`assera.Target.Model.AMD_75F3` | AMD 75F3
`assera.Target.Model.AMD_7601` | AMD 7601
`assera.Target.Model.AMD_7642` | AMD 7642
`assera.Target.Model.AMD_7643` | AMD 7643
`assera.Target.Model.AMD_7662` | AMD 7662
`assera.Target.Model.AMD_7663` | AMD 7663
`assera.Target.Model.AMD_7702` | AMD 7702
`assera.Target.Model.AMD_7702P` | AMD 7702P
`assera.Target.Model.AMD_7713` | AMD 7713
`assera.Target.Model.AMD_7713P` | AMD 7713P
`assera.Target.Model.AMD_7742` | AMD 7742
`assera.Target.Model.AMD_7763` | AMD 7763
`assera.Target.Model.AMD_7F32` | AMD 7F32
`assera.Target.Model.AMD_7F52` | AMD 7F52
`assera.Target.Model.AMD_7F72` | AMD 7F72
`assera.Target.Model.AMD_7H12` | AMD 7H12
`assera.Target.Model.AMD_7V12` | AMD 7V12
`assera.Target.Model.AMD_FIREFLIGHT` | AMD FireFlight
`assera.Target.Model.AMD_PRO_1200` | AMD PRO 1200
`assera.Target.Model.AMD_PRO_1300` | AMD PRO 1300
`assera.Target.Model.AMD_PRO_1500` | AMD PRO 1500
`assera.Target.Model.AMD_PRO_1600` | AMD PRO 1600
`assera.Target.Model.AMD_PRO_1700` | AMD PRO 1700
`assera.Target.Model.AMD_PRO_1700X` | AMD PRO 1700X
`assera.Target.Model.AMD_PRO_200GE` | AMD PRO 200GE
`assera.Target.Model.AMD_PRO_2200G` | AMD PRO 2200G
`assera.Target.Model.AMD_PRO_2200GE` | AMD PRO 2200GE
`assera.Target.Model.AMD_PRO_2300U` | AMD PRO 2300U
`assera.Target.Model.AMD_PRO_2400G` | AMD PRO 2400G
`assera.Target.Model.AMD_PRO_2400GE` | AMD PRO 2400GE
`assera.Target.Model.AMD_PRO_2500U` | AMD PRO 2500U
`assera.Target.Model.AMD_PRO_2600` | AMD PRO 2600
`assera.Target.Model.AMD_PRO_2700` | AMD PRO 2700
`assera.Target.Model.AMD_PRO_2700U` | AMD PRO 2700U
`assera.Target.Model.AMD_PRO_2700X` | AMD PRO 2700X
`assera.Target.Model.AMD_PRO_300GE` | AMD PRO 300GE
`assera.Target.Model.AMD_PRO_300U` | AMD PRO 300U
`assera.Target.Model.AMD_PRO_3200G` | AMD PRO 3200G
`assera.Target.Model.AMD_PRO_3200GE` | AMD PRO 3200GE
`assera.Target.Model.AMD_PRO_3300U` | AMD PRO 3300U
`assera.Target.Model.AMD_PRO_3400G` | AMD PRO 3400G
`assera.Target.Model.AMD_PRO_3400GE` | AMD PRO 3400GE
`assera.Target.Model.AMD_PRO_3500U` | AMD PRO 3500U
`assera.Target.Model.AMD_PRO_3600` | AMD PRO 3600
`assera.Target.Model.AMD_PRO_3700` | AMD PRO 3700
`assera.Target.Model.AMD_PRO_3700U` | AMD PRO 3700U
`assera.Target.Model.AMD_PRO_3900` | AMD PRO 3900
`assera.Target.Model.AMD_PRO_4350G` | AMD PRO 4350G
`assera.Target.Model.AMD_PRO_4350GE` | AMD PRO 4350GE
`assera.Target.Model.AMD_PRO_4450U` | AMD PRO 4450U
`assera.Target.Model.AMD_PRO_4650G` | AMD PRO 4650G
`assera.Target.Model.AMD_PRO_4650GE` | AMD PRO 4650GE
`assera.Target.Model.AMD_PRO_4650U` | AMD PRO 4650U
`assera.Target.Model.AMD_PRO_4750G` | AMD PRO 4750G
`assera.Target.Model.AMD_PRO_4750GE` | AMD PRO 4750GE
`assera.Target.Model.AMD_PRO_4750U` | AMD PRO 4750U
`assera.Target.Model.AMD_PRO_5350G` | AMD PRO 5350G
`assera.Target.Model.AMD_PRO_5350GE` | AMD PRO 5350GE
`assera.Target.Model.AMD_PRO_5450U` | AMD PRO 5450U
`assera.Target.Model.AMD_PRO_5650G` | AMD PRO 5650G
`assera.Target.Model.AMD_PRO_5650GE` | AMD PRO 5650GE
`assera.Target.Model.AMD_PRO_5650U` | AMD PRO 5650U
`assera.Target.Model.AMD_PRO_5750G` | AMD PRO 5750G
`assera.Target.Model.AMD_PRO_5750GE` | AMD PRO 5750GE
`assera.Target.Model.AMD_PRO_5850U` | AMD PRO 5850U
`assera.Target.Model.AMD_R1102G` | AMD R1102G
`assera.Target.Model.AMD_R1305G` | AMD R1305G
`assera.Target.Model.AMD_R1505G` | AMD R1505G
`assera.Target.Model.AMD_R1606G` | AMD R1606G
`assera.Target.Model.AMD_V1202B` | AMD V1202B
`assera.Target.Model.AMD_V1404I` | AMD V1404I
`assera.Target.Model.AMD_V1500B` | AMD V1500B
`assera.Target.Model.AMD_V1605B` | AMD V1605B
`assera.Target.Model.AMD_V1756B` | AMD V1756B
`assera.Target.Model.AMD_V1780B` | AMD V1780B
`assera.Target.Model.AMD_V1807B` | AMD V1807B
`assera.Target.Model.AMD_V2516` | AMD V2516
`assera.Target.Model.AMD_V2546` | AMD V2546
`assera.Target.Model.AMD_V2718` | AMD V2718
`assera.Target.Model.AMD_V2748` | AMD V2748
`assera.Target.Model.ARM_CORTEX_M4` | ARM Cortex-M4
`assera.Target.Model.ARM_CORTEX_M4F` | ARM Cortex-M4F
`assera.Target.Model.APPLE_M1_MAX` | Apple M1 Max
`assera.Target.Model.INTEL_1000G1` | Intel 1000G1
`assera.Target.Model.INTEL_1000G4` | Intel 1000G4
`assera.Target.Model.INTEL_1005G1` | Intel 1005G1
`assera.Target.Model.INTEL_10100` | Intel 10100
`assera.Target.Model.INTEL_10100F` | Intel 10100F
`assera.Target.Model.INTEL_10100T` | Intel 10100T
`assera.Target.Model.INTEL_10300` | Intel 10300
`assera.Target.Model.INTEL_10300T` | Intel 10300T
`assera.Target.Model.INTEL_1030G4` | Intel 1030G4
`assera.Target.Model.INTEL_1030G7` | Intel 1030G7
`assera.Target.Model.INTEL_10320` | Intel 10320
`assera.Target.Model.INTEL_1035G1` | Intel 1035G1
`assera.Target.Model.INTEL_1035G4` | Intel 1035G4
`assera.Target.Model.INTEL_1035G7` | Intel 1035G7
`assera.Target.Model.INTEL_10400` | Intel 10400
`assera.Target.Model.INTEL_10400F` | Intel 10400F
`assera.Target.Model.INTEL_10400T` | Intel 10400T
`assera.Target.Model.INTEL_10500` | Intel 10500
`assera.Target.Model.INTEL_10500T` | Intel 10500T
`assera.Target.Model.INTEL_10600` | Intel 10600
`assera.Target.Model.INTEL_10600K` | Intel 10600K
`assera.Target.Model.INTEL_10600KF` | Intel 10600KF
`assera.Target.Model.INTEL_10600T` | Intel 10600T
`assera.Target.Model.INTEL_1060G7` | Intel 1060G7
`assera.Target.Model.INTEL_1065G7` | Intel 1065G7
`assera.Target.Model.INTEL_1068G7` | Intel 1068G7
`assera.Target.Model.INTEL_10700` | Intel 10700
`assera.Target.Model.INTEL_10700F` | Intel 10700F
`assera.Target.Model.INTEL_10700K` | Intel 10700K
`assera.Target.Model.INTEL_10700KF` | Intel 10700KF
`assera.Target.Model.INTEL_10700T` | Intel 10700T
`assera.Target.Model.INTEL_10850K` | Intel 10850K
`assera.Target.Model.INTEL_10900` | Intel 10900
`assera.Target.Model.INTEL_10900F` | Intel 10900F
`assera.Target.Model.INTEL_10900K` | Intel 10900K
`assera.Target.Model.INTEL_10900KF` | Intel 10900KF
`assera.Target.Model.INTEL_10900T` | Intel 10900T
`assera.Target.Model.INTEL_10910` | Intel 10910
`assera.Target.Model.INTEL_11100B` | Intel 11100B
`assera.Target.Model.INTEL_1115G7` | Intel 1115G7
`assera.Target.Model.INTEL_1125G7` | Intel 1125G7
`assera.Target.Model.INTEL_1135G7` | Intel 1135G7
`assera.Target.Model.INTEL_11400` | Intel 11400
`assera.Target.Model.INTEL_11400F` | Intel 11400F
`assera.Target.Model.INTEL_11400T` | Intel 11400T
`assera.Target.Model.INTEL_1145G7` | Intel 1145G7
`assera.Target.Model.INTEL_11500` | Intel 11500
`assera.Target.Model.INTEL_11500B` | Intel 11500B
`assera.Target.Model.INTEL_11500T` | Intel 11500T
`assera.Target.Model.INTEL_1155G7` | Intel 1155G7
`assera.Target.Model.INTEL_11600` | Intel 11600
`assera.Target.Model.INTEL_11600K` | Intel 11600K
`assera.Target.Model.INTEL_11600KF` | Intel 11600KF
`assera.Target.Model.INTEL_11600T` | Intel 11600T
`assera.Target.Model.INTEL_1165G7` | Intel 1165G7
`assera.Target.Model.INTEL_11700` | Intel 11700
`assera.Target.Model.INTEL_11700B` | Intel 11700B
`assera.Target.Model.INTEL_11700F` | Intel 11700F
`assera.Target.Model.INTEL_11700K` | Intel 11700K
`assera.Target.Model.INTEL_11700KF` | Intel 11700KF
`assera.Target.Model.INTEL_11700T` | Intel 11700T
`assera.Target.Model.INTEL_11850H` | Intel 11850H
`assera.Target.Model.INTEL_1185G7` | Intel 1185G7
`assera.Target.Model.INTEL_11900` | Intel 11900
`assera.Target.Model.INTEL_11900F` | Intel 11900F
`assera.Target.Model.INTEL_11900K` | Intel 11900K
`assera.Target.Model.INTEL_11900KB` | Intel 11900KB
`assera.Target.Model.INTEL_11900KF` | Intel 11900KF
`assera.Target.Model.INTEL_11900T` | Intel 11900T
`assera.Target.Model.INTEL_1195G7` | Intel 1195G7
`assera.Target.Model.INTEL_2104G` | Intel 2104G
`assera.Target.Model.INTEL_2124` | Intel 2124
`assera.Target.Model.INTEL_2124G` | Intel 2124G
`assera.Target.Model.INTEL_2126G` | Intel 2126G
`assera.Target.Model.INTEL_2134` | Intel 2134
`assera.Target.Model.INTEL_2136` | Intel 2136
`assera.Target.Model.INTEL_2144G` | Intel 2144G
`assera.Target.Model.INTEL_2146G` | Intel 2146G
`assera.Target.Model.INTEL_2174G` | Intel 2174G
`assera.Target.Model.INTEL_2176G` | Intel 2176G
`assera.Target.Model.INTEL_2186G` | Intel 2186G
`assera.Target.Model.INTEL_2314` | Intel 2314
`assera.Target.Model.INTEL_2324G` | Intel 2324G
`assera.Target.Model.INTEL_2334` | Intel 2334
`assera.Target.Model.INTEL_2336` | Intel 2336
`assera.Target.Model.INTEL_2356G` | Intel 2356G
`assera.Target.Model.INTEL_2374G` | Intel 2374G
`assera.Target.Model.INTEL_2378` | Intel 2378
`assera.Target.Model.INTEL_2378G` | Intel 2378G
`assera.Target.Model.INTEL_2386G` | Intel 2386G
`assera.Target.Model.INTEL_2388G` | Intel 2388G
`assera.Target.Model.INTEL_3204` | Intel 3204
`assera.Target.Model.INTEL_4108` | Intel 4108
`assera.Target.Model.INTEL_4109T` | Intel 4109T
`assera.Target.Model.INTEL_4110` | Intel 4110
`assera.Target.Model.INTEL_4112` | Intel 4112
`assera.Target.Model.INTEL_4114` | Intel 4114
`assera.Target.Model.INTEL_4208` | Intel 4208
`assera.Target.Model.INTEL_4209T` | Intel 4209T
`assera.Target.Model.INTEL_4210` | Intel 4210
`assera.Target.Model.INTEL_4210R` | Intel 4210R
`assera.Target.Model.INTEL_4214` | Intel 4214
`assera.Target.Model.INTEL_4214R` | Intel 4214R
`assera.Target.Model.INTEL_4214Y` | Intel 4214Y
`assera.Target.Model.INTEL_4215` | Intel 4215
`assera.Target.Model.INTEL_4215R` | Intel 4215R
`assera.Target.Model.INTEL_4216` | Intel 4216
`assera.Target.Model.INTEL_5215` | Intel 5215
`assera.Target.Model.INTEL_5215L` | Intel 5215L
`assera.Target.Model.INTEL_5215M` | Intel 5215M
`assera.Target.Model.INTEL_5217` | Intel 5217
`assera.Target.Model.INTEL_5218` | Intel 5218
`assera.Target.Model.INTEL_5218B` | Intel 5218B
`assera.Target.Model.INTEL_5218N` | Intel 5218N
`assera.Target.Model.INTEL_5218R` | Intel 5218R
`assera.Target.Model.INTEL_5218T` | Intel 5218T
`assera.Target.Model.INTEL_5220` | Intel 5220
`assera.Target.Model.INTEL_5220R` | Intel 5220R
`assera.Target.Model.INTEL_5220S` | Intel 5220S
`assera.Target.Model.INTEL_5220T` | Intel 5220T
`assera.Target.Model.INTEL_5222` | Intel 5222
`assera.Target.Model.INTEL_6035` | Intel 6035
`assera.Target.Model.INTEL_6098P` | Intel 6098P
`assera.Target.Model.INTEL_6100` | Intel 6100
`assera.Target.Model.INTEL_6100T` | Intel 6100T
`assera.Target.Model.INTEL_6209U` | Intel 6209U
`assera.Target.Model.INTEL_6210U` | Intel 6210U
`assera.Target.Model.INTEL_6212U` | Intel 6212U
`assera.Target.Model.INTEL_6222V` | Intel 6222V
`assera.Target.Model.INTEL_6226` | Intel 6226
`assera.Target.Model.INTEL_6226R` | Intel 6226R
`assera.Target.Model.INTEL_6230` | Intel 6230
`assera.Target.Model.INTEL_6230N` | Intel 6230N
`assera.Target.Model.INTEL_6230R` | Intel 6230R
`assera.Target.Model.INTEL_6230T` | Intel 6230T
`assera.Target.Model.INTEL_6234` | Intel 6234
`assera.Target.Model.INTEL_6238` | Intel 6238
`assera.Target.Model.INTEL_6238L` | Intel 6238L
`assera.Target.Model.INTEL_6238M` | Intel 6238M
`assera.Target.Model.INTEL_6238R` | Intel 6238R
`assera.Target.Model.INTEL_6238T` | Intel 6238T
`assera.Target.Model.INTEL_6240` | Intel 6240
`assera.Target.Model.INTEL_6240L` | Intel 6240L
`assera.Target.Model.INTEL_6240M` | Intel 6240M
`assera.Target.Model.INTEL_6240R` | Intel 6240R
`assera.Target.Model.INTEL_6240Y` | Intel 6240Y
`assera.Target.Model.INTEL_6242` | Intel 6242
`assera.Target.Model.INTEL_6242R` | Intel 6242R
`assera.Target.Model.INTEL_6244` | Intel 6244
`assera.Target.Model.INTEL_6246` | Intel 6246
`assera.Target.Model.INTEL_6246R` | Intel 6246R
`assera.Target.Model.INTEL_6248` | Intel 6248
`assera.Target.Model.INTEL_6248R` | Intel 6248R
`assera.Target.Model.INTEL_6252` | Intel 6252
`assera.Target.Model.INTEL_6252N` | Intel 6252N
`assera.Target.Model.INTEL_6254` | Intel 6254
`assera.Target.Model.INTEL_6258R` | Intel 6258R
`assera.Target.Model.INTEL_6262V` | Intel 6262V
`assera.Target.Model.INTEL_6300` | Intel 6300
`assera.Target.Model.INTEL_6300T` | Intel 6300T
`assera.Target.Model.INTEL_6320` | Intel 6320
`assera.Target.Model.INTEL_6400` | Intel 6400
`assera.Target.Model.INTEL_6400T` | Intel 6400T
`assera.Target.Model.INTEL_6402P` | Intel 6402P
`assera.Target.Model.INTEL_6500` | Intel 6500
`assera.Target.Model.INTEL_6500T` | Intel 6500T
`assera.Target.Model.INTEL_6585R` | Intel 6585R
`assera.Target.Model.INTEL_6600` | Intel 6600
`assera.Target.Model.INTEL_6600K` | Intel 6600K
`assera.Target.Model.INTEL_6600T` | Intel 6600T
`assera.Target.Model.INTEL_6685R` | Intel 6685R
`assera.Target.Model.INTEL_6700` | Intel 6700
`assera.Target.Model.INTEL_6700K` | Intel 6700K
`assera.Target.Model.INTEL_6700T` | Intel 6700T
`assera.Target.Model.INTEL_6785R` | Intel 6785R
`assera.Target.Model.INTEL_6820HQ` | Intel 6820HQ
`assera.Target.Model.INTEL_7100` | Intel 7100
`assera.Target.Model.INTEL_7100T` | Intel 7100T
`assera.Target.Model.INTEL_7101E` | Intel 7101E
`assera.Target.Model.INTEL_7101TE` | Intel 7101TE
`assera.Target.Model.INTEL_7300` | Intel 7300
`assera.Target.Model.INTEL_7300T` | Intel 7300T
`assera.Target.Model.INTEL_7320` | Intel 7320
`assera.Target.Model.INTEL_7350K` | Intel 7350K
`assera.Target.Model.INTEL_7400` | Intel 7400
`assera.Target.Model.INTEL_7400T` | Intel 7400T
`assera.Target.Model.INTEL_7500` | Intel 7500
`assera.Target.Model.INTEL_7500T` | Intel 7500T
`assera.Target.Model.INTEL_7505` | Intel 7505
`assera.Target.Model.INTEL_7600` | Intel 7600
`assera.Target.Model.INTEL_7600K` | Intel 7600K
`assera.Target.Model.INTEL_7600T` | Intel 7600T
`assera.Target.Model.INTEL_7640X` | Intel 7640X
`assera.Target.Model.INTEL_7700` | Intel 7700
`assera.Target.Model.INTEL_7700K` | Intel 7700K
`assera.Target.Model.INTEL_7700T` | Intel 7700T
`assera.Target.Model.INTEL_7740X` | Intel 7740X
`assera.Target.Model.INTEL_7800X` | Intel 7800X
`assera.Target.Model.INTEL_7820X` | Intel 7820X
`assera.Target.Model.INTEL_7900X` | Intel 7900X
`assera.Target.Model.INTEL_7920X` | Intel 7920X
`assera.Target.Model.INTEL_7940X` | Intel 7940X
`assera.Target.Model.INTEL_7960X` | Intel 7960X
`assera.Target.Model.INTEL_7980XE` | Intel 7980XE
`assera.Target.Model.INTEL_8086K` | Intel 8086K
`assera.Target.Model.INTEL_8100` | Intel 8100
`assera.Target.Model.INTEL_8100F` | Intel 8100F
`assera.Target.Model.INTEL_8100T` | Intel 8100T
`assera.Target.Model.INTEL_8253` | Intel 8253
`assera.Target.Model.INTEL_8256` | Intel 8256
`assera.Target.Model.INTEL_8260` | Intel 8260
`assera.Target.Model.INTEL_8260L` | Intel 8260L
`assera.Target.Model.INTEL_8260M` | Intel 8260M
`assera.Target.Model.INTEL_8260Y` | Intel 8260Y
`assera.Target.Model.INTEL_8268` | Intel 8268
`assera.Target.Model.INTEL_8270` | Intel 8270
`assera.Target.Model.INTEL_8272CL` | Intel 8272CL
`assera.Target.Model.INTEL_8273CL` | Intel 8273CL
`assera.Target.Model.INTEL_8276` | Intel 8276
`assera.Target.Model.INTEL_8276L` | Intel 8276L
`assera.Target.Model.INTEL_8276M` | Intel 8276M
`assera.Target.Model.INTEL_8280` | Intel 8280
`assera.Target.Model.INTEL_8280L` | Intel 8280L
`assera.Target.Model.INTEL_8280M` | Intel 8280M
`assera.Target.Model.INTEL_8284` | Intel 8284
`assera.Target.Model.INTEL_8300` | Intel 8300
`assera.Target.Model.INTEL_8300T` | Intel 8300T
`assera.Target.Model.INTEL_8350K` | Intel 8350K
`assera.Target.Model.INTEL_8351N` | Intel 8351N
`assera.Target.Model.INTEL_8352S` | Intel 8352S
`assera.Target.Model.INTEL_8352V` | Intel 8352V
`assera.Target.Model.INTEL_8352Y` | Intel 8352Y
`assera.Target.Model.INTEL_8358` | Intel 8358
`assera.Target.Model.INTEL_8358P` | Intel 8358P
`assera.Target.Model.INTEL_8360Y` | Intel 8360Y
`assera.Target.Model.INTEL_8362` | Intel 8362
`assera.Target.Model.INTEL_8368` | Intel 8368
`assera.Target.Model.INTEL_8368Q` | Intel 8368Q
`assera.Target.Model.INTEL_8380` | Intel 8380
`assera.Target.Model.INTEL_8400` | Intel 8400
`assera.Target.Model.INTEL_8400T` | Intel 8400T
`assera.Target.Model.INTEL_8500` | Intel 8500
`assera.Target.Model.INTEL_8500T` | Intel 8500T
`assera.Target.Model.INTEL_8550U` | Intel 8550U
`assera.Target.Model.INTEL_8600` | Intel 8600
`assera.Target.Model.INTEL_8600K` | Intel 8600K
`assera.Target.Model.INTEL_8600T` | Intel 8600T
`assera.Target.Model.INTEL_8650U` | Intel 8650U
`assera.Target.Model.INTEL_8700` | Intel 8700
`assera.Target.Model.INTEL_8700K` | Intel 8700K
`assera.Target.Model.INTEL_8700T` | Intel 8700T
`assera.Target.Model.INTEL_9221` | Intel 9221
`assera.Target.Model.INTEL_9222` | Intel 9222
`assera.Target.Model.INTEL_9242` | Intel 9242
`assera.Target.Model.INTEL_9282` | Intel 9282
`assera.Target.Model.INTEL_9800X` | Intel 9800X
`assera.Target.Model.INTEL_9820X` | Intel 9820X
`assera.Target.Model.INTEL_9900X` | Intel 9900X
`assera.Target.Model.INTEL_9920X` | Intel 9920X
`assera.Target.Model.INTEL_9940X` | Intel 9940X
`assera.Target.Model.INTEL_9960X` | Intel 9960X
`assera.Target.Model.INTEL_9980XE` | Intel 9980XE
`assera.Target.Model.INTEL_9990XE` | Intel 9990XE
`assera.Target.Model.INTEL_E3_1220_V6` | Intel E3-1220 v6
`assera.Target.Model.INTEL_E3_1225_V6` | Intel E3-1225 v6
`assera.Target.Model.INTEL_E3_1230_V6` | Intel E3-1230 v6
`assera.Target.Model.INTEL_E3_1240_V6` | Intel E3-1240 v6
`assera.Target.Model.INTEL_E3_1245_V6` | Intel E3-1245 v6
`assera.Target.Model.INTEL_E3_1270_V6` | Intel E3-1270 v6
`assera.Target.Model.INTEL_E3_1275_V6` | Intel E3-1275 v6
`assera.Target.Model.INTEL_E3_1280_V6` | Intel E3-1280 v6
`assera.Target.Model.INTEL_E3_1285_V6` | Intel E3-1285 v6
`assera.Target.Model.INTEL_E5_1607_V2` | Intel E5-1607 v2
`assera.Target.Model.INTEL_E5_1620_V2` | Intel E5-1620 v2
`assera.Target.Model.INTEL_E5_1650_V2` | Intel E5-1650 v2
`assera.Target.Model.INTEL_E5_1650_V3` | Intel E5-1650 v3
`assera.Target.Model.INTEL_E5_1660_V2` | Intel E5-1660 v2
`assera.Target.Model.INTEL_E5_1660_V3` | Intel E5-1660 v3
`assera.Target.Model.INTEL_E5_1680_V2` | Intel E5-1680 v2
`assera.Target.Model.INTEL_E5_1680_V3` | Intel E5-1680 v3
`assera.Target.Model.INTEL_E5_2620_V3` | Intel E5-2620 v3
`assera.Target.Model.INTEL_E5_2673_V4` | Intel E5-2673 v4
`assera.Target.Model.INTEL_G3900` | Intel G3900
`assera.Target.Model.INTEL_G3900T` | Intel G3900T
`assera.Target.Model.INTEL_G3900TE` | Intel G3900TE
`assera.Target.Model.INTEL_G3920` | Intel G3920
`assera.Target.Model.INTEL_G4400` | Intel G4400
`assera.Target.Model.INTEL_G4400T` | Intel G4400T
`assera.Target.Model.INTEL_G4400TE` | Intel G4400TE
`assera.Target.Model.INTEL_G4500` | Intel G4500
`assera.Target.Model.INTEL_G4500T` | Intel G4500T
`assera.Target.Model.INTEL_G4520` | Intel G4520
`assera.Target.Model.INTEL_W_1250` | Intel W-1250
`assera.Target.Model.INTEL_W_1250P` | Intel W-1250P
`assera.Target.Model.INTEL_W_1270` | Intel W-1270
`assera.Target.Model.INTEL_W_1270P` | Intel W-1270P
`assera.Target.Model.INTEL_W_1290` | Intel W-1290
`assera.Target.Model.INTEL_W_1290P` | Intel W-1290P
`assera.Target.Model.INTEL_W_1290T` | Intel W-1290T
`assera.Target.Model.INTEL_W_1350` | Intel W-1350
`assera.Target.Model.INTEL_W_1350P` | Intel W-1350P
`assera.Target.Model.INTEL_W_1370` | Intel W-1370
`assera.Target.Model.INTEL_W_1370P` | Intel W-1370P
`assera.Target.Model.INTEL_W_1390` | Intel W-1390
`assera.Target.Model.INTEL_W_1390P` | Intel W-1390P
`assera.Target.Model.INTEL_W_1390T` | Intel W-1390T
`assera.Target.Model.INTEL_W_2102` | Intel W-2102
`assera.Target.Model.INTEL_W_2104` | Intel W-2104
`assera.Target.Model.INTEL_W_2123` | Intel W-2123
`assera.Target.Model.INTEL_W_2125` | Intel W-2125
`assera.Target.Model.INTEL_W_2133` | Intel W-2133
`assera.Target.Model.INTEL_W_2135` | Intel W-2135
`assera.Target.Model.INTEL_W_2140B` | Intel W-2140B
`assera.Target.Model.INTEL_W_2150B` | Intel W-2150B
`assera.Target.Model.INTEL_W_3175X` | Intel W-3175X
`assera.Target.Model.INTEL_W_3223` | Intel W-3223
`assera.Target.Model.INTEL_W_3225` | Intel W-3225
`assera.Target.Model.INTEL_W_3235` | Intel W-3235
`assera.Target.Model.INTEL_W_3245` | Intel W-3245
`assera.Target.Model.INTEL_W_3245M` | Intel W-3245M
`assera.Target.Model.INTEL_W_3265` | Intel W-3265
`assera.Target.Model.INTEL_W_3265M` | Intel W-3265M
`assera.Target.Model.INTEL_W_3275` | Intel W-3275
`assera.Target.Model.INTEL_W_3275M` | Intel W-3275M
`assera.Target.Model.RASPBERRY_PI_3B` | Raspberry Pi 3B
`assera.Target.Model.RASPBERRY_PI_4B` | Raspberry Pi 4B
`assera.Target.Model.RASPBERRY_PI_ZERO` | Raspberry Pi Zero

The enum also defines constants for some well-known GPU models.

type | description
--- | ---
`assera.Target.Model.AMD_MI100` | AMD MI100
`assera.Target.Model.AMD_MI200` | AMD MI200
`assera.Target.Model.AMD_MI50` | AMD MI50
`assera.Target.Model.AMD_RADEON7` | AMD Radeon7
`assera.Target.Model.NVIDIA_A100` | NVidia A100
`assera.Target.Model.NVIDIA_P100` | NVidia P100
`assera.Target.Model.NVIDIA_RTX_A6000` | NVidia RTX A6000
`assera.Target.Model.NVIDIA_V100` | NVidia V100

<div style="page-break-after: always;"></div>
