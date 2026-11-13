"""F7c acceptance with real data (--channel-parquet, --search-dir, --sin-clima-dir and --prueba-dir): the new figures of
the derivable reports of unit U7.5 (contrato §7.2, §7.3, §7.4 and §8.4; especificación §6.F7), frozen after Jesús's
independent check of 2026-11-12. The formulas are frozen in test_f7c_formulas.py and the identities with the earlier
evidence in test_f7c_identidades_real.py; the inputs of each period are the ones described there.

Provenance of the literals: computed by reports.py (176f89d) from a clean tree; evidence in
reports/forecasting/evidence/reportes_20261112/ (manifiesto_reportes.json: recipe de74d4ab…fda5, test execution 1).
Jesús's independent calculation reproduced every figure (classification at tau global and by municipality, Murphy and
residuals, calibration by year, events, durations from t_c, weeks under alert, development against test with the recipe
against the ablation 16/15 and 9/10, AMOC with 22 and 24 false alarms). The only difference was the channel ratio in
development (reports.py 9/63, the independent calculation 10/63), explained by convention D4: Villavicencio 2022-10-31
is warned only by a non-judgeable alert (t_c 2022-10-03), so it does not enter the numerator; the ratio of
events._summarize (9/63) is accepted.
Tolerances: exact for counts and interval numbers; rel 1e-9 (abs 1e-12) for rates, scores and components.
"""

import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
TAU = 0.05
D, T = "desarrollo", "prueba"
NAN = math.nan

# (periodo, sistema, nivel, grupo): (emisiones_positivas, vp, fp, fn, vn, sensibilidad, especificidad, vpp, f1, mcc)
CLASSIFICATION = {
    (D, 'modelo', 'global', 'todos'): (663, 313, 350, 6, 579, 0.9811912225705328, 0.6232508073196986, 0.4720965309200603, 0.6374745417515275, 0.5283517578457425),
    (D, 'B2', 'global', 'todos'): (877, 314, 563, 5, 366, 0.9843260188087776, 0.3939720129171152, 0.3580387685290764, 0.5250836120401338, 0.3610360164767376),
    (D, 'modelo', 'anio', '2019'): (215, 117, 98, 2, 95, 0.9831932773109244, 0.4922279792746113, 0.5441860465116279, 0.7005988023952096, 0.4989139343745459),
    (D, 'B2', 'anio', '2019'): (234, 114, 120, 5, 73, 0.957983193277311, 0.3782383419689119, 0.4871794871794871, 0.6458923512747875, 0.3771572253108393),
    (D, 'modelo', 'anio', '2020'): (120, 24, 96, 0, 192, 1.0, 0.6666666666666666, 0.2, 0.3333333333333333, 0.3651483716701107),
    (D, 'B2', 'anio', '2020'): (177, 24, 153, 0, 135, 1.0, 0.46875, 0.135593220338983, 0.2388059701492537, 0.2521097420448053),
    (D, 'modelo', 'anio', '2021'): (146, 68, 78, 4, 162, 0.9444444444444444, 0.675, 0.4657534246575342, 0.6238532110091743, 0.5230506660255734),
    (D, 'B2', 'anio', '2021'): (187, 72, 115, 0, 125, 1.0, 0.5208333333333334, 0.3850267379679144, 0.555984555984556, 0.447811075519899),
    (D, 'modelo', 'anio', '2022'): (182, 104, 78, 0, 130, 1.0, 0.625, 0.5714285714285714, 0.7272727272727273, 0.5976143046671968),
    (D, 'B2', 'anio', '2022'): (279, 104, 175, 0, 33, 1.0, 0.1586538461538461, 0.3727598566308244, 0.5430809399477807, 0.24318672855696),
    (D, 'modelo', 'municipio', '08001'): (142, 77, 65, 1, 65, 0.9871794871794872, 0.5, 0.5422535211267606, 0.7, 0.5067477527945231),
    (D, 'B2', 'municipio', '08001'): (149, 78, 71, 0, 59, 1.0, 0.4538461538461538, 0.5234899328859061, 0.6872246696035242, 0.4874257816503448),
    (D, 'modelo', 'municipio', '08758'): (157, 85, 72, 2, 49, 0.9770114942528736, 0.4049586776859504, 0.5414012738853503, 0.6967213114754098, 0.4379725051243652),
    (D, 'B2', 'municipio', '08758'): (192, 83, 109, 4, 12, 0.9540229885057472, 0.0991735537190082, 0.4322916666666667, 0.5949820788530465, 0.0984748005484365),
    (D, 'modelo', 'municipio', '41001'): (89, 35, 54, 0, 119, 1.0, 0.6878612716763006, 0.3932584269662921, 0.5645161290322581, 0.5201031067687015),
    (D, 'B2', 'municipio', '41001'): (109, 34, 75, 1, 98, 0.9714285714285714, 0.5664739884393064, 0.3119266055045872, 0.4722222222222222, 0.4029302177812313),
    (D, 'modelo', 'municipio', '50001'): (82, 55, 27, 0, 126, 1.0, 0.8235294117647058, 0.6707317073170732, 0.8029197080291971, 0.7432141605074315),
    (D, 'B2', 'municipio', '50001'): (108, 55, 53, 0, 100, 1.0, 0.6535947712418301, 0.5092592592592593, 0.6748466257668712, 0.576930835593262),
    (D, 'modelo', 'municipio', '54001'): (107, 43, 64, 1, 100, 0.9772727272727272, 0.6097560975609756, 0.4018691588785046, 0.5695364238410596, 0.4796838682828987),
    (D, 'B2', 'municipio', '54001'): (152, 44, 108, 0, 56, 1.0, 0.3414634146341463, 0.2894736842105263, 0.4489795918367347, 0.3143957262070413),
    (D, 'modelo', 'municipio', '76001'): (86, 18, 68, 2, 120, 0.9, 0.6382978723404256, 0.2093023255813953, 0.3396226415094339, 0.322245932871307),
    (D, 'B2', 'municipio', '76001'): (167, 20, 147, 0, 41, 1.0, 0.2180851063829787, 0.1197604790419161, 0.213903743315508, 0.1616105714745566),
    (T, 'modelo', 'global', 'todos'): (611, 308, 303, 14, 311, 0.9565217391304348, 0.506514657980456, 0.5040916530278232, 0.6602357984994641, 0.4620244577633473),
    (T, 'B2', 'global', 'todos'): (829, 322, 507, 0, 107, 1.0, 0.1742671009771987, 0.3884197828709288, 0.5595134665508253, 0.2601706930519072),
    (T, 'modelo', 'anio', '2023'): (206, 134, 72, 8, 98, 0.943661971830986, 0.5764705882352941, 0.6504854368932039, 0.7701149425287356, 0.5468851506714392),
    (T, 'B2', 'anio', '2023'): (281, 142, 139, 0, 31, 1.0, 0.1823529411764705, 0.505338078291815, 0.6713947990543735, 0.3035619950932231),
    (T, 'modelo', 'anio', '2024'): (241, 122, 119, 2, 75, 0.9838709677419356, 0.3865979381443299, 0.5062240663900415, 0.6684931506849315, 0.4218029243909482),
    (T, 'B2', 'anio', '2024'): (313, 124, 189, 0, 5, 1.0, 0.0257731958762886, 0.3961661341853035, 0.5675057208237986, 0.1010468573282211),
    (T, 'modelo', 'anio', '2025'): (164, 52, 112, 4, 138, 0.9285714285714286, 0.552, 0.3170731707317073, 0.4727272727272727, 0.372611169022305),
    (T, 'B2', 'anio', '2025'): (235, 56, 179, 0, 71, 1.0, 0.284, 0.2382978723404255, 0.3848797250859107, 0.2601472578073442),
    (T, 'modelo', 'municipio', '08001'): (121, 47, 74, 0, 35, 1.0, 0.3211009174311927, 0.3884297520661157, 0.5595238095238095, 0.3531644797343024),
    (T, 'B2', 'municipio', '08001'): (128, 47, 81, 0, 28, 1.0, 0.2568807339449541, 0.3671875, 0.5371428571428571, 0.3071211397729125),
    (T, 'modelo', 'municipio', '08758'): (114, 46, 68, 5, 37, 0.9019607843137256, 0.3523809523809524, 0.4035087719298245, 0.5575757575757576, 0.2689801919216923),
    (T, 'B2', 'municipio', '08758'): (156, 51, 105, 0, 0, 1.0, 0.0, 0.3269230769230769, 0.4927536231884058, NAN),
    (T, 'modelo', 'municipio', '41001'): (101, 54, 47, 1, 54, 0.9818181818181818, 0.5346534653465347, 0.5346534653465347, 0.6923076923076923, 0.5164716471647165),
    (T, 'B2', 'municipio', '41001'): (145, 55, 90, 0, 11, 1.0, 0.1089108910891089, 0.3793103448275862, 0.55, 0.2032511442882661),
    (T, 'modelo', 'municipio', '50001'): (100, 52, 48, 1, 55, 0.981132075471698, 0.5339805825242718, 0.52, 0.6797385620915033, 0.5085864153086944),
    (T, 'B2', 'municipio', '50001'): (155, 53, 102, 0, 1, 1.0, 0.0097087378640776, 0.3419354838709677, 0.5096153846153846, 0.0576173756720122),
    (T, 'modelo', 'municipio', '54001'): (100, 49, 51, 1, 55, 0.98, 0.5188679245283019, 0.49, 0.6533333333333333, 0.4853214679196869),
    (T, 'B2', 'municipio', '54001'): (139, 50, 89, 0, 17, 1.0, 0.160377358490566, 0.3597122302158273, 0.5291005291005291, 0.2401867966786783),
    (T, 'modelo', 'municipio', '76001'): (75, 60, 15, 6, 75, 0.9090909090909092, 0.8333333333333334, 0.8, 0.851063829787234, 0.7341287391266414),
    (T, 'B2', 'municipio', '76001'): (106, 66, 40, 0, 50, 1.0, 0.5555555555555556, 0.6226415094339622, 0.7674418604651163, 0.5881427970191717),
}
# (periodo, sistema, grupo): (intervalos, brier, confiabilidad, resolucion, incertidumbre, varianza_intra, covarianza_intra, residuo)
MURPHY = {
    (D, 'modelo', 'todos'): (10, 0.1009935494539258, 0.0033376799792841, 0.093156194493489, 0.1902730265861275, 0.0016446277009612, 0.000552795159479, 0.0005390373820031),
    (D, 'ablacion', 'todos'): (10, 0.1155922418966752, 0.0026386507106155, 0.0757991845679307, 0.1902730265861275, 0.0015650525817932, 0.0015426517069652, -0.0015202508321371),
    (D, 'modelo', '2019'): (10, 0.1238858461539901, 0.0074967391623769, 0.1174210415738796, 0.2359364727153188, 0.0014234045974612, 0.0017748643736436, -0.002126324149826),
    (D, 'ablacion', '2019'): (10, 0.1550464602428088, 0.0154895263803288, 0.0967169472810757, 0.2359364727153188, 0.0011974172711324, 0.0004300044214478, 0.0003374084282368),
    (D, 'modelo', '2020'): (10, 0.1008229540245054, 0.0371455800905608, 0.0105705108322198, 0.0710059171597633, 0.0027621581496313, -0.0002399047283849, 0.0032419676064012),
    (D, 'ablacion', '2020'): (10, 0.1161823995793057, 0.0517589127657118, 0.0063314703028567, 0.0710059171597633, 0.0023709332061906, 0.0013109466247516, -0.0002509600433126),
    (D, 'modelo', '2021'): (10, 0.1006576855072684, 0.0205470212107204, 0.0960134229655786, 0.1775147928994082, 0.0018195086846562, 0.0016051071609689, -0.0013907056372816),
    (D, 'ablacion', '2021'): (10, 0.1047232167255971, 0.0110725540400741, 0.0831702933129732, 0.1775147928994082, 0.0030232740695741, 0.0018585554852431, -0.000693836900912),
    (D, 'modelo', '2022'): (10, 0.0786077121299393, 0.0192896867347145, 0.1574863868210642, 0.2222222222222222, 0.0019047230850365, 0.0036612665454848, -0.0054178100059331),
    (D, 'ablacion', '2022'): (10, 0.0864168910389891, 0.0051966754126635, 0.1392733750344637, 0.2222222222222222, 0.0013125546803917, 0.0015205931209123, -0.0017286315614329),
    (T, 'modelo', 'todos'): (10, 0.1413204696650012, 0.0058410320447327, 0.0874073909871474, 0.2256693330411279, 0.0013485048403181, 0.002065504637015, -0.0027825044337119),
    (T, 'ablacion', 'todos'): (10, 0.1434225655235304, 0.0038678654630066, 0.0860237083234373, 0.2256693330411279, 0.0012496698462679, 0.0006702972517173, -9.092465716678544e-05),
    (T, 'modelo', '2023'): (10, 0.1258553121448158, 0.0070967486289643, 0.1305340819918983, 0.2479865220249835, 0.0011478321888872, -7.91456469395422e-05, 0.0013061234827662),
    (T, 'ablacion', '2023'): (10, 0.1196789789902059, 0.0057140131197114, 0.1315679943161332, 0.2479865220249835, 0.0013710724126905, 0.0019123171255231, -0.0024535618383558),
    (T, 'modelo', '2024'): (10, 0.1475543818704198, 0.0059896832994456, 0.0985006351193233, 0.2378861595664728, 0.001335332507782, -0.0004219208080213, 0.0021791741238246),
    (T, 'ablacion', '2024'): (10, 0.1673036115242194, 0.0104719188891511, 0.0813952384871464, 0.2378861595664728, 0.0013694583479562, 0.0005143433961071, 0.0003407715557418),
    (T, 'modelo', '2025'): (10, 0.1506104862172064, 0.022972275391287, 0.0253848464679286, 0.149515143748131, 0.0017842898449326, -0.0008618118503921, 0.003507913545717),
    (T, 'ablacion', '2025'): (10, 0.1428141549685571, 0.018827026134529, 0.0290604385685681, 0.149515143748131, 0.0015714009144021, -0.0009805113700315, 0.0035324236544652),
}
# (periodo, anio): (filas, positivas, calibracion modelo, ablacion, B2, B1)
CALIBRATION = {
    (D, '2019'): (312, 119, -0.0635089865633219, -0.0946894772247354, -0.1515671525729032, -0.2332247516990163),
    (D, '2020'): (312, 24, 0.079626698404349, 0.0811739998922023, 0.0620685620558499, 0.0998148827515865),
    (D, '2021'): (312, 72, -0.085710247941796, -0.0678616507816685, -0.0783879211973518, -0.0593094517232105),
    (D, '2022'): (312, 104, -0.0490156602245991, 0.0258239570744377, -0.0508353366874169, -0.1620521221151601),
    (D, 'todos'): (1248, 319, -0.029652049081342, -0.0138882927599409, -0.0546804621004555, -0.0886928606964501),
    (T, '2023'): (312, 142, -0.0673593318065919, -0.055081091994475, -0.0910389203026388, -0.269687576478979),
    (T, '2024'): (318, 124, -0.0016442029706868, -0.0332761067644432, -0.0177281344303922, -0.1812251350944188),
    (T, '2025'): (306, 56, 0.0115592452521719, 0.0461257634202011, 0.0326144484131793, 0.0436241982259522),
    (T, 'todos'): (936, 322, -0.019232733996977, -0.0145861314602431, -0.0257069109709991, -0.1372041796627406),
}
# (periodo, sistema, grupo): (inicios_evaluables, inicios_avisados, inicios_avisados_por_juzgable, alertas,
#   alertas_continuidad, alertas_juzgables, alertas_censuradas, alertas_que_avisan, alertas_con_varios_inicios,
#   falsas_alarmas, falsas_seguidas_epidemia, municipio_anios)
EVENTS = {
    (D, 'modelo', 'todos'): (10, 8, 8, 43, 5, 38, 0, 13, 0, 24, 6, 24),
    (D, 'modelo', '08001'): (2, 2, 2, 6, 1, 5, 0, 3, 0, 2, 1, 4),
    (D, 'modelo', '08758'): (3, 2, 2, 9, 3, 6, 0, 4, 0, 2, 1, 4),
    (D, 'modelo', '41001'): (1, 1, 1, 8, 0, 8, 0, 1, 0, 7, 0, 4),
    (D, 'modelo', '50001'): (2, 1, 1, 3, 0, 3, 0, 1, 0, 1, 1, 4),
    (D, 'modelo', '54001'): (1, 1, 1, 8, 1, 7, 0, 3, 0, 4, 0, 4),
    (D, 'modelo', '76001'): (1, 1, 1, 9, 0, 9, 0, 1, 0, 8, 3, 4),
    (D, 'canal', 'todos'): (10, 10, 9, 65, 0, 63, 2, 13, 0, 48, 12, 24),
    (D, 'canal', '08001'): (2, 2, 2, 13, 0, 13, 0, 2, 0, 11, 2, 4),
    (D, 'canal', '08758'): (3, 3, 3, 14, 0, 14, 0, 5, 0, 9, 7, 4),
    (D, 'canal', '41001'): (1, 1, 1, 11, 0, 11, 0, 2, 0, 9, 1, 4),
    (D, 'canal', '50001'): (2, 2, 1, 7, 0, 6, 1, 1, 0, 3, 0, 4),
    (D, 'canal', '54001'): (1, 1, 1, 9, 0, 8, 1, 2, 0, 6, 0, 4),
    (D, 'canal', '76001'): (1, 1, 1, 11, 0, 11, 0, 1, 0, 10, 2, 4),
    (T, 'modelo', 'todos'): (10, 4, 4, 26, 4, 22, 0, 4, 1, 18, 5, 18),
    (T, 'modelo', '08001'): (1, 0, 0, 3, 1, 2, 0, 0, 0, 2, 1, 3),
    (T, 'modelo', '08758'): (3, 1, 1, 5, 1, 4, 0, 1, 0, 3, 2, 3),
    (T, 'modelo', '41001'): (3, 2, 2, 7, 0, 7, 0, 2, 1, 5, 0, 3),
    (T, 'modelo', '50001'): (1, 0, 0, 7, 2, 5, 0, 0, 0, 5, 0, 3),
    (T, 'modelo', '54001'): (2, 1, 1, 3, 0, 3, 0, 1, 0, 2, 1, 3),
    (T, 'modelo', '76001'): (0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 3),
    (T, 'canal', 'todos'): (10, 8, 8, 39, 0, 39, 0, 9, 1, 30, 6, 18),
    (T, 'canal', '08001'): (1, 1, 1, 6, 0, 6, 0, 1, 0, 5, 1, 3),
    (T, 'canal', '08758'): (3, 2, 2, 9, 0, 9, 0, 3, 0, 6, 3, 3),
    (T, 'canal', '41001'): (3, 3, 3, 9, 0, 9, 0, 3, 1, 6, 1, 3),
    (T, 'canal', '50001'): (1, 0, 0, 4, 0, 4, 0, 0, 0, 4, 1, 3),
    (T, 'canal', '54001'): (2, 2, 2, 8, 0, 8, 0, 2, 0, 6, 0, 3),
    (T, 'canal', '76001'): (0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 3),
}
# (periodo, sistema, grupo): (alertas_no_continuidad, alertas_truncadas, duracion_mediana, duracion_p75,
#   duracion_max, semanas_emision, semanas_bajo_alerta, semanas_bajo_alerta_no_continuidad)
DURATIONS = {
    (D, 'modelo', 'todos'): (38, 3, 3.5, 11.5, 70.0, 1248, 586, 440),
    (D, 'modelo', '08001'): (5, 1, 25.0, 25.0, 70.0, 208, 134, 123),
    (D, 'modelo', '08758'): (6, 1, 5.5, 9.0, 25.0, 208, 142, 47),
    (D, 'modelo', '41001'): (8, 0, 1.0, 10.5, 47.0, 208, 74, 74),
    (D, 'modelo', '50001'): (3, 1, 9.0, 37.0, 65.0, 208, 78, 78),
    (D, 'modelo', '54001'): (7, 0, 3.0, 6.5, 30.0, 208, 90, 50),
    (D, 'modelo', '76001'): (9, 0, 2.0, 8.0, 30.0, 208, 68, 68),
    (D, 'canal', 'todos'): (65, 1, 3.0, 7.0, 39.0, 1248, 406, 406),
    (D, 'canal', '08001'): (13, 0, 4.0, 7.0, 11.0, 208, 59, 59),
    (D, 'canal', '08758'): (14, 0, 2.0, 2.0, 6.0, 208, 29, 29),
    (D, 'canal', '41001'): (11, 0, 3.0, 7.5, 15.0, 208, 57, 57),
    (D, 'canal', '50001'): (7, 0, 2.0, 5.5, 26.0, 208, 42, 42),
    (D, 'canal', '54001'): (9, 0, 4.0, 8.0, 39.0, 208, 84, 84),
    (D, 'canal', '76001'): (11, 1, 13.0, 16.0, 35.0, 208, 135, 135),
    (T, 'modelo', 'todos'): (22, 1, 10.0, 36.75, 73.0, 936, 565, 454),
    (T, 'modelo', '08001'): (2, 0, 39.0, 56.0, 73.0, 156, 117, 78),
    (T, 'modelo', '08758'): (4, 0, 20.0, 26.5, 43.0, 156, 106, 96),
    (T, 'modelo', '41001'): (7, 0, 3.0, 7.5, 66.0, 156, 88, 88),
    (T, 'modelo', '50001'): (5, 1, 6.0, 9.0, 11.0, 156, 90, 28),
    (T, 'modelo', '54001'): (3, 0, 42.0, 43.5, 45.0, 156, 92, 92),
    (T, 'modelo', '76001'): (1, 0, 72.0, 72.0, 72.0, 156, 72, 72),
    (T, 'canal', 'todos'): (39, 2, 4.0, 17.0, 36.0, 936, 393, 393),
    (T, 'canal', '08001'): (6, 0, 11.0, 16.5, 20.0, 156, 63, 63),
    (T, 'canal', '08758'): (9, 0, 4.0, 6.0, 28.0, 156, 58, 58),
    (T, 'canal', '41001'): (9, 0, 3.0, 8.0, 36.0, 156, 77, 77),
    (T, 'canal', '50001'): (4, 1, 27.5, 29.75, 35.0, 156, 91, 91),
    (T, 'canal', '54001'): (8, 1, 2.0, 15.75, 27.0, 156, 71, 71),
    (T, 'canal', '76001'): (3, 0, 13.0, 14.5, 16.0, 156, 33, 33),
}
# (periodo, sistema, referencia): (ganadas_brier, ganadas_logloss, bss_filas, bss_casillas, mejora_logloss_filas,
#   mejora_logloss_casillas)
SKILL = {
    (D, 'modelo', 'B2'): (16, 16, 0.1875549619162526, 0.1899196861540463, 0.1840358141555348, 0.2109095829397954),
    (D, 'modelo', 'B1'): (18, 21, 0.5164400753546546, 0.3704915166455009, 0.512623487450631, 0.4446990853207942),
    (D, 'modelo', 'ablacion'): (16, 15, 0.1262947426506248, 0.1008735821728509, 0.1385161196115536, 0.1517346063985895),
    (D, 'ablacion', 'B2'): (14, 14, 0.0701154293742921, -0.0843679743579539, 0.0528387072355389, -0.0239681814240181),
    (D, 'ablacion', 'B1'): (17, 17, 0.4465411297714321, 0.1253371224153357, 0.4342592779221719, 0.2909216552135944),
    (T, 'modelo', 'B2'): (12, 12, 0.0805221856615173, 0.171652466430175, 0.0634783282186487, 0.1498140922911422),
    (T, 'modelo', 'B1'): (15, 15, 0.4684107819641709, 0.3986885488072351, 0.4354522303114413, 0.3955574447710962),
    (T, 'modelo', 'ablacion'): (9, 10, 0.0146566605530726, 0.0614653660383988, 0.0203431503138784, 0.0636107532021726),
    (T, 'ablacion', 'B2'): (11, 12, 0.0668452532955821, 0.1020616924458287, 0.0440309052282854, 0.0812039119913567),
    (T, 'ablacion', 'B1'): (14, 15, 0.4605035658593788, 0.3654547357704937, 0.4237290640396811, 0.3661477013855058),
}
COMMON_CURVE_WARNED = [8, 5, 4, 3, 2, 1, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
COMMON_CURVE_FALSE_ALARMS = [24, 24, 10, 6, 7, 4, 7, 4, 2, 2, 2, 3, 3, 4, 2, 1, 1, 0, 0]


def _config():
    from src.forecasting.config import load_config

    return load_config()


@functools.lru_cache(maxsize=None)
def _product(channel_file: str):
    from src.forecasting.dataset import load_channel_product

    return load_channel_product(channel_file, expected_md5=_config().datos.version_dvc_canal)


_CACHE = {}


@pytest.fixture
def tables(channel_path, search_dir, sin_clima_dir, prueba_dir, tmp_path):
    from src.forecasting.dataset import build_examples
    from src.forecasting.guard import RecipeAccess
    from src.forecasting.run_selection import development_rows
    from src.forecasting import reports

    key = (str(channel_path), str(search_dir), str(sin_clima_dir), str(prueba_dir))
    if key not in _CACHE:
        config = _config()
        product = _product(str(channel_path))
        observed = product.to_pandas()
        examples = build_examples(product, h=config.horizonte.h, canal=config.canal, features=config.features)
        development = reports.development_systems(development_rows(examples, config), search_dir=search_dir,
                                                  sin_clima_dir=sin_clima_dir, familia="logistica")
        none = RecipeAccess.from_config(config, recipe_path=tmp_path / "receta_inexistente.yaml")
        recipe = RecipeAccess.from_config(config, recipe_path=ROOT / "configs" / "receta_congelada.yaml")
        dev = reports.period_reports(development, observed, config=config, tau=TAU, access=none)
        test = reports.period_reports(reports.test_systems(prueba_dir), observed, config=config, tau=TAU,
                                      access=recipe)
        _CACHE[key] = {name: pd.concat([dev[name], test[name]], ignore_index=True) for name in dev}
    return _CACHE[key]


def _close(value, expected) -> bool:
    if isinstance(expected, float) and math.isnan(expected):
        return isinstance(value, float) and math.isnan(value)
    return value == pytest.approx(expected, rel=1e-9, abs=1e-12)


def _check(frame: pd.DataFrame, keys: list[str], columns: list[str], expected: dict, n_exact: int) -> None:
    got = {tuple(row[key] for key in keys): row for _, row in frame.iterrows()}
    assert sorted(got) == sorted(expected)
    for key, values in expected.items():
        row = got[key]
        for i, (column, value) in enumerate(zip(columns, values)):
            if i < n_exact:
                assert int(row[column]) == value, (key, column)
            else:
                assert _close(float(row[column]), value), (key, column, row[column], value)


def test_classification_at_tau(tables):
    _check(tables["clasificacion_tau"], ["periodo", "sistema", "nivel", "grupo"],
           ["emisiones_positivas", "vp", "fp", "fn", "vn", "sensibilidad", "especificidad", "vpp", "f1", "mcc"],
           CLASSIFICATION, n_exact=5)
    for row in tables["clasificacion_tau"].itertuples():
        assert row.f1_referencia == pytest.approx(2 * row.prevalencia / (1 + row.prevalencia))
        assert row.tau == TAU


def test_murphy_decomposition(tables):
    _check(tables["murphy"], ["periodo", "sistema", "grupo"],
           ["intervalos", "brier", "confiabilidad", "resolucion", "incertidumbre", "varianza_intra",
            "covarianza_intra", "residuo"], MURPHY, n_exact=1)


def test_calibration_in_the_large_by_year(tables):
    frame = tables["calibracion_anual"].assign(anio=lambda f: f["anio"].astype(str))
    _check(frame, ["periodo", "anio"],
           ["filas", "positivas", "calibracion_modelo", "calibracion_ablacion", "calibracion_B2", "calibracion_B1"],
           CALIBRATION, n_exact=2)


def test_events_ratio_and_false_alarm_breakdown(tables):
    events = tables["eventos_razon"]
    _check(events, ["periodo", "sistema", "grupo"],
           ["inicios_evaluables", "inicios_avisados", "inicios_avisados_por_juzgable", "alertas", "alertas_continuidad",
            "alertas_juzgables", "alertas_censuradas", "alertas_que_avisan", "alertas_con_varios_inicios",
            "falsas_alarmas", "falsas_seguidas_epidemia", "municipio_anios"], EVENTS, n_exact=12)
    for row in events.itertuples():
        assert _close(row.razon_alertas, row.inicios_avisados_por_juzgable / row.alertas_juzgables
                      if row.alertas_juzgables else NAN)
        assert _close(row.compuesto, 0.5 * row.deteccion + 0.5 * row.razon_alertas)
        assert row.falsas_no_seguidas == row.falsas_alarmas - row.falsas_seguidas_epidemia


def test_alert_durations(tables):
    _check(tables["duracion_alertas"], ["periodo", "sistema", "grupo"],
           ["alertas_no_continuidad", "alertas_truncadas", "duracion_mediana", "duracion_p75", "duracion_max",
            "semanas_emision", "semanas_bajo_alerta", "semanas_bajo_alerta_no_continuidad"], DURATIONS, n_exact=2)


def test_development_against_test_skill(tables):
    _check(tables["habilidad"], ["periodo", "sistema", "referencia"],
           ["ganadas_brier", "ganadas_logloss", "bss_filas", "bss_casillas", "mejora_logloss_filas",
            "mejora_logloss_casillas"], SKILL, n_exact=2)


def test_development_amoc_with_the_common_configuration(tables):
    curve = tables["curva_amoc"]
    own = curve[(curve["periodo"] == D) & (curve["configuracion"] == "comun_fuera_de_anio")
                & (curve["sistema"] == "modelo")]
    assert own["inicios_avisados"].tolist() == COMMON_CURVE_WARNED
    assert own["falsas_alarmas"].tolist() == COMMON_CURVE_FALSE_ALARMS
    assert np.isclose(own["tau"].to_numpy(), _config().decision.tau_grid).all()
