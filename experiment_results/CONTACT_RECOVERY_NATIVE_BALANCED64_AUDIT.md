# Native Contact-Recovery Gate-B Audit

- Demos: 64; frames: 14614
- Admission: PASS
- Direction counts: {'0': 8, '1': 8, '2': 8, '3': 8, '4': 8, '5': 8, '6': 8, '7': 8}
- Contact XY (m): {'p00': 0.004843342117965221, 'p05': 0.005034936056472361, 'p50': 0.005970624508336186, 'p95': 0.007618548977188766, 'p100': 0.008299151435494423}
- Contact height (m): {'p00': 0.025001555681228638, 'p05': 0.02500269152224064, 'p50': 0.025018034502863884, 'p95': 0.02504211086779833, 'p100': 0.025044728070497513}
- Torque excursion: {'p00': 0.07749170809984207, 'p05': 0.11808346360921859, 'p50': 0.1608303189277649, 'p95': 0.30737731307744975, 'p100': 0.3361072540283203}
- Recovery-label frames: {'p00': 44.0, 'p05': 46.15, 'p50': 51.0, 'p95': 56.849999999999994, 'p100': 58.0}

## Decision

This contact set is admitted only when every trajectory has native-reset, pre-action alignment, an uninserted rim-blocked state, a 30-step real torque history, torque excursion, and strict physical recovery.

## Errors
