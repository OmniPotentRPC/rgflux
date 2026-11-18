# Native PAFI benchmark

## Verdict

All 1728 planned executions completed with 0 process failures. rgflux retained 361 numerical or quadrature rejections; the reference accepted every execution.

- Curved-path agreement improves strongly from 17 to 33 planes. At 33 planes and 256 samples, median maximum profile differences are below 3.7e-6 in every temperature and thread cell.
- The accepted 17-plane straight profiles differ by 0.029 to 0.062 at 256 samples, exceeding the combined profile standard errors. This cell does not establish profile equivalence.
- Every 9-plane straight run is rejected by the unchanged zero-temperature quadrature tolerance. Curved runs additionally expose a direct-exponential overflow in reactant-partition evaluation.
- The scalar native surface has no four-thread speedup. Four OpenMP threads slow the two-atom reference case, as expected when scheduling dominates potential work.

The table reports the largest sample count. Values are medians with interquartile ranges; profile and barrier differences are paired by seed.

| case | planes | T | threads | max |dF| | barrier |dF| | combined profile SE | rgflux wall (s) | reference wall (s) | wall p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| curved | 17 | 0.05 | 1 | 3.09e-06 [2.107e-06, 4.154e-06] | 3.085e-06 [1.568e-06, 4.071e-06] | 7.903e-07 | 0.005717 [0.005707, 0.005743] | 0.05258 [0.05228, 0.05273] | 3.97e-05 |
| curved | 17 | 0.05 | 4 | 3.064e-06 [1.72e-06, 5.052e-06] | 1.757e-06 [1.436e-06, 4.468e-06] | 1.765e-06 | 0.00584 [0.005756, 0.005857] | 0.06112 [0.06084, 0.06182] | 0.000108 |
| curved | 17 | 0.1 | 1 | 1.212e-05 [8.844e-06, 1.368e-05] | 4.177e-06 [1.835e-06, 1.191e-05] | 5.663e-06 | 0.005796 [0.005766, 0.005856] | 0.06452 [0.06422, 0.0648] | 0.000143 |
| curved | 17 | 0.1 | 4 | 1.543e-05 [8.632e-06, 1.897e-05] | 6.296e-06 [3.708e-06, 1.407e-05] | 4.912e-06 | 0.005775 [0.005737, 0.006122] | 0.1123 [0.1116, 0.1131] | 0.000143 |
| curved | 17 | 0.2 | 1 | 1.481e-05 [8.388e-06, 2.531e-05] | 6.632e-06 [3.907e-06, 9.601e-06] | 6.774e-06 | 0.006371 [0.006149, 0.006481] | 0.06833 [0.06724, 0.07066] | 3.97e-05 |
| curved | 17 | 0.2 | 4 | 8.639e-06 [6.79e-06, 1.992e-05] | 4.268e-06 [3.959e-06, 6.778e-06] | 7.034e-06 | 0.006472 [0.005849, 0.006571] | 0.1138 [0.1099, 0.1171] | 8.73e-05 |
| curved | 33 | 0.05 | 1 | 2.767e-07 [1.594e-07, 3.764e-07] | 2.764e-07 [1.545e-07, 3.668e-07] | 8.531e-08 | 0.01131 [0.01124, 0.01144] | 0.09501 [0.09422, 0.09608] | 8.73e-05 |
| curved | 33 | 0.05 | 4 | 2.684e-07 [1.467e-07, 3.826e-07] | 1.66e-07 [1.33e-07, 3.826e-07] | 8.965e-08 | 0.01125 [0.01116, 0.01132] | 0.1113 [0.109, 0.1128] | 0.000143 |
| curved | 33 | 0.1 | 1 | 1.317e-06 [1.033e-06, 1.691e-06] | 6.693e-07 [2.532e-07, 1.493e-06] | 4.697e-07 | 0.01315 [0.01166, 0.01457] | 0.1304 [0.1218, 0.1424] | 1.59e-05 |
| curved | 33 | 0.1 | 4 | 1.049e-06 [5.192e-07, 1.79e-06] | 4.126e-07 [1.37e-07, 6.138e-07] | 4.266e-07 | 0.01233 [0.01188, 0.01333] | 0.2143 [0.2093, 0.2185] | 3.97e-05 |
| curved | 33 | 0.2 | 1 | 2.099e-06 [1.504e-06, 3.148e-06] | 1.921e-06 [1.034e-06, 2.469e-06] | 5.977e-07 | 0.01139 [0.01122, 0.01153] | 0.1184 [0.118, 0.1188] | 1.59e-05 |
| curved | 33 | 0.2 | 4 | 1.901e-06 [1.011e-06, 3.617e-06] | 6.875e-07 [5.22e-07, 1.403e-06] | 6.212e-07 | 0.01121 [0.01116, 0.01134] | 0.2055 [0.2039, 0.2095] | 0.000143 |
| straight | 9 | 0.05 | 1 | n/a | n/a | n/a | n/a | 0.03301 [0.03242, 0.03322] | n/a |
| straight | 9 | 0.05 | 4 | n/a | n/a | n/a | n/a | 0.03642 [0.03624, 0.03667] | n/a |
| straight | 9 | 0.1 | 1 | n/a | n/a | n/a | n/a | 0.04138 [0.03934, 0.04385] | n/a |
| straight | 9 | 0.1 | 4 | n/a | n/a | n/a | n/a | 0.06467 [0.06395, 0.06538] | n/a |
| straight | 9 | 0.2 | 1 | n/a | n/a | n/a | n/a | 0.03949 [0.0393, 0.04013] | n/a |
| straight | 9 | 0.2 | 4 | n/a | n/a | n/a | n/a | 0.06728 [0.06505, 0.07002] | n/a |
| straight | 17 | 0.05 | 1 | 0.03204 [0.02749, 0.04132] | 0.003838 [0.002191, 0.007057] | 0.002917 | 0.00589 [0.005811, 0.006002] | 0.05295 [0.05271, 0.05394] | 3.66e-05 |
| straight | 17 | 0.05 | 4 | 0.03426 [0.03027, 0.04101] | 0.00745 [0.004301, 0.0168] | 0.004605 | 0.005853 [0.005792, 0.005919] | 0.05921 [0.05895, 0.0602] | 3.66e-05 |
| straight | 17 | 0.1 | 1 | 0.04205 [0.032, 0.05364] | 0.03099 [0.007414, 0.04488] | 0.008345 | 0.005979 [0.00591, 0.006387] | 0.06609 [0.06563, 0.06715] | 3.66e-05 |
| straight | 17 | 0.1 | 4 | 0.02933 [0.02485, 0.03874] | 0.02548 [0.01586, 0.03461] | 0.007432 | 0.005879 [0.005845, 0.005939] | 0.1152 [0.1148, 0.1174] | 3.66e-05 |
| straight | 17 | 0.2 | 1 | 0.06216 [0.05079, 0.1092] | 0.06216 [0.05079, 0.1089] | 0.01175 | 0.005914 [0.005861, 0.005951] | 0.06635 [0.06451, 0.06664] | 3.66e-05 |
| straight | 17 | 0.2 | 4 | 0.05491 [0.04435, 0.07351] | 0.05433 [0.04112, 0.07319] | 0.009035 | 0.005901 [0.005736, 0.005922] | 0.1125 [0.1115, 0.1139] | 3.66e-05 |

### Cost and force calls at the largest sample count

| case | planes | T | threads | rgflux calls/plane | reference calls/plane | rgflux overhead/plane (s) |
|---|---:|---:|---:|---:|---:|---:|
| curved | 17 | 0.05 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0002988 [0.0002981, 0.0003] |
| curved | 17 | 0.05 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003054 [0.0002999, 0.0003062] |
| curved | 17 | 0.1 | 1 | 386 [386, 386] | 384 [384, 384] | 0.000303 [0.0003016, 0.0003042] |
| curved | 17 | 0.1 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003021 [0.0002999, 0.0003194] |
| curved | 17 | 0.2 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003336 [0.0003219, 0.00034] |
| curved | 17 | 0.2 | 4 | 386 [386, 386] | 384 [384, 384] | 0.000338 [0.0003057, 0.0003437] |
| curved | 33 | 0.05 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003032 [0.0003026, 0.0003074] |
| curved | 33 | 0.05 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003028 [0.0003003, 0.0003049] |
| curved | 33 | 0.1 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003542 [0.0003122, 0.0003858] |
| curved | 33 | 0.1 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003313 [0.0003201, 0.0003594] |
| curved | 33 | 0.2 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003066 [0.0003008, 0.0003103] |
| curved | 33 | 0.2 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003014 [0.0003003, 0.000305] |
| straight | 9 | 0.05 | 1 | n/a | 384 [384, 384] | n/a |
| straight | 9 | 0.05 | 4 | n/a | 384 [384, 384] | n/a |
| straight | 9 | 0.1 | 1 | n/a | 384 [384, 384] | n/a |
| straight | 9 | 0.1 | 4 | n/a | 384 [384, 384] | n/a |
| straight | 9 | 0.2 | 1 | n/a | 384 [384, 384] | n/a |
| straight | 9 | 0.2 | 4 | n/a | 384 [384, 384] | n/a |
| straight | 17 | 0.05 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003085 [0.0003042, 0.0003144] |
| straight | 17 | 0.05 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003059 [0.000303, 0.0003094] |
| straight | 17 | 0.1 | 1 | 386 [386, 386] | 384 [384, 384] | 0.0003133 [0.0003092, 0.0003379] |
| straight | 17 | 0.1 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003073 [0.0003058, 0.000311] |
| straight | 17 | 0.2 | 1 | 386 [386, 386] | 384 [384, 384] | 0.00031 [0.0003061, 0.0003116] |
| straight | 17 | 0.2 | 4 | 386 [386, 386] | 384 [384, 384] | 0.0003087 [0.0003, 0.0003102] |

## Convergence and A/A noise

| case | planes | T | threads | samples | max |dF| | A/A profile noise | barrier p |
|---|---:|---:|---:|---:|---:|---:|---:|
| curved | 17 | 0.05 | 1 | 64 | 6.756e-06 [4.687e-06, 1.114e-05] | 1.072e-05 [7.784e-06, 1.594e-05] | 6.35e-05 |
| curved | 17 | 0.05 | 4 | 64 | 1.318e-05 [7.861e-06, 1.909e-05] | 1.856e-05 [1.464e-05, 2.728e-05] | 3.97e-05 |
| curved | 17 | 0.05 | 1 | 128 | 7.459e-06 [6.236e-06, 1.142e-05] | 6.425e-06 [4.853e-06, 7.641e-06] | 0.000143 |
| curved | 17 | 0.05 | 4 | 128 | 1e-05 [8.597e-06, 2.008e-05] | 6.966e-06 [6.543e-06, 7.388e-06] | 0.000323 |
| curved | 17 | 0.05 | 1 | 256 | 3.09e-06 [2.107e-06, 4.154e-06] | 4.439e-06 [4.162e-06, 4.566e-06] | 3.97e-05 |
| curved | 17 | 0.05 | 4 | 256 | 3.064e-06 [1.72e-06, 5.052e-06] | 5.964e-06 [3.978e-06, 8.92e-06] | 0.000108 |
| curved | 17 | 0.1 | 1 | 64 | 3.336e-05 [2.697e-05, 3.979e-05] | 2.936e-05 [1.943e-05, 3.029e-05] | 0.829 |
| curved | 17 | 0.1 | 4 | 64 | 3.081e-05 [1.935e-05, 4.803e-05] | 2.88e-05 [1.686e-05, 3.798e-05] | 0.432 |
| curved | 17 | 0.1 | 1 | 128 | 3.62e-05 [2.323e-05, 5.305e-05] | 1.483e-05 [9.188e-06, 2.005e-05] | 0.135 |
| curved | 17 | 0.1 | 4 | 128 | 3.579e-05 [2.91e-05, 5.458e-05] | 2.947e-05 [2.171e-05, 4.118e-05] | 0.964 |
| curved | 17 | 0.1 | 1 | 256 | 1.212e-05 [8.844e-06, 1.368e-05] | 6.516e-06 [5.798e-06, 9.204e-06] | 0.271 |
| curved | 17 | 0.1 | 4 | 256 | 1.543e-05 [8.632e-06, 1.897e-05] | 6.868e-06 [5.322e-06, 8.25e-06] | 0.859 |
| curved | 17 | 0.2 | 1 | 64 | 5.536e-05 [4.818e-05, 6.692e-05] | 3.903e-05 [3.569e-05, 4.656e-05] | 0.531 |
| curved | 17 | 0.2 | 4 | 64 | 4.313e-05 [2.576e-05, 9.699e-05] | 1.149e-05 [1.052e-05, 1.489e-05] | 0.448 |
| curved | 17 | 0.2 | 1 | 128 | 4.484e-05 [1.914e-05, 5.923e-05] | 2.847e-05 [2.337e-05, 3.271e-05] | 0.241 |
| curved | 17 | 0.2 | 4 | 128 | 5.029e-05 [3.563e-05, 8.92e-05] | 2.084e-05 [1.445e-05, 5.866e-05] | 0.167 |
| curved | 17 | 0.2 | 1 | 256 | 1.481e-05 [8.388e-06, 2.531e-05] | 6.127e-06 [5.127e-06, 9.644e-06] | 0.482 |
| curved | 17 | 0.2 | 4 | 256 | 8.639e-06 [6.79e-06, 1.992e-05] | 9.713e-06 [5.348e-06, 1.273e-05] | 0.0806 |
| curved | 33 | 0.05 | 1 | 64 | 6.595e-07 [4.838e-07, 1.451e-06] | 3.979e-07 [3.957e-07, 1.109e-06] | 8.73e-05 |
| curved | 33 | 0.05 | 4 | 64 | 5.235e-07 [4.598e-07, 1.591e-06] | 3.055e-06 [3.055e-06, 3.055e-06] | 0.0044 |
| curved | 33 | 0.05 | 1 | 128 | 6.508e-07 [4.981e-07, 7.443e-07] | 9.109e-07 [8.826e-07, 9.504e-07] | 0.000191 |
| curved | 33 | 0.05 | 4 | 128 | 7.058e-07 [5.643e-07, 9.355e-07] | 7.532e-07 [5.11e-07, 1.279e-06] | 1.59e-05 |
| curved | 33 | 0.05 | 1 | 256 | 2.767e-07 [1.594e-07, 3.764e-07] | 3.479e-07 [2.208e-07, 5.018e-07] | 8.73e-05 |
| curved | 33 | 0.05 | 4 | 256 | 2.684e-07 [1.467e-07, 3.826e-07] | 2.898e-07 [2.29e-07, 4.059e-07] | 0.000143 |
| curved | 33 | 0.1 | 1 | 64 | 2.734e-06 [8.942e-07, 4.298e-06] | 8.843e-07 [7.751e-07, 1.105e-06] | 0.571 |
| curved | 33 | 0.1 | 4 | 64 | 2.611e-06 [1.88e-06, 4.49e-06] | 1.224e-06 [7.929e-07, 1.729e-06] | 0.427 |
| curved | 33 | 0.1 | 1 | 128 | 2.51e-06 [1.654e-06, 3.261e-06] | 8.277e-07 [5.018e-07, 1.146e-06] | 0.892 |
| curved | 33 | 0.1 | 4 | 128 | 1.246e-06 [1.001e-06, 1.807e-06] | 1.25e-06 [1.165e-06, 1.562e-06] | 0.616 |
| curved | 33 | 0.1 | 1 | 256 | 1.317e-06 [1.033e-06, 1.691e-06] | 8.122e-07 [7.365e-07, 9.463e-07] | 0.181 |
| curved | 33 | 0.1 | 4 | 256 | 1.049e-06 [5.192e-07, 1.79e-06] | 2.558e-07 [2.222e-07, 4.99e-07] | 0.299 |
| curved | 33 | 0.2 | 1 | 64 | 2.399e-06 [1.878e-06, 3.332e-06] | 1.905e-06 [1.497e-06, 2.433e-06] | 0.157 |
| curved | 33 | 0.2 | 4 | 64 | 5.721e-06 [3.691e-06, 6.602e-06] | 1.35e-06 [8.277e-07, 1.448e-06] | 0.0518 |
| curved | 33 | 0.2 | 1 | 128 | 2.85e-06 [1.413e-06, 4.122e-06] | 1.136e-06 [9.606e-07, 1.507e-06] | 0.546 |
| curved | 33 | 0.2 | 4 | 128 | 3.542e-06 [3.146e-06, 4.82e-06] | 1.652e-06 [1.493e-06, 1.873e-06] | 0.437 |
| curved | 33 | 0.2 | 1 | 256 | 2.099e-06 [1.504e-06, 3.148e-06] | 1.075e-06 [5.159e-07, 1.668e-06] | 0.305 |
| curved | 33 | 0.2 | 4 | 256 | 1.901e-06 [1.011e-06, 3.617e-06] | 7.369e-07 [6.772e-07, 8.357e-07] | 0.337 |
| straight | 9 | 0.05 | 1 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.05 | 4 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.05 | 1 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.05 | 4 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.05 | 1 | 256 | n/a | n/a | n/a |
| straight | 9 | 0.05 | 4 | 256 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 1 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 4 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 1 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 4 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 1 | 256 | n/a | n/a | n/a |
| straight | 9 | 0.1 | 4 | 256 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 1 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 4 | 64 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 1 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 4 | 128 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 1 | 256 | n/a | n/a | n/a |
| straight | 9 | 0.2 | 4 | 256 | n/a | n/a | n/a |
| straight | 17 | 0.05 | 1 | 64 | 0.01543 [0.01021, 0.01791] | 0.01514 [0.01117, 0.02029] | 0.175 |
| straight | 17 | 0.05 | 4 | 64 | 0.01727 [0.01323, 0.01999] | 0.01274 [0.007324, 0.02666] | 1 |
| straight | 17 | 0.05 | 1 | 128 | 0.02268 [0.01845, 0.03355] | 0.02045 [0.01587, 0.02525] | 0.0061 |
| straight | 17 | 0.05 | 4 | 128 | 0.02663 [0.01794, 0.03256] | 0.01655 [0.01098, 0.0221] | 0.507 |
| straight | 17 | 0.05 | 1 | 256 | 0.03204 [0.02749, 0.04132] | 0.0163 [0.0107, 0.02128] | 0.507 |
| straight | 17 | 0.05 | 4 | 256 | 0.03426 [0.03027, 0.04101] | 0.01755 [0.01199, 0.02518] | 1 |
| straight | 17 | 0.1 | 1 | 64 | 0.02302 [0.01757, 0.04429] | 0.01642 [0.01036, 0.04124] | 0.141 |
| straight | 17 | 0.1 | 4 | 64 | 0.03158 [0.02084, 0.04019] | 0.02178 [0.01245, 0.03398] | 0.544 |
| straight | 17 | 0.1 | 1 | 128 | 0.03126 [0.02453, 0.03738] | 0.03877 [0.02941, 0.05106] | 0.371 |
| straight | 17 | 0.1 | 4 | 128 | 0.03811 [0.01754, 0.04584] | 0.03029 [0.02233, 0.03703] | 0.141 |
| straight | 17 | 0.1 | 1 | 256 | 0.04205 [0.032, 0.05364] | 0.03757 [0.03291, 0.04718] | 0.000731 |
| straight | 17 | 0.1 | 4 | 256 | 0.02933 [0.02485, 0.03874] | 0.01948 [0.01661, 0.03755] | 0.237 |
| straight | 17 | 0.2 | 1 | 64 | 0.04577 [0.03888, 0.0545] | 0.03545 [0.02778, 0.0368] | 0.000731 |
| straight | 17 | 0.2 | 4 | 64 | 0.03818 [0.03227, 0.04938] | 0.0345 [0.02658, 0.04212] | 0.000246 |
| straight | 17 | 0.2 | 1 | 128 | 0.04845 [0.03834, 0.054] | 0.04131 [0.0239, 0.05913] | 0.000592 |
| straight | 17 | 0.2 | 4 | 128 | 0.04457 [0.03948, 0.05455] | 0.02217 [0.01901, 0.02699] | 7.66e-05 |
| straight | 17 | 0.2 | 1 | 256 | 0.06216 [0.05079, 0.1092] | 0.04396 [0.03476, 0.05131] | 7.66e-05 |
| straight | 17 | 0.2 | 4 | 256 | 0.05491 [0.04435, 0.07351] | 0.03641 [0.02575, 0.05176] | 9.73e-05 |

## Comparator and protocol

- rgflux: `5949ac5e905a221acbc5cd3d9345e51db1bb3fcd`
- rgpot: `424629e5ba9c0948b8d6224f5b49c014ffb4ac7c`
- rgsaddle: `92a74493d16364a1ba8542a63598ce8bcee5387c`
- rgmin: `eb8d3e4123183bd3f5ca82852332eeddb5d17f54`
- LAMMPS: `stable_22Jul2025_update6` (`9c5ab448c78a14fd534619622162ba418d6a1fb1`)
- PAFI driver: `v0.9` (`3a9070058bfaf86bcdeb4ec14023ee084376d291`), package 0.9.9.1
- Potential: Morse D=1, rho=6, r_e=1, cutoff=4; rgpot-pair 3.7.0 and LAMMPS morse/omp
- Build: release; rustc 1.100.0; GCC 13.3.0; CMake 3.28.3; Open MPI 4.1.6; MPI and OpenMP enabled

The straight case stretches a symmetric Morse dimer from r=1.0 to r=1.4. The curved case rotates the equilibrium dimer through a quarter circle while preserving its centre of mass. Both are finite two-atom systems in a box wider than twice the cutoff, so no periodic image enters the interaction. The rgpot pair kernel and LAMMPS `morse/omp` use D=1, rho=6, r_e=1, and cutoff=4.

Plane counts are explicit in every cell: 9 and 17 for the straight path, 17 and 33 for the curved path. The zero-temperature tolerance remains 0.01. Rejections are retained in the hit-rate table and are never rerun.

The matched setting was declared before collection: dt=0.002, friction=1.0, identical burn-in and production counts, and the same seeds, path nodes, masses, potential constants, and plane counts. Other settings use each implementation's documented defaults. Runs are release builds, fixed to the declared CPU sets, with one and four requested threads. rgflux currently evaluates this scalar pair surface on one effective thread; that limitation is recorded rather than hidden.

The reported LAMMPS force-call count covers thermalization and production integration steps. Setup `run 0` evaluations are not exposed by the reference driver. rgflux reports measured potential calls and separates wall time accumulated inside the potential from integrator overhead.

Static minimum energies agree exactly in accepted paired runs. The largest median saddle-energy discrepancy is below 3e-11. At 256 samples, rgflux reports 386 force calls per accepted plane and the reference reports 384 integration-step calls per plane; the two extra native calls cover path/setup evaluation.

## Reproduction of the rgflux numerical rejection

A fixed curved-path seed can make direct evaluation of `exp(-F/kBT)` overflow while integrating the reactant partition. `repro_partition_overflow.py` checks the retained `PAFI reactant partition must be finite and positive` rejection. It requires the release native executable through `--native-bin`. The benchmark does not alter rgflux or retry the rejected run.

## Reproduction

Build and run commands are documented in `bench/pafi/README.md`. Raw logs are intentionally excluded.
