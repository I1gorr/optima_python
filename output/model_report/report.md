# Optima Model Enrichment Report

- Functions analyzed: **672**
- Models analyzed: **7**
- Models: llama-3.2-3b-instruct, microsoft/phi-4-mini-reasoning, mistralai/ministral-3-3b, opencoder-1.5b-instruct, qwen2.5-1.5b-instruct, qwen2.5-coder-3b-instruct, smollm2-1.7b-instruct

## Overall Model Comparison

| model                          |   functions |   json_valid_rate |   success_rate |   avg_latency |   median_latency |   avg_input_tokens |   avg_output_tokens |   avg_total_tokens |   avg_keywords |   avg_concepts |   field_completeness_rate |
|:-------------------------------|------------:|------------------:|---------------:|--------------:|-----------------:|-------------------:|--------------------:|-------------------:|---------------:|---------------:|--------------------------:|
| mistralai/ministral-3-3b       |          96 |               100 |            100 |       6.07336 |           5.8385 |            3501.09 |            442.812  |            3943.91 |      5.05208   |       3.53125  |                   90.625  |
| microsoft/phi-4-mini-reasoning |          96 |               100 |            100 |       4.7737  |           4.576  |            2952.68 |            308.917  |            3261.59 |      5.90625   |       3.67708  |                   89.2045 |
| qwen2.5-coder-3b-instruct      |          96 |               100 |            100 |       3.04771 |           3.007  |            3282.4  |            244.938  |            3527.33 |      4.27083   |       3.47917  |                   85.4167 |
| llama-3.2-3b-instruct          |          96 |               100 |            100 |       2.79589 |           2.759  |            2967.44 |            212.427  |            3179.86 |      3.04167   |       2.4375   |                   84.1856 |
| qwen2.5-1.5b-instruct          |          96 |               100 |            100 |       2.36684 |           2.398  |            3282.4  |            241.698  |            3524.09 |      4.91667   |       3.53125  |                   80.303  |
| smollm2-1.7b-instruct          |          96 |               100 |            100 |       2.82843 |           2.7625 |            3729.96 |            200.312  |            3930.27 |      1.21875   |       0.697917 |                   63.6364 |
| opencoder-1.5b-instruct        |          96 |               100 |            100 |       5.32862 |           5.402  |            1527.57 |             81.0833 |            1608.66 |      0.0416667 |       0.03125  |                   46.5909 |

## Model × Function Complexity

| model                          | complexity_category   |   functions |   json_valid_rate |   success_rate |   avg_latency |   median_latency |   avg_input_tokens |   avg_output_tokens |   avg_total_tokens |   avg_keywords |   avg_concepts |   field_completeness_rate |
|:-------------------------------|:----------------------|------------:|------------------:|---------------:|--------------:|-----------------:|-------------------:|--------------------:|-------------------:|---------------:|---------------:|--------------------------:|
| llama-3.2-3b-instruct          | Simple                |          32 |               100 |            100 |       2.84772 |           2.788  |            2721.59 |            223.281  |            2944.88 |        3.09375 |        2.8125  |                   84.9432 |
| microsoft/phi-4-mini-reasoning | Simple                |          32 |               100 |            100 |       4.87447 |           4.6695 |            2714.12 |            316.75   |            3030.88 |        6.1875  |        4.0625  |                   90.3409 |
| mistralai/ministral-3-3b       | Simple                |          32 |               100 |            100 |       6.56284 |           5.872  |            3241.56 |            487.844  |            3729.41 |        4.90625 |        3.59375 |                   91.4773 |
| opencoder-1.5b-instruct        | Simple                |          32 |               100 |            100 |       5.55741 |           5.452  |            1497.69 |             88.625  |            1586.31 |        0.125   |        0.09375 |                   47.4432 |
| qwen2.5-1.5b-instruct          | Simple                |          32 |               100 |            100 |       2.41841 |           2.398  |            3018.38 |            251.062  |            3269.44 |        4.96875 |        3.75    |                   80.9659 |
| qwen2.5-coder-3b-instruct      | Simple                |          32 |               100 |            100 |       3.08316 |           3.03   |            3018.38 |            253.875  |            3272.25 |        4.5625  |        3.625   |                   84.9432 |
| smollm2-1.7b-instruct          | Simple                |          32 |               100 |            100 |       2.88225 |           2.867  |            3451    |            212.812  |            3663.81 |        1.6875  |        0.8125  |                   64.7727 |
| llama-3.2-3b-instruct          | Moderate              |          32 |               100 |            100 |       2.70509 |           2.704  |            3037.31 |            201.938  |            3239.25 |        2.9375  |        2.21875 |                   83.2386 |
| microsoft/phi-4-mini-reasoning | Moderate              |          32 |               100 |            100 |       4.62081 |           4.2915 |            3014.84 |            297.688  |            3312.53 |        5.0625  |        3.34375 |                   87.7841 |
| mistralai/ministral-3-3b       | Moderate              |          32 |               100 |            100 |       5.65581 |           5.2385 |            3585.38 |            407.062  |            3992.44 |        5       |        3.4375  |                   89.4886 |
| opencoder-1.5b-instruct        | Moderate              |          32 |               100 |            100 |       5.36225 |           5.371  |            1545.44 |             79.0625 |            1624.5  |        0       |        0       |                   45.7386 |
| qwen2.5-1.5b-instruct          | Moderate              |          32 |               100 |            100 |       2.28716 |           2.3725 |            3377.06 |            230.031  |            3607.09 |        4.5625  |        3.34375 |                   79.5455 |
| qwen2.5-coder-3b-instruct      | Moderate              |          32 |               100 |            100 |       2.81322 |           2.774  |            3377.06 |            219.438  |            3596.5  |        3.78125 |        3.28125 |                   84.0909 |
| smollm2-1.7b-instruct          | Moderate              |          32 |               100 |            100 |       2.82031 |           2.643  |            3802.38 |            192      |            3994.38 |        1.0625  |        0.59375 |                   63.6364 |
| llama-3.2-3b-instruct          | Complex               |          32 |               100 |            100 |       2.83484 |           2.792  |            3143.41 |            212.062  |            3355.47 |        3.09375 |        2.28125 |                   84.375  |
| microsoft/phi-4-mini-reasoning | Complex               |          32 |               100 |            100 |       4.82581 |           4.829  |            3129.06 |            312.312  |            3441.38 |        6.46875 |        3.625   |                   89.4886 |
| mistralai/ministral-3-3b       | Complex               |          32 |               100 |            100 |       6.00144 |           6.051  |            3676.34 |            433.531  |            4109.88 |        5.25    |        3.5625  |                   90.9091 |
| opencoder-1.5b-instruct        | Complex               |          32 |               100 |            100 |       5.06622 |           5.4055 |            1539.59 |             75.5625 |            1615.16 |        0       |        0       |                   46.5909 |
| qwen2.5-1.5b-instruct          | Complex               |          32 |               100 |            100 |       2.39497 |           2.4595 |            3451.75 |            244      |            3695.75 |        5.21875 |        3.5     |                   80.3977 |
| qwen2.5-coder-3b-instruct      | Complex               |          32 |               100 |            100 |       3.24675 |           3.089  |            3451.75 |            261.5    |            3713.25 |        4.46875 |        3.53125 |                   87.2159 |
| smollm2-1.7b-instruct          | Complex               |          32 |               100 |            100 |       2.78272 |           2.7065 |            3936.5  |            196.125  |            4132.62 |        0.90625 |        0.6875  |                   62.5    |

## Model × Function Type

| model                          | function_type   |   functions |   json_valid_rate |   success_rate |   avg_latency |   median_latency |   avg_input_tokens |   avg_output_tokens |   avg_total_tokens |   avg_keywords |   avg_concepts |   field_completeness_rate |
|:-------------------------------|:----------------|------------:|------------------:|---------------:|--------------:|-----------------:|-------------------:|--------------------:|-------------------:|---------------:|---------------:|--------------------------:|
| llama-3.2-3b-instruct          | Method          |          76 |               100 |            100 |       2.7728  |           2.7255 |            3075.66 |            208.592  |            3284.25 |      3.02632   |      2.42105   |                   83.8517 |
| llama-3.2-3b-instruct          | Test            |          20 |               100 |            100 |       2.8836  |           2.835  |            2556.2  |            227      |            2783.2  |      3.1       |      2.5       |                   85.4545 |
| microsoft/phi-4-mini-reasoning | Method          |          76 |               100 |            100 |       4.70436 |           4.475  |            3059.64 |            303.671  |            3363.32 |      5.46053   |      3.55263   |                   88.8756 |
| microsoft/phi-4-mini-reasoning | Test            |          20 |               100 |            100 |       5.0372  |           5.122  |            2546.2  |            328.85   |            2875.05 |      7.6       |      4.15      |                   90.4545 |
| mistralai/ministral-3-3b       | Method          |          76 |               100 |            100 |       6.07934 |           5.828  |            3643.91 |            437.013  |            4080.92 |      5.03947   |      3.46053   |                   90.1914 |
| mistralai/ministral-3-3b       | Test            |          20 |               100 |            100 |       6.05065 |           6.015  |            2958.4  |            464.85   |            3423.25 |      5.1       |      3.8       |                   92.2727 |
| opencoder-1.5b-instruct        | Method          |          76 |               100 |            100 |       5.43364 |           5.4395 |            1559.36 |             81.2105 |            1640.57 |      0.0526316 |      0.0394737 |                   46.7703 |
| opencoder-1.5b-instruct        | Test            |          20 |               100 |            100 |       4.92955 |           4.8995 |            1406.8  |             80.6    |            1487.4  |      0         |      0         |                   45.9091 |
| qwen2.5-1.5b-instruct          | Method          |          76 |               100 |            100 |       2.3547  |           2.355  |            3420.33 |            237.539  |            3657.87 |      4.81579   |      3.55263   |                   79.4258 |
| qwen2.5-1.5b-instruct          | Test            |          20 |               100 |            100 |       2.413   |           2.4325 |            2758.25 |            257.5    |            3015.75 |      5.3       |      3.45      |                   83.6364 |
| qwen2.5-coder-3b-instruct      | Method          |          76 |               100 |            100 |       3.00332 |           2.968  |            3420.33 |            237.934  |            3658.26 |      4.27632   |      3.44737   |                   85.0478 |
| qwen2.5-coder-3b-instruct      | Test            |          20 |               100 |            100 |       3.2164  |           3.096  |            2758.25 |            271.55   |            3029.8  |      4.25      |      3.6       |                   86.8182 |
| smollm2-1.7b-instruct          | Method          |          76 |               100 |            100 |       2.83847 |           2.7065 |            3861.93 |            197.5    |            4059.43 |      1.01316   |      0.710526  |                   63.1579 |
| smollm2-1.7b-instruct          | Test            |          20 |               100 |            100 |       2.79025 |           2.8185 |            3228.45 |            211      |            3439.45 |      2         |      0.65      |                   65.4545 |

## Model × Graph Complexity

| model                          | graph_complexity   |   functions |   json_valid_rate |   success_rate |   avg_latency |   median_latency |   avg_input_tokens |   avg_output_tokens |   avg_total_tokens |   avg_keywords |   avg_concepts |   field_completeness_rate |
|:-------------------------------|:-------------------|------------:|------------------:|---------------:|--------------:|-----------------:|-------------------:|--------------------:|-------------------:|---------------:|---------------:|--------------------------:|
| llama-3.2-3b-instruct          | Leaf               |          96 |               100 |            100 |       2.79589 |           2.759  |            2967.44 |            212.427  |            3179.86 |      3.04167   |       2.4375   |                   84.1856 |
| microsoft/phi-4-mini-reasoning | Leaf               |          96 |               100 |            100 |       4.7737  |           4.576  |            2952.68 |            308.917  |            3261.59 |      5.90625   |       3.67708  |                   89.2045 |
| mistralai/ministral-3-3b       | Leaf               |          96 |               100 |            100 |       6.07336 |           5.8385 |            3501.09 |            442.812  |            3943.91 |      5.05208   |       3.53125  |                   90.625  |
| opencoder-1.5b-instruct        | Leaf               |          96 |               100 |            100 |       5.32862 |           5.402  |            1527.57 |             81.0833 |            1608.66 |      0.0416667 |       0.03125  |                   46.5909 |
| qwen2.5-1.5b-instruct          | Leaf               |          96 |               100 |            100 |       2.36684 |           2.398  |            3282.4  |            241.698  |            3524.09 |      4.91667   |       3.53125  |                   80.303  |
| qwen2.5-coder-3b-instruct      | Leaf               |          96 |               100 |            100 |       3.04771 |           3.007  |            3282.4  |            244.938  |            3527.33 |      4.27083   |       3.47917  |                   85.4167 |
| smollm2-1.7b-instruct          | Leaf               |          96 |               100 |            100 |       2.82843 |           2.7625 |            3729.96 |            200.312  |            3930.27 |      1.21875   |       0.697917 |                   63.6364 |

## Complexity Definition

Function complexity is calculated from the combined source-code and LLVM representation size. Functions are divided into Simple, Moderate, and Complex using the lower and upper tertiles of the complete benchmark.

## Graph Complexity Definition

Graph complexity currently uses dependency count as a structural proxy: Leaf = 0 dependencies, Low = 1-2, Medium = 3-5, High = more than 5. CFG/call-graph metrics should replace this proxy once those metrics are populated reliably.

## Generated Graphs

- `complexity_completeness.png`
- `complexity_concepts.png`
- `complexity_keywords.png`
- `complexity_latency.png`
- `complexity_tokens.png`
- `function_type_completeness.png`
- `function_type_latency.png`
- `graph_complexity_completeness.png`
- `graph_complexity_latency.png`
