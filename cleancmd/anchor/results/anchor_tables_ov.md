# Object-picking benchmark (1pXnuDYAj8r,5LpN3gDmAk7,gTV8FGcVJC9,jh4fc5c5qoQ,JmbYfDe2QKZ_1,JmbYfDe2QKZ_2,mJXqzFtmKg4,ur6pFq6Qu1A,UwV83HsGsw3,Vt2qJdWjCF2,YmJkqBEsHnH,zsNo4HB9uLZ)

Correct = chosen detection centre within 1.0 m of the annotated instance centre. LLM: qwen3.5:latest, options {'temperature': 0, 'seed': 0, 'num_predict': 8, 'num_ctx': 2048}.

## Per command form
| method | form | n | accuracy | ask rate | missed det. | wrong choice | asked (target detected) |
|---|---|---|---|---|---|---|---|
| vlmaps_front | bare | 3018 | 4.2% | 39.2% | 1623 | 863 | 404 |
| vlmaps_front | landmark | 585 | 4.3% | 46.7% | 348 | 135 | 77 |
| vlmaps_front | room | 264 | 4.5% | 40.2% | 132 | 79 | 41 |
| vlmaps_front | all | 3867 | 4.3% | 40.4% | 2103 | 1077 | 522 |
| vlmaps_nearest | bare | 3018 | 6.1% | 10.6% | 1623 | 1212 | 0 |
| vlmaps_nearest | landmark | 585 | 7.9% | 14.9% | 348 | 191 | 0 |
| vlmaps_nearest | room | 264 | 9.8% | 10.2% | 132 | 106 | 0 |
| vlmaps_nearest | all | 3867 | 6.6% | 11.2% | 2103 | 1509 | 0 |
| room_first | bare | 3018 | 2.3% | 57.8% | 1623 | 635 | 692 |
| room_first | landmark | 585 | 2.1% | 68.5% | 348 | 83 | 142 |
| room_first | room | 264 | 39.0% | 42.0% | 132 | 29 | 0 |
| room_first | all | 3867 | 4.7% | 58.3% | 2103 | 747 | 834 |
| c_gated | bare | 3018 | 6.1% | 10.6% | 1623 | 1212 | 0 |
| c_gated | landmark | 585 | 7.9% | 14.9% | 348 | 191 | 0 |
| c_gated | room | 264 | 39.0% | 42.0% | 132 | 29 | 0 |
| c_gated | all | 3867 | 8.6% | 13.4% | 2103 | 1432 | 0 |

## Per scene
| method | scene | n | accuracy | ask rate | missed det. | wrong choice | asked (target detected) |
|---|---|---|---|---|---|---|---|
| vlmaps_front | 1pXnuDYAj8r | 471 | 4.0% | 26.8% | 222 | 186 | 44 |
| vlmaps_front | 5LpN3gDmAk7 | 285 | 6.3% | 38.6% | 114 | 101 | 52 |
| vlmaps_front | JmbYfDe2QKZ_1 | 120 | 6.7% | 61.7% | 60 | 24 | 28 |
| vlmaps_front | JmbYfDe2QKZ_2 | 255 | 7.8% | 27.5% | 78 | 110 | 47 |
| vlmaps_front | UwV83HsGsw3 | 348 | 4.0% | 51.7% | 261 | 29 | 44 |
| vlmaps_front | Vt2qJdWjCF2 | 294 | 4.1% | 57.1% | 198 | 38 | 46 |
| vlmaps_front | YmJkqBEsHnH | 111 | 6.3% | 18.0% | 69 | 26 | 9 |
| vlmaps_front | gTV8FGcVJC9 | 357 | 2.8% | 41.7% | 222 | 82 | 43 |
| vlmaps_front | jh4fc5c5qoQ | 240 | 7.9% | 54.2% | 147 | 32 | 42 |
| vlmaps_front | mJXqzFtmKg4 | 576 | 1.6% | 35.9% | 294 | 216 | 57 |
| vlmaps_front | ur6pFq6Qu1A | 438 | 0.9% | 44.5% | 309 | 81 | 44 |
| vlmaps_front | zsNo4HB9uLZ | 372 | 6.7% | 35.8% | 129 | 152 | 66 |
| vlmaps_front | all | 3867 | 4.3% | 40.4% | 2103 | 1077 | 522 |
| vlmaps_nearest | 1pXnuDYAj8r | 471 | 5.5% | 9.6% | 222 | 223 | 0 |
| vlmaps_nearest | 5LpN3gDmAk7 | 285 | 7.0% | 6.3% | 114 | 151 | 0 |
| vlmaps_nearest | JmbYfDe2QKZ_1 | 120 | 15.8% | 27.5% | 60 | 41 | 0 |
| vlmaps_nearest | JmbYfDe2QKZ_2 | 255 | 11.0% | 0.0% | 78 | 149 | 0 |
| vlmaps_nearest | UwV83HsGsw3 | 348 | 5.7% | 15.5% | 261 | 67 | 0 |
| vlmaps_nearest | Vt2qJdWjCF2 | 294 | 5.4% | 4.1% | 198 | 80 | 0 |
| vlmaps_nearest | YmJkqBEsHnH | 111 | 7.2% | 0.0% | 69 | 34 | 0 |
| vlmaps_nearest | gTV8FGcVJC9 | 357 | 6.7% | 19.3% | 222 | 111 | 0 |
| vlmaps_nearest | jh4fc5c5qoQ | 240 | 12.1% | 11.2% | 147 | 64 | 0 |
| vlmaps_nearest | mJXqzFtmKg4 | 576 | 3.5% | 17.2% | 294 | 262 | 0 |
| vlmaps_nearest | ur6pFq6Qu1A | 438 | 2.3% | 6.8% | 309 | 119 | 0 |
| vlmaps_nearest | zsNo4HB9uLZ | 372 | 9.4% | 12.9% | 129 | 208 | 0 |
| vlmaps_nearest | all | 3867 | 6.6% | 11.2% | 2103 | 1509 | 0 |
| room_first | 1pXnuDYAj8r | 471 | 2.3% | 39.3% | 222 | 164 | 74 |
| room_first | 5LpN3gDmAk7 | 285 | 11.9% | 55.8% | 114 | 63 | 74 |
| room_first | JmbYfDe2QKZ_1 | 120 | 5.0% | 55.8% | 60 | 32 | 22 |
| room_first | JmbYfDe2QKZ_2 | 255 | 7.8% | 42.7% | 78 | 84 | 73 |
| room_first | UwV83HsGsw3 | 348 | 4.6% | 65.8% | 261 | 27 | 44 |
| room_first | Vt2qJdWjCF2 | 294 | 2.0% | 79.6% | 198 | 20 | 70 |
| room_first | YmJkqBEsHnH | 111 | 7.2% | 8.1% | 69 | 28 | 6 |
| room_first | gTV8FGcVJC9 | 357 | 3.9% | 59.4% | 222 | 52 | 69 |
| room_first | jh4fc5c5qoQ | 240 | 5.0% | 72.9% | 147 | 15 | 66 |
| room_first | mJXqzFtmKg4 | 576 | 4.0% | 59.9% | 294 | 138 | 121 |
| room_first | ur6pFq6Qu1A | 438 | 1.1% | 67.1% | 309 | 46 | 78 |
| room_first | zsNo4HB9uLZ | 372 | 7.5% | 64.0% | 129 | 78 | 137 |
| room_first | all | 3867 | 4.7% | 58.3% | 2103 | 747 | 834 |
| c_gated | 1pXnuDYAj8r | 471 | 6.2% | 9.6% | 222 | 220 | 0 |
| c_gated | 5LpN3gDmAk7 | 285 | 16.5% | 10.5% | 114 | 124 | 0 |
| c_gated | JmbYfDe2QKZ_1 | 120 | 15.8% | 27.5% | 60 | 41 | 0 |
| c_gated | JmbYfDe2QKZ_2 | 255 | 14.1% | 3.5% | 78 | 141 | 0 |
| c_gated | UwV83HsGsw3 | 348 | 8.0% | 18.1% | 261 | 59 | 0 |
| c_gated | Vt2qJdWjCF2 | 294 | 5.4% | 7.1% | 198 | 80 | 0 |
| c_gated | YmJkqBEsHnH | 111 | 7.2% | 2.7% | 69 | 34 | 0 |
| c_gated | gTV8FGcVJC9 | 357 | 7.8% | 20.2% | 222 | 107 | 0 |
| c_gated | jh4fc5c5qoQ | 240 | 13.3% | 11.2% | 147 | 61 | 0 |
| c_gated | mJXqzFtmKg4 | 576 | 5.7% | 21.9% | 294 | 249 | 0 |
| c_gated | ur6pFq6Qu1A | 438 | 2.3% | 8.2% | 309 | 119 | 0 |
| c_gated | zsNo4HB9uLZ | 372 | 12.4% | 14.5% | 129 | 197 | 0 |
| c_gated | all | 3867 | 8.6% | 13.4% | 2103 | 1432 | 0 |
