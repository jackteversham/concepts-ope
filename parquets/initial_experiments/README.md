# Parquet File Descriptions

## Methods for generating "good" policies
- Given a dataset which is a subset of the full dataset, containing trajectories which pass through a subset of concepts, we can generate good policies by average out the action distribution over each concept in the subset.


## Over concepts (0,1,3)
### ope_estimates_c_in_013_good_policy
- Policy used: [0.31353633, 0.15028605, 0.46337089, 0.07280675]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These concepts could be classified as "good" concepts as they don't have very severe wind.
- On policy value: -301

### ope_estimates_c_in_013_bad_policy
- Policy used: [0.15028605, 0.31353633, 0.46337089, 0.07280675]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These concepts could be classified as "good" concepts as they don't have very severe wind.
- On policy value: -301

## Over concepts (0,4)
### ope_estimates_c_in_04_good_policy
- Policy used: [0.49108653, 0.0755275, 0.35700847, 0.0763775]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These concepts could be classified as "bad" concepts as concept four has severe wind
- On policy value: -253

### ope_estimates_c_in_04_bad_policy
- Policy used: [0.35700847, 0.0755275, 0.49108653 , 0.0763775]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These concepts could be classified as "bad" concepts as concept four has severe wind
- On policy value: -253

## Over concepts (0,1,2,3)
### ope_estimates_c_in_0123_good_policy
- Policy used: [0.3447435,  0.14230822, 0.43917918, 0.0737691]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These trajectories have different starting points in the state space, resulting in different trajectroy lengths and potentially more variance
- On policy value: -365

### ope_estimates_c_in_0123_bad_policy
- Policy used:  [0.43917918,  0.14230822, 0.3447435, 0.0737691]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These trajectories have different starting points in the state space, resulting in different trajectroy lengths and potentially more variance
- On policy value: -365


## Over concepts (0,1,2,3,4) (all)
### ope_estimates_c_in_all_good_policy
- Policy used: [0.38660187, 0.1295562,  0.409095 ,  0.07474691]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These trajectories have different starting points in the state space, resulting in different trajectroy lengths and potentially more variance
- On policy value: -339

### ope_estimates_c_in_all_bad_policy
- Policy used: [0.409095, 0.1295562, 0.38660187, 0.07474691]
- Only 10 trials were conducted for each dataset size. More should be conducted.
- These trajectories have different starting points in the state space, resulting in different trajectroy lengths and potentially more variance
- On policy value: -339
