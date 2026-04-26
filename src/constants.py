

expensive_thinning = {
        "num_seq": 10,
        "num_sample": 1,
        "num_exp": 500, # number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
        "look_ahead_time": 10,
        "patience_counter": 5, # the maximum iteration used in adaptive thinning
        "over_sample_rate": 5,
        "num_samples_boundary": 5,
        "dtime_max": 5
        }

cheap_thinning = {
        "num_seq": 3,
        "num_sample": 1,
        "num_exp": 4, 
        "look_ahead_time": 4,
        "patience_counter": 5,
        "over_sample_rate": 5,
        "num_samples_boundary": 5,
        "dtime_max": 5
    }

