# Overall scale:
len_probes = 58692
len_probes_in_richiardi = 16906
len_samples = 3702
len_samples_cortical = 1777

# Sample groupings:
sub = [         'H03511009',  # L only, 363 samples ( 180 cortical)
                'H03511012',  # L only, 529 samples ( 260 cortical)
                'H03511015',  # L only, 470 samples ( 222 cortical)
                'H03511016',  # L only, 501 samples ( 237 cortical)
                'H03512001',  # L + R,  946 samples ( 498 cortical)
                'H03512002',  # L + R,  893 samples ( 380 cortical)
                'A',          # all,   3702 samples (1777 cortical)
]
hem = [         'L',          # MNI coordinates < 0.0
                'R',          # MNI coordinates > 0.0
                'A',          # all samples
]  # even left-only donors have mid-brain samples from both >0 and <0
samp = [        'cor',        # samples from cortex
                'sub',        # samples not from cortex
                'all',        # all samples
                'glasser',  # samples mappable into glasser parcels
]

# Algorithms:
tgt = [         'min',        # minimize the correlation, drive it toward -1
                'max',        # maximize the correlation, drive it toward +1
]
algo = [        'once',       # Order by r once, then remove genes one-at-a-time
                'smrt',       # Order by r once, then again when succeeding r falls
                'evry',       # Order by r once, then every time a gene is removed
]

# Comparators:
comp = [        'conn',       # for now, just NKI
                'cons',       # connectivity similarity (NKI)
                'dist',       # euclidean distance matrix based on MNI coordinates
                'hcpc',       # FUTURE- Could be HCP fMRI connectivity, DWI, etc
                'dist2d',     # FUTURE- map MNI to cortex & flatten
]

# Adjustments:
adj = [         'NULL####',  # shuffled to create null distribution w/#### seed
                'distlin',   # partial regression from linear distances
                'distlog',   # partial regression of log(distance)
]

# Summary:
print("""
    Running a given algorithm on each donor, each hemisphere, each ...
    (7 donors * 3 hemis * 3 cortex * 2 direction * 3 algos) = 378 results files
    But we don't have R hemis for 4 donors, and can settle on the 'smart' algo ...
    (17 donor-hemisphere-combos * 3 cortex * 2 direction) = 102 results files
    Each result needs a null distribution, linear and log-corrected sets.
    (102 results * 10 nulls * 2 corrections) = 2040 results files
""")
