import os

project = ["stad_PC1", "stad_PC2", "stad_PC3", "stad_PC4", "stad_PC5",
           "coad_PC1", "coad_PC2", "coad_PC3", "coad_PC4", "coad_PC5",
           "ucec_PC1", "ucec_PC2", "ucec_PC3", "ucec_PC4", "ucec_PC5",
           "stad_g", "stad_c", "stad_m","stad_gc", "stad_gm", "stad_cm",
           "coad_g", "coad_c", "coad_m", "coad_gc", "coad_gm", "coad_cm",
           "ucec_g", "ucec_c", "ucec_m", "ucec_gc", "ucec_gm", "ucec_cm",
           ]

cmd = ["python pglcn.py --project " + i for i in project]

for item in cmd:
    os.system(item)
