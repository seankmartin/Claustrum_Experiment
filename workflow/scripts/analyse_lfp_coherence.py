# %%
import numpy as np
from bvmpc.bv_session import Session
import simuran as smr
from skm_pyutils.table import df_from_file
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from simuran import Recording

here = Path(__file__).parent.parent.parent
# %%

input_table_path = r"I:\Claustrum_Experiment\results\converted_data.csv"
df = df_from_file(input_table_path)
df = df[df["has_behaviour"]]
df = df[~df["directory"].str.contains("Batch_1")]
rc = smr.RecordingContainer.from_table(df, loader=smr.loader("NWB"))

# %%
r = rc[0]
r.source_file = here / r.source_file
r.load()

# %%
def analyse_lfp_coherence(recording : "Recording"):
    """Analyse the LFP coherence of the recording."""
    session = Session(recording=recording)
    info = extract_trial_info(session)
    return info

def extract_trial_info(session : "Session"):
    trial_ends = session.get_rw_ts()
    trial_types = session.info_arrays["Trial Type"]
    block_types = ["FR" if x == 1 else "FI" for x in trial_types]
    block_times = session.get_block_ends()
    trial_types = []
    for t in trial_ends:
        for i, b in enumerate(block_times):
            if t < b:
                trial_types.append(block_types[i])
                break 
    return {
        "trial_times": trial_ends,
        "trial_types": trial_types,
    }

print(analyse_lfp_coherence(r))
# %%
