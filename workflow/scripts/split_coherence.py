from pathlib import Path
import simuran as smr

def main(input_table_location, out_dir):
    df = smr.load_table(input_table_location)
    regions = df["Brain regions"].unique()
    for r in regions:
        grouped = df[df["Brain regions"] == r]
        grouped = grouped[grouped["Estimated trial type"] != "Fail"]
        grouped.to_csv(out_dir / f"coherence_{r}.csv", index=False)
    
if __name__ == "__main__":
    main(snakemake.input[0], Path(snakemake.output[0]).parent)