import fire
import glob
import pickle
import tarfile

from pathlib import Path
from tqdm import tqdm


class AudioSetExporter:
    def export(self, tar_files_prefix: str, output_dir: str):
        tar_lst = glob.glob(f"{tar_files_prefix}*.tar")
        for tar_filename in tqdm(tar_lst):
            entries = []
            with tarfile.open(tar_filename, "r") as db:
                for tarinfo in db:
                    entries.append(
                        (
                            Path(tarinfo.name).stem,
                            tar_filename,
                            tarinfo.offset_data,
                            tarinfo.size,
                        )
                    )
            # write index file and data file
            index_filename = Path(tar_filename).stem + ".pkl"
            with open(f"{output_dir}/{index_filename}", "wb") as idx_fp:
                pickle.dump(entries, idx_fp)


if __name__ == "__main__":
    exporter = AudioSetExporter()
    fire.Fire(exporter.export)
