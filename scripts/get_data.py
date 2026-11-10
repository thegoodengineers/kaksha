"""Download ESA's Collision Avoidance Challenge dataset (CC-BY-4.0) into data/.

Zenodo record 10.5281/zenodo.4463683. 221 MB zip, checked by MD5, then unzipped.
"""
import hashlib
import pathlib
import urllib.request
import zipfile

URL = ("https://zenodo.org/api/records/4463683/files/"
       "Collision%20Avoidance%20Challenge%20-%20Dataset.zip/content")
MD5 = "d19dc8875229f2f6893253c38adddc87"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    zpath = DATA / "dataset.zip"
    if not zpath.exists() or hashlib.md5(zpath.read_bytes()).hexdigest() != MD5:
        print("downloading 221 MB from Zenodo ...")
        urllib.request.urlretrieve(URL, zpath)
    got = hashlib.md5(zpath.read_bytes()).hexdigest()
    if got != MD5:
        raise SystemExit(f"MD5 mismatch: {got} != {MD5}")
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            if "kelvins_competition_data/" in m and m.endswith((".csv", ".zip")):
                out = DATA / pathlib.Path(m).name
                if not out.exists():
                    out.write_bytes(z.read(m))
    with zipfile.ZipFile(DATA / "train_data.zip") as z:  # train set is a zip inside the zip
        for m in z.namelist():
            if m.endswith(".csv") and not (DATA / pathlib.Path(m).name).exists():
                (DATA / pathlib.Path(m).name).write_bytes(z.read(m))
    for f in sorted(DATA.glob("*.csv")):
        print("ok", f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
