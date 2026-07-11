"""Patient-level SSL split has no patient leakage across labeled/unlabeled."""
from endosemi.data.ssl_split import patient_id_of, make_ssl_split


def test_patient_id_parsing():
    assert patient_id_of("/x/R0003_20220704_100122_000.jpg") == "R0003"
    assert patient_id_of("S0640_20231104_113307_020.jpg") == "S0640"


def test_no_patient_leakage(tmp_path):
    train = tmp_path / "train.txt"
    lines = [f"/data/R{p:04d}_{f:03d}.jpg" for p in range(1, 21) for f in range(5)]
    train.write_text("\n".join(lines) + "\n")

    labeled, unlabeled = make_ssl_split(train, label_ratio=0.25, seed=0)

    lab_pat = {patient_id_of(x) for x in labeled}
    unl_pat = {patient_id_of(x) for x in unlabeled}
    assert lab_pat.isdisjoint(unl_pat)          # no patient in both pools
    assert len(lab_pat) == 5                    # 25% of 20 patients
    assert len(labeled) + len(unlabeled) == len(lines)
