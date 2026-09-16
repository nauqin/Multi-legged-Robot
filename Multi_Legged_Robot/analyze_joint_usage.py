import numpy as np

d = np.load("joint_usage_flat.npz", allow_pickle=True)
pos, dones = d["joint_pos"], d["dones"]
names, default = list(d["joint_names"]), d["default_pos"]
dev = pos - default[None, None, :]

# 종료 전후 30스텝을 버린다 (넘어짐/리셋 구간 제거)
W = 30
keep = np.ones(dones.shape, bool)
T = dones.shape[0]
for t, e in zip(*np.where(dones)):
    keep[max(0, t - W):min(T, t + W), e] = False
print(f"유효 샘플 {keep.mean() * 100:.1f}%")

def report(pattern):
    idx = [i for i, n in enumerate(names) if n.endswith(pattern)]
    per_leg = []
    for i in idx:
        v = dev[:, :, i][keep]
        per_leg.append((names[i], v.mean(), v.std()))
    means = np.array([m for _, m, _ in per_leg])
    stds = np.array([s for _, _, s in per_leg])
    within = np.sqrt((stds ** 2).mean())
    between = means.std()
    print(f"\n{pattern}")
    print(f"  다리 내부 변동  {within:.4f}")
    print(f"  다리 간 오프셋  {between:.4f}")
    for n, m, s in per_leg:
        print(f"    {n:<24} mean {m:+.4f}  std {s:.4f}")

for p in ["joint1_roll", "joint2_pitch", "joint3_pitch",
          "prismatic1", "prismatic2"]:
    report(p)
