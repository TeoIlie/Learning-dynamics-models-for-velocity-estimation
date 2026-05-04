"""Parse a ROS2 bag (zip or extracted dir) and plot:
  1. XY trajectory from OptiTrack
  2. Sideslip angle beta over time
  3. Steering command over time
  4. Motor (velocity-actuating) command over time

Usage:
    python code/utils/bag_viz.py path/to/bag.zip [--out plots.png]

The bag is the format used by this dataset (see dataset/data/*).
Custom message types (vesc_msgs, optitrack_interfaces_msgs) are registered
on-the-fly from dataset/msgs/.

Requires `rosbags` (`pip install rosbags`); does not need a ROS install.
"""

import argparse
import os
import sqlite3
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


REPO_ROOT = Path(__file__).resolve().parents[2]
MSG_ROOT = REPO_ROOT / "dataset" / "msgs"

TOPIC_POSE = "/optitrack/rigid_body_0"
TOPIC_STEER_CANDIDATES = ["/commands/servo/position", "/vesc/servo_position_command"]
TOPIC_VEL_CMD_CANDIDATES = [
    "/commands/motor/speed",
    "/commands/motor/current",
    "/commands/motor/brake",
]


def build_typestore():
    """Register the repo's custom .msg types into a fresh ROS2 typestore."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    add_types = {}
    for pkg_dir in MSG_ROOT.iterdir():
        if not pkg_dir.is_dir():
            continue
        pkg = pkg_dir.name
        for msg_file in (pkg_dir / "msg").glob("*.msg"):
            type_name = f"{pkg}/msg/{msg_file.stem}"
            add_types.update(get_types_from_msg(msg_file.read_text(), type_name))
    ts.register(add_types)
    return ts


def resolve_bag_dir(path: Path, stack):
    """Return a directory containing the bag. Extracts archives if needed."""
    if path.is_dir():
        return path
    name = path.name.lower()
    is_tar = name.endswith((".tar.xz", ".tar.gz", ".tar.bz2", ".tar"))
    if path.suffix == ".zip" or is_tar:
        tmp = Path(tempfile.mkdtemp(prefix="bagviz_"))
        stack.append(tmp)
        if is_tar:
            with tarfile.open(path) as tf:
                tf.extractall(tmp)
        else:
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmp)
        # archive may wrap a single top-level dir; otherwise look for the dir
        # that actually contains the bag (metadata.yaml or *.db3)
        for root, _, files in os.walk(tmp):
            if "metadata.yaml" in files or any(f.endswith(".db3") for f in files):
                return Path(root)
        entries = [p for p in tmp.iterdir() if p.is_dir()]
        return entries[0] if len(entries) == 1 else tmp
    raise ValueError(f"Unsupported bag path: {path}")


def quat_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _iter_messages(bag_dir: Path):
    """Yield (topic, msgtype, t_ns, raw_cdr) from a bag dir.

    Uses rosbags Reader if metadata.yaml is present, else falls back to
    reading the sqlite3 .db3 directly (some zips ship without metadata).
    """
    if (bag_dir / "metadata.yaml").exists():
        with Reader(bag_dir) as reader:
            for conn, t_ns, raw in reader.messages():
                yield conn.topic, conn.msgtype, t_ns, raw
        return

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        raise FileNotFoundError(f"no metadata.yaml or .db3 in {bag_dir}")
    for db3 in db3_files:
        con = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
        try:
            topics = {
                tid: (name, mtype)
                for tid, name, mtype in con.execute(
                    "SELECT id, name, type FROM topics"
                )
            }
            cur = con.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
            )
            for topic_id, t_ns, raw in cur:
                name, mtype = topics[topic_id]
                yield name, mtype, t_ns, raw
        finally:
            con.close()


def _list_topics(bag_dir: Path):
    """Return {topic: (msgtype, count)}."""
    if (bag_dir / "metadata.yaml").exists():
        out = {}
        with Reader(bag_dir) as reader:
            for c in reader.connections:
                out[c.topic] = (c.msgtype, c.msgcount)
        return out
    out = {}
    for db3 in sorted(bag_dir.glob("*.db3")):
        con = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
        try:
            for tid, name, mtype in con.execute("SELECT id, name, type FROM topics"):
                cnt = con.execute(
                    "SELECT COUNT(*) FROM messages WHERE topic_id=?", (tid,)
                ).fetchone()[0]
                prev_t, prev_c = out.get(name, (mtype, 0))
                out[name] = (mtype, prev_c + cnt)
        finally:
            con.close()
    return out


def read_bag(bag_dir: Path, ts):
    pose_t, pose_x, pose_y = [], [], []
    pose_qx, pose_qy, pose_qz, pose_qw = [], [], [], []
    steer_t, steer_v, steer_topic = [], [], None
    velcmd_t, velcmd_v, velcmd_topic = [], [], None

    topics = _list_topics(bag_dir)
    for cand in TOPIC_STEER_CANDIDATES:
        if cand in topics and topics[cand][1] > 0:
            steer_topic = cand; break
    for cand in TOPIC_VEL_CMD_CANDIDATES:
        if cand in topics and topics[cand][1] > 0:
            velcmd_topic = cand; break

    wanted = {TOPIC_POSE, steer_topic, velcmd_topic} - {None}

    for topic, mtype, t_ns, raw in _iter_messages(bag_dir):
        if topic not in wanted:
            continue
        msg = ts.deserialize_cdr(raw, mtype)
        t = t_ns * 1e-9
        if topic == TOPIC_POSE:
            p, q = msg.pose.position, msg.pose.orientation
            pose_t.append(t); pose_x.append(p.x); pose_y.append(p.y)
            pose_qx.append(q.x); pose_qy.append(q.y)
            pose_qz.append(q.z); pose_qw.append(q.w)
        elif topic == steer_topic:
            steer_t.append(t); steer_v.append(float(msg.data))
        elif topic == velcmd_topic:
            velcmd_t.append(t); velcmd_v.append(float(msg.data))

    pose = {
        "t": np.asarray(pose_t),
        "x": np.asarray(pose_x), "y": np.asarray(pose_y),
        "qx": np.asarray(pose_qx), "qy": np.asarray(pose_qy),
        "qz": np.asarray(pose_qz), "qw": np.asarray(pose_qw),
    }
    steer = {"t": np.asarray(steer_t), "v": np.asarray(steer_v), "topic": steer_topic}
    velcmd = {"t": np.asarray(velcmd_t), "v": np.asarray(velcmd_v), "topic": velcmd_topic}
    return pose, steer, velcmd


def compute_body_velocity(pose, min_speed=0.3):
    """Return (t, v_x, v_y, beta) in body frame. beta is NaN below min_speed."""
    t = pose["t"]
    if t.size < 3:
        z = np.array([])
        return z, z, z, z
    t0 = t - t[0]
    yaw = quat_to_yaw(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    vx_w = np.gradient(pose["x"], t0)
    vy_w = np.gradient(pose["y"], t0)
    c, s = np.cos(yaw), np.sin(yaw)
    v_x =  c * vx_w + s * vy_w
    v_y = -s * vx_w + c * vy_w
    speed = np.hypot(v_x, v_y)
    beta = np.arctan2(v_y, v_x)
    beta[speed < min_speed] = np.nan
    return t0, v_x, v_y, beta


def plot_all(pose, steer, velcmd, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))

    ax = axes[0, 0]
    ax.plot(pose["x"], pose["y"], lw=0.8)
    if pose["x"].size:
        ax.scatter(pose["x"][0], pose["y"][0], c="g", s=30, label="start", zorder=5)
        ax.scatter(pose["x"][-1], pose["y"][-1], c="r", s=30, label="end", zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("XY trajectory (OptiTrack)")
    ax.grid(True); ax.legend()

    ax = axes[0, 1]
    tb, v_x_b, v_y_b, beta = compute_body_velocity(pose)
    if tb.size:
        ax.plot(tb, np.rad2deg(beta), lw=0.6)
    ax.set_xlabel("t [s]"); ax.set_ylabel(r"$\beta$ [deg]")
    ax.set_title("Sideslip angle  (NaN where |v| < 0.3 m/s)")
    ax.grid(True)

    ax = axes[1, 0]
    if steer["t"].size:
        t0 = steer["t"] - steer["t"][0]
        ax.plot(t0, steer["v"], lw=0.6)
        ax.set_title(f"Steering command  ({steer['topic']})")
    else:
        ax.set_title("Steering command  (no data)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("servo position [-]")
    ax.grid(True)

    ax = axes[1, 1]
    if velcmd["t"].size:
        t0 = velcmd["t"] - velcmd["t"][0]
        ax.plot(t0, velcmd["v"], lw=0.6)
        ax.set_title(f"Velocity (motor) command  ({velcmd['topic']})")
    else:
        ax.set_title("Velocity command  (no data)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("command [-]")
    ax.grid(True)

    ax = axes[2, 0]
    beta_valid = beta[np.isfinite(beta)] if tb.size else np.array([])
    if beta_valid.size:
        ax.hist(np.rad2deg(beta_valid), bins=80, color="C0", edgecolor="k", lw=0.3)
    ax.set_xlabel(r"$\beta$ [deg]"); ax.set_ylabel("count")
    ax.set_title("Sideslip angle histogram")
    ax.grid(True)

    ax = axes[2, 1]
    if tb.size:
        ax.hist(v_x_b, bins=80, alpha=0.7, label=r"$v_x$ (long.)", edgecolor="k", lw=0.3)
        ax.hist(v_y_b, bins=80, alpha=0.7, label=r"$v_y$ (lat.)",  edgecolor="k", lw=0.3)
        ax.legend()
    ax.set_xlabel("velocity [m/s]"); ax.set_ylabel("count")
    ax.set_title("Body-frame velocity histogram")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bag", type=Path, help="path to .zip or extracted bag dir")
    p.add_argument("--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    cleanup = []
    try:
        bag_dir = resolve_bag_dir(args.bag, cleanup)
        ts = build_typestore()
        pose, steer, velcmd = read_bag(bag_dir, ts)
        out = args.out or args.bag.with_suffix(".png")
        plot_all(pose, steer, velcmd, out)
        if args.show:
            plt.show()
    finally:
        for d in cleanup:
            import shutil; shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
