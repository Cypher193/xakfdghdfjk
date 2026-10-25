"""Render scene specs to mp4 with Blender (pip `bpy`, Python 3.11).

Positions are keyframed from the analytic tracks in each spec, so ground truth is exact.
Lighting/background: Poly Haven HDRIs; floor and optional back wall: Poly Haven textures (CC0).

    python -m pilot.render data/scenes/kmain --out data/renders/kmain --gpu                          # full clips, 24 fps
    python -m pilot.render data/scenes/kgrav --out data/renders/kgrav --gpu --fps 48 --seconds 1.125 --suffix _input   # inputs
    python -m pilot.render data/scenes/kcont --out data/renders/kcont --gpu --fps 16 --seconds 5.8125  # Cosmos C2-g
"""
import argparse
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import bpy
import numpy as np

from pilot.scenes import object_track

CACHE = Path("assets_cache")
API = "https://api.polyhaven.com"
RGB = dict(red=(0.8, 0.05, 0.04), blue=(0.05, 0.15, 0.8), yellow=(0.9, 0.75, 0.05), green=(0.08, 0.6, 0.12),
           orange=(0.95, 0.35, 0.03), white=(0.85, 0.85, 0.85), purple=(0.45, 0.1, 0.65))


def _get(url, dest=None):
    req = urllib.request.Request(url, headers={"User-Agent": "cvpr-vail-pilot"})
    for attempt in range(5):        # Poly Haven from some hosts: "SSL: UNEXPECTED_EOF_WHILE_READING" (phase 2 canary)
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(10 * 2 ** attempt)
    if dest:                      # write then rename: parallel render jobs never read a half-written file
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return data


TINTED = {"red", "blue", "teal", "green", "purple", "pink", "neon", "orange", "colorful", "colourful", "night"}


def asset_ids(kind, category):
    f = CACHE / f"{kind}_{category}.json"
    if not f.exists():
        _get(f"{API}/assets?type={kind}&categories={category}", f)
    assets = json.loads(f.read_text(encoding="utf-8"))
    if kind == "hdris":   # neutral daylight only: coloured or night lighting tints every object
        assets = {k: v for k, v in assets.items()
                  if "natural light" in v.get("categories", []) and not TINTED & set(v.get("tags", []) + v.get("categories", []))}
    return sorted(assets)


def asset_file(asset_id, key, res="1k", ext="jpg"):
    """Download one map of a Poly Haven asset; key = 'hdri' or a texture map name such as 'Diffuse'."""
    files = CACHE / f"{asset_id}.files.json"
    if not files.exists():
        _get(f"{API}/files/{asset_id}", files)
    info = json.loads(files.read_text(encoding="utf-8"))[key][res][ext]
    dest = CACHE / Path(info["url"]).name
    if not dest.exists():
        _get(info["url"], dest)
    return str(dest.resolve())


def material(name, color=None, image=None, uv_scale=1.0, roughness=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    if color:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
    if image:
        nt = m.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(image)
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (uv_scale, uv_scale, uv_scale)
        coord = nt.nodes.new("ShaderNodeTexCoord")
        nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m


def setup(scene_spec, gpu, samples, fps, seconds):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # linear between keyframes, so motion blur samples sit on the true path (no Bezier overshoot)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    sc = bpy.context.scene
    cam_spec = scene_spec["camera"]
    n = int(round(seconds * fps))
    t = np.arange(n) / fps
    sc.render.engine = "CYCLES"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = cam_spec["width"], cam_spec["height"]
    sc.render.resolution_percentage = 100
    sc.render.fps = fps
    sc.frame_start, sc.frame_end = 1, n
    sc.render.use_persistent_data = True                       # reuse BVH and textures across frames
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    if gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                if any(d.type == backend for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = d.type == backend
                    sc.cycles.device = "GPU"
                    if hasattr(sc.cycles, "denoising_use_gpu"):     # OIDN on the GPU (CPU was the bottleneck)
                        sc.cycles.denoising_use_gpu = True
                    break
            except TypeError:
                continue

    # camera: at (0, -D, 0) looking along +y; horizontal sensor so lens matches the spec's fx
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = 18.0 / math.tan(math.radians(cam_spec["hfov_deg"] / 2))
    cam.location = (0.0, -cam_spec["distance"], cam_spec.get("cam_z", 0.0))
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
    sc.collection.objects.link(cam)
    sc.camera = cam

    rnd = random.Random(scene_spec["env_seed"])
    world = bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    env = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    hdri = rnd.choice(asset_ids("hdris", "indoor") + asset_ids("hdris", "outdoor"))
    env.image = bpy.data.images.load(asset_file(hdri, "hdri", ext="hdr"))
    world.node_tree.links.new(env.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])

    floor_tex = rnd.choice(asset_ids("textures", "floor"))
    bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, cam_spec["floor_z"]))   # horizon reaches frame centre
    bpy.context.object.data.materials.append(
        material("floor", image=asset_file(floor_tex, "Diffuse"), uv_scale=200, roughness=0.6))
    used = dict(hdri=hdri, floor=floor_tex)
    if rnd.random() < 0.5:                                   # back wall behind the motion plane
        wall_tex = rnd.choice(asset_ids("textures", "wall"))
        bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 4.0, cam_spec["floor_z"] + 20),
                                         rotation=(math.pi / 2, 0, 0))
        bpy.context.object.data.materials.append(
            material("wall", image=asset_file(wall_tex, "Diffuse"), uv_scale=10, roughness=0.8))
        used["wall"] = wall_tex

    for i, o in enumerate(scene_spec["objects"]):
        s = o["size_m"]
        if o["shape"] == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=s / 2, segments=48, ring_count=24)
            bpy.ops.object.shade_smooth()
        elif o["shape"] == "cube":
            bpy.ops.mesh.primitive_cube_add(size=s)
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=s / 2, depth=s, vertices=48)
        ob = bpy.context.object
        ob.name = f"obj{i}"
        ob.data.materials.append(material(f"m{i}", color=RGB[o["color"]], roughness=0.35))
        xs, zs = object_track(scene_spec, o, t)               # exact positions at this frame rate
        for f, (x, z) in enumerate(zip(xs, zs), start=1):
            ob.location = (x, 0.0, z)
            ob.keyframe_insert("location", frame=f)
    return used


def render(spec_path, out_dir, gpu=False, samples=32, fps=24, seconds=6.0, suffix=""):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    out = Path(out_dir) / f"{spec['id']}{suffix}.mp4"
    if out.exists():
        return
    used = setup(spec, gpu, samples, fps, seconds)
    sc = bpy.context.scene
    if hasattr(sc.render.image_settings, "media_type"):     # Blender 5: choose video output first
        sc.render.image_settings.media_type = "VIDEO"
    sc.render.image_settings.file_format = "FFMPEG"
    sc.render.ffmpeg.format = "MPEG4"
    sc.render.ffmpeg.codec = "H264"
    sc.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
    partial = out.with_suffix(".partial.mp4")              # finished clips only ever appear under the final name
    sc.render.filepath = str(partial.resolve())
    bpy.ops.render.render(animation=True)
    os.replace(partial, out)
    (Path(out_dir) / f"{spec['id']}{suffix}.assets.json").write_text(json.dumps(used | dict(fps=fps, seconds=seconds)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", help="folder of scene JSONs, or one JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--suffix", default="", help="e.g. _input for the 48 fps input clips")
    ap.add_argument("--frames", type=int, default=0, help="debug: render only the first N frames")
    a = ap.parse_args()
    p = Path(a.scenes)
    specs = sorted(p.glob("*.json")) if p.is_dir() else [p]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for s in specs:
        if a.frames:
            spec = json.loads(s.read_text(encoding="utf-8"))
            setup(spec, a.gpu, a.samples, a.fps, a.seconds)
            sc = bpy.context.scene
            sc.frame_end = a.frames
            sc.render.image_settings.file_format = "PNG"
            sc.render.filepath = str((Path(a.out) / f"{spec['id']}_").resolve())
            bpy.ops.render.render(animation=True)
        else:
            render(s, a.out, a.gpu, a.samples, a.fps, a.seconds, a.suffix)
        print("rendered", s.name, flush=True)
