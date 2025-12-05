import os
import time
import shutil
import tempfile
import logging
from collections import deque, defaultdict
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

try:
    import imageio
except ImportError:
    imageio = None

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


from engine.atoms import Atom
from engine.elements_data import ELEMENT_DATA
from engine.simulation_manager import SimulationManager
from visual.colors import get_element_color

# ────────────────────────────
# Utility: ensure_dir
# ────────────────────────────

def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

# ────────────────────────────
# Frame export
# ────────────────────────────

def save_frame_png(sim: SimulationManager, filename: Optional[str] = None):
    """
    Save the current simulation frame as a PNG.
    """
    if filename is None:
        filename = os.path.join("outputs", f"frame_{int(time.time())}.png")
    if sim.fig is None:
        from visual.renderer import render_simulation_frame
        render_simulation_frame(sim)
    sim.fig.savefig(filename, dpi=200, bbox_inches='tight')
    logger.info(f"Saved frame to {filename}")
    return filename


def _render_frames_to_images(sim: SimulationManager, n_frames: int, folder: str):
    """
    Render n_frames from sim history and save PNG frames into folder.
    """
    os.makedirs(folder, exist_ok=True)
    if not sim.atoms or not hasattr(sim.atoms[0], 'history'):
        raise RuntimeError("No atom history available for rendering frames.")

    total = len(sim.atoms[0].history)
    n = min(n_frames, total)
    orig_positions = [a.pos.copy() for a in sim.atoms]

    from visual.renderer import render_simulation_frame

    for i in range(n):
        for a in sim.atoms:
            a.pos = np.array(a.history[i]['pos'])
        render_simulation_frame(sim)
        out_path = os.path.join(folder, f"frame_{i:04d}.png")
        sim.fig.savefig(out_path, dpi=150, bbox_inches='tight')

    for a, p in zip(sim.atoms, orig_positions):
        a.pos = p
    return n


# -----------------------------
# GIF / MP4 Export
# -----------------------------

def export_simulation_to_gif(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """
    Export the simulation to an animated GIF.
    """
    if not sim.atoms or not hasattr(sim.atoms[0], 'history'):
        raise RuntimeError("No atoms present in simulation for export.")

    n_frames = n_frames or len(sim.atoms[0].history)
    tmpdir = tempfile.mkdtemp(prefix='sim_frames_')
    try:
        n = _render_frames_to_images(sim, n_frames, tmpdir)
        if imageio is not None:
            images = [imageio.imread(os.path.join(tmpdir, f"frame_{i:04d}.png")) for i in range(n)]
            imageio.mimsave(filename, images, fps=fps)
            logger.info(f"Saved GIF to {filename}")
        else:
            raise RuntimeError("imageio not installed, cannot export GIF")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_simulation_to_mp4(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """
    Export the simulation to MP4 using imageio-ffmpeg or ffmpeg subprocess.
    """
    if not sim.atoms or not hasattr(sim.atoms[0], 'history'):
        raise RuntimeError("No atoms present in simulation for export.")

    n_frames = n_frames or len(sim.atoms[0].history)
    tmpdir = tempfile.mkdtemp(prefix='sim_frames_')
    try:
        n = _render_frames_to_images(sim, n_frames, tmpdir)
        if imageio is not None:
            writer = imageio.get_writer(filename, fps=fps, codec='libx264')
            for i in range(n):
                img = imageio.imread(os.path.join(tmpdir, f"frame_{i:04d}.png"))
                writer.append_data(img)
            writer.close()
            logger.info(f"Saved MP4 to {filename}")
        else:
            # fallback to ffmpeg subprocess
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(tmpdir, 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', filename
            ]
            subprocess.check_call(cmd)
            logger.info(f"Saved MP4 to {filename} via ffmpeg")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------
# Plotly HTML Export
# -----------------------------

def export_simulation_to_plotly_html(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """
    Export the simulation history as an interactive Plotly HTML animation.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is not installed.")

    n_frames = n_frames or (len(sim.atoms[0].history) if sim.atoms else sim.frame)
    frames = []
    uid_to_index = {a.uid: i for i, a in enumerate(sim.atoms)}

    for f in range(n_frames):
        xs = [a.history[f]['pos'][0] for a in sim.atoms]
        ys = [a.history[f]['pos'][1] for a in sim.atoms]
        colors = [get_element_color(a.symbol) for a in sim.atoms]
        sizes = [max(6, int(a.radius*2000) if hasattr(a, 'radius') else 10) for a in sim.atoms]

        atom_trace = go.Scatter(
            x=xs, y=ys, mode='markers+text', text=[a.symbol for a in sim.atoms],
            marker=dict(size=sizes, color=colors, line=dict(width=1, color='black')),
            textposition='bottom center', hoverinfo='text'
        )

        bond_x = []
        bond_y = []
        for b in sim.bonds:
            bond_x.extend([b.atom1.pos[0], b.atom2.pos[0], None])
            bond_y.extend([b.atom1.pos[1], b.atom2.pos[1], None])
        bond_trace = go.Scatter(x=bond_x, y=bond_y, mode='lines', line=dict(color='#888888', width=2), hoverinfo='none')

        frames.append(go.Frame(data=[atom_trace, bond_trace], name=str(f), traces=[0,1]))

    fig = go.Figure(data=[frames[0].data[0], frames[0].data[1]], frames=frames)
    fig.update_layout(
        title=f"Simulation Export ({n_frames} frames)",
        showlegend=False,
        xaxis=dict(range=[0,1], visible=False),
        yaxis=dict(range=[0,1], visible=False),
        width=800, height=800,
        updatemenus=[dict(type='buttons', showactive=False, y=1, x=1.12,
                          buttons=[
                              dict(label='Play', method='animate', args=[None, {'frame': {'duration': int(1000/fps), 'redraw': True}, 'fromcurrent': True}]),
                              dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0}, 'mode':'immediate', 'transition': {'duration':0}}])
                          ])]
    )

    sliders = [dict(steps=[dict(method='animate', args=[[fr.name], {'mode':'immediate','frame':{'duration': int(1000/fps),'redraw':True},'transition':{'duration':0}}], label=str(i)) for i, fr in enumerate(frames)], active=0, x=0, y=0, len=1.0)]
    fig.update_layout(sliders=sliders)
    fig.write_html(filename, include_plotlyjs='cdn')
    logger.info(f"Exported interactive HTML to {filename}")


# -----------------------------
# GitHub Pages publishing
# -----------------------------

def publish_directory_to_github(dir_path: str, repo_url: str, branch: str = 'gh-pages', token: Optional[str] = None, commit_message: str = 'Publish simulation'):
    """
    Publish a directory to GitHub Pages. Embeds token if provided.
    """
    import subprocess
    tmpdir = tempfile.mkdtemp(prefix='publish_repo_')
    try:
        shutil.copytree(dir_path, os.path.join(tmpdir, os.path.basename(dir_path)), dirs_exist_ok=True)
        subprocess.check_call(['git', 'init'], cwd=tmpdir)
        subprocess.check_call(['git', 'add', '.'], cwd=tmpdir)
        subprocess.check_call(['git', 'commit', '-m', commit_message], cwd=tmpdir)
        subprocess.check_call(['git', 'branch', '-M', 'master'], cwd=tmpdir)
        remote = repo_url
        if token and remote.startswith('https://'):
            remote = remote.replace('https://', f'https://{token}@')
        subprocess.check_call(['git', 'remote', 'add', 'origin', remote], cwd=tmpdir)
        subprocess.check_call(['git', 'push', '-f', 'origin', f'master:{branch}'], cwd=tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------
# Small wrappers / utilities
# -----------------------------
def now_str() -> str:
    """Return a timestamp string used for filenames."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def export_product_timeline(sim: SimulationManager, component_key: str, out_path: str) -> None:
    """
    Wrapper that exports the timeline of events for a detected product component.
    This calls into `engine.products.export_product_timeline` using the simulation's
    recorded events (if present) and the atoms for the component.
    """
    try:
        # Prefer simulation-managed KB events if available
        events = []
        if hasattr(sim, 'kb_events') and isinstance(getattr(sim, 'kb_events'), list):
            events = sim.kb_events
        elif hasattr(sim, 'events') and isinstance(getattr(sim, 'events'), list):
            events = sim.events

        detected = sim.detect_products()
        comp = detected.get(component_key)
        if not comp:
            raise ValueError(f"Component '{component_key}' not found in simulation products")
        comp_atoms, _ = comp

        # Delegate to engine.products
        from engine.products import export_product_timeline as _engine_export_product_timeline
        _engine_export_product_timeline(events, comp_atoms, out_path)
    except Exception:
        logger.exception("Failed to export product timeline to %s", out_path)