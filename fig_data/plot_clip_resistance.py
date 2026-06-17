"""
Plot: EMA-GRPO resistance to max_token truncation
Data: grpo_raw.csv, ema_grpo_raw.csv
Output: fig_clip_resistance.png (same directory)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams

# ── Style ─────────────────────────────────────────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size': 13,
    'axes.labelsize': 16,
    'axes.titlesize': 13,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#e0e0e0',
    'grid.linewidth': 0.8,
    'axes.axisbelow': True,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

C_GRPO = '#E41A1C'
C_EMA  = '#377EB8'

# ── Load data ─────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

grpo = pd.read_csv(os.path.join(HERE, 'grpo_raw.csv'))
ema  = pd.read_csv(os.path.join(HERE, 'ema_grpo_raw.csv'))

# ── Moving average (valid convolution — no edge artifacts) ────────────────────
W = 30

def smooth(steps, values, w=W):
    half = w // 2
    kernel = np.ones(w) / w
    s = np.convolve(values, kernel, mode='valid')
    t = steps[half: half + len(s)]
    return t, s

g_s,  g_score_ma  = smooth(grpo['step'].values, grpo['reward_score'].values)
g_l,  g_len_ma    = smooth(grpo['step'].values, grpo['response_length_tokens'].values)
g_c,  g_clip_ma   = smooth(grpo['step'].values, grpo['clip_ratio'].values)

e_s,  e_score_ma  = smooth(ema['step'].values, ema['reward_score'].values)
e_l,  e_len_ma    = smooth(ema['step'].values, ema['response_length_tokens'].values)
e_c,  e_clip_ma   = smooth(ema['step'].values, ema['clip_ratio'].values)

# ── Figure ────────────────────────────────────────────────────────────────────
COLLAPSE_START, COLLAPSE_END = 900, 1020
RAW_ALPHA = 0.13
MA_LW = 2.2

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
fig.subplots_adjust(hspace=0.35)

def shade(ax):
    ax.axvspan(COLLAPSE_START, COLLAPSE_END, color=C_GRPO, alpha=0.08, zorder=0)
    ax.axvline(1000, color='#888888', lw=1.0, ls='--', zorder=1)

# ── (a) Reward Score ──────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(grpo['step'], grpo['reward_score'],    color=C_GRPO, alpha=RAW_ALPHA, lw=0.8)
ax.plot(ema['step'],  ema['reward_score'],     color=C_EMA,  alpha=RAW_ALPHA, lw=0.8)
ax.plot(g_s, g_score_ma, color=C_GRPO, lw=MA_LW, label='GRPO (baseline)')
ax.plot(e_s, e_score_ma, color=C_EMA,  lw=MA_LW, label='EMA-GRPO')
shade(ax)
ax.set_ylabel('Reward Score')
ax.set_title('(a) Reward Score', loc='left', fontsize=13, fontweight='normal', pad=4)
ax.legend(loc='upper right')
ax.set_ylim(0.3, 1.05)
ax.annotate('Score drops\n(step ~950)',
            xy=(960, 0.54), xytext=(800, 0.38),
            color=C_GRPO, fontsize=10,
            arrowprops=dict(arrowstyle='->', color=C_GRPO, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GRPO, lw=0.8, alpha=0.9))

# ── (b) Response Length ───────────────────────────────────────────────────────
ax = axes[1]
ax.plot(grpo['step'], grpo['response_length_tokens'], color=C_GRPO, alpha=RAW_ALPHA, lw=0.8)
ax.plot(ema['step'],  ema['response_length_tokens'],  color=C_EMA,  alpha=RAW_ALPHA, lw=0.8)
ax.plot(g_l, g_len_ma, color=C_GRPO, lw=MA_LW)
ax.plot(e_l, e_len_ma, color=C_EMA,  lw=MA_LW)
shade(ax)
ax.axhline(8192, color='#aaaaaa', lw=1.0, ls=':', zorder=0)
ax.text(50, 8350, 'max_token = 8192', color='#999999', fontsize=9)
ax.set_ylabel('Response Length (tokens)')
ax.set_title('(b) Average Response Length', loc='left', fontsize=13, fontweight='normal', pad=4)
ax.set_ylim(-200, 9500)
ax.annotate('GRPO collapses\nto ~1358 tokens',
            xy=(1200, g_len_ma[-1]), xytext=(1030, 500),
            color=C_GRPO, fontsize=10,
            arrowprops=dict(arrowstyle='->', color=C_GRPO, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GRPO, lw=0.8, alpha=0.9))
ax.annotate('EMA-GRPO stable\n~3200-3600 tokens',
            xy=(1000, e_len_ma[np.searchsorted(e_l, 1000)]),
            xytext=(850, 5800),
            color=C_EMA, fontsize=10,
            arrowprops=dict(arrowstyle='->', color=C_EMA, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_EMA, lw=0.8, alpha=0.9))

# ── (c) Clip Ratio ────────────────────────────────────────────────────────────
ax = axes[2]
ax.plot(grpo['step'], grpo['clip_ratio'], color=C_GRPO, alpha=RAW_ALPHA, lw=0.8)
ax.plot(ema['step'],  ema['clip_ratio'],  color=C_EMA,  alpha=RAW_ALPHA, lw=0.8)
ax.plot(g_c, g_clip_ma, color=C_GRPO, lw=MA_LW)
ax.plot(e_c, e_clip_ma, color=C_EMA,  lw=MA_LW)
shade(ax)
ax.set_ylabel('Truncation Ratio')
ax.set_xlabel('Training Step')
ax.set_title('(c) Truncation Ratio (Clip Ratio)', loc='left', fontsize=13, fontweight='normal', pad=4)
ax.set_ylim(-0.01, 0.45)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax.annotate('(1) Spike to 0.21\n-> large neg. advantage\n-> penalizes long output',
            xy=(930, 0.26), xytext=(630, 0.34),
            color=C_GRPO, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_GRPO, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GRPO, lw=0.8, alpha=0.9))
ax.annotate('(2) Model shortens output\nclip ratio -> ~0',
            xy=(1060, 0.015), xytext=(1060, 0.18),
            color=C_GRPO, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_GRPO, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_GRPO, lw=0.8, alpha=0.9))
ax.annotate('EMA-GRPO stable\n0.10-0.13',
            xy=(600, e_clip_ma[np.searchsorted(e_c, 600)]),
            xytext=(200, 0.28),
            color=C_EMA, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=C_EMA, lw=1.2),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=C_EMA, lw=0.8, alpha=0.9))

axes[-1].set_xlim(1, 1290)
axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(200))

# ── Save ──────────────────────────────────────────────────────────────────────
out = os.path.join(HERE, 'fig_clip_resistance.png')
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
