"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import os
import json
import math
import zlib
import sqlite3
from typing import Any
from datetime import timedelta
from unaiverse.utils.logger import log
from sortedcontainers import SortedDict
from typing_extensions import deprecated

# A fixed palette for consistent coloring
THEMES = {
    "dark": {
        "bg_paper": "rgba(0,0,0,0)",
        "bg_plot": "rgba(0,0,0,0)",
        "text_main": "#C8CDD3",
        "text_light": "#677385",
        "grid": "#222A36",
        "edge": "#495464",
        "node_border": "#F5F6F8",
        "main": "#1A5CFF",
        "main_light": "#4D7FFF",
        "table": {
            "header_bg": "#16171C",
            "header_txt": "#F5F6F8",
            "cell_bg": "#0E0F14",
            "cell_txt": "#C8CDD3",
            "line": "#222A36",
        },
        "peers": [
            "#1A5CFF", "#FF3B30", "#00D4AA", "#FFB347", "#4D7FFF",
            "#00B391", "#FF6692", "#33EABD", "#FFD080", "#6B9BFF",
        ],
    },
    "light": {
        "bg_paper": "rgba(255,255,255,0)",
        "bg_plot": "rgba(255,255,255,0)",
        "text_main": "#0A1628",
        "text_light": "#677385",
        "grid": "#DBE1EB",
        "edge": "#909CB0",
        "node_border": "#0A1628",
        "main": "#1A5CFF",
        "main_light": "#4D7FFF",
        "table": {
            "header_bg": "#F6F8FA",
            "header_txt": "#0A1628",
            "cell_bg": "#FFFFFF",
            "cell_txt": "#0A1628",
            "line": "#C0C8D6",
        },
        "peers": [
            "#1A5CFF", "#FF3B30", "#00D4AA", "#FFB347", "#4D7FFF",
            "#00B391", "#FF6692", "#33EABD", "#FFD080", "#6B9BFF",
        ],
    },
}

# Backward-compatible flat alias
THEME = THEMES["dark"]


class UIPlot:
    """
    A Python abstraction for a UI Panel (specifically a Plotly chart).
    Allows users to build plots using Python methods instead of dicts/JSON.
    """

    def __init__(self, title: str = '', height: int = 400):
        """Initializes a UIPlot with an empty data list and a pre-configured dark-theme layout.

        Args:
            title: Title text shown at the top of the plot.
            height: Height of the plot in pixels.
        """
        self._data: list[dict[str, Any]] = []

        # Define the standard axis style for a "boxed" look
        axis_style = {
            'gridcolor': THEME['grid'],
            'gridwidth': 1,
            'griddash': 'dot',
            'color': THEME['text_light'],
            'showline': True,  # Draw the axis line
            'mirror': True,  # Mirror it on top/right (creates the box)
            'linewidth': 2,  # Width of the box border
            'linecolor': THEME['grid'],  # Color of the box border
            'zeroline': False,  # Prevents double-thick borderlines at 0
            'layer': 'below traces'  # Key fix: puts grid BEHIND the box border
        }

        self._layout: dict[str, Any] = {
            'title': title,
            'height': height,
            'xaxis': {**axis_style, 'title': 'Time'},
            'yaxis': {**axis_style, 'title': 'Value'},
            'margin': {'l': 50, 'r': 50, 'b': 50, 't': 50},
            # Default dark theme friendly styling
            'paper_bgcolor': THEME['bg_paper'],
            'plot_bgcolor': THEME['bg_plot'],
            'font': {'color': THEME['text_main']}
        }

    def add_line(self, x: list[Any], y: list[Any], name: str, color: str | None = None,
                 legend_group: str | None = None, show_legend: bool = True):
        """Adds a standard time-series line.

        Args:
            x: Sequence of x-axis values (e.g., timestamps).
            y: Sequence of y-axis values.
            name: Label shown in the legend.
            color: CSS color string for the line. Defaults to the primary theme color.
            legend_group: Plotly legend group name for toggling related traces together.
            show_legend: Whether to display this trace in the legend.
        """
        if color is None:
            color = THEME['main']
        trace = {
            'x': x, 'y': y,
            'name': name,
            'type': 'scatter',
            'mode': 'lines+markers',
            'line': {'color': color},
            "legendgroup": legend_group,
            "showlegend": show_legend
        }
        self._data.append(trace)

    def add_area(self, x: list[Any], y: list[Any], name: str, color: str | None = None):
        """Adds a filled area chart.

        Args:
            x: Sequence of x-axis values.
            y: Sequence of y-axis values.
            name: Label shown in the legend.
            color: CSS color string for the fill/line. Defaults to the primary theme color.
        """
        if color is None:
            color = THEME['main']
        trace = {
            'x': x, 'y': y, 'name': name,
            'type': 'scatter', 'fill': 'tozeroy',
            'line': {'color': color}
        }
        self._data.append(trace)

    def add_indicator(self, value: Any, title: str):
        """Adds a big number indicator.

        Args:
            value: Numeric (or any) value to display prominently.
            title: Subtitle text rendered below the number.
        """
        self._data.append({
            'type': 'indicator',
            'mode': 'number',
            'value': value,
            'title': {'text': title}
        })
        self._layout['height'] = 300  # Indicators usually need less height

    def add_table(self, headers: list[str] | None, columns: list[list[Any]]):
        """Adds a data table.

        Args:
            headers: Column header labels, or ``None`` to hide the header row.
            columns: Column data as a list of lists, one inner list per column.
        """
        num_columns = len(columns) if columns else 0
        if headers:
            header_cfg = {
                'values': headers,
                'fill': {'color': THEME['table']['header_bg']},
                'font': {'color': THEME['table']['header_txt']},
                'line': {'color': THEME['table']['line']}
            }
        else:
            header_cfg = {
                'values': [''] * num_columns,
                'height': 0,  # Hide it
                'fill': {'color': 'rgba(0,0,0,0)'},  # Transparent just in case
                'line': {'width': 0}  # No border
            }

        trace = {
            'type': 'table',
            'header': header_cfg,
            'cells': {
                'values': columns,
                'fill': {'color': THEME['table']['cell_bg']},
                'font': {'color': THEME['table']['cell_txt']},
                'line': {'color': THEME['table']['line']}
            }
        }
        self._data.append(trace)

    def add_bar(self, xs: list[Any], ys: list[Any], names: list[str],
                colors: list[str] | str | None = None):
        """Adds a bar chart trace.

        Args:
            xs: Category labels for the x-axis.
            ys: Numeric values for bar heights.
            names: Text annotations placed on top of each bar.
            colors: Single CSS color string or a list of per-bar color strings.
                Defaults to the primary theme color.
        """
        if colors is None:
            colors = THEME['main']
        trace = {
            'type': 'bar',
            'x': xs,
            'y': ys,
            'marker': {'color': colors},
            'showlegend': False,
            'text': names,
            'textposition': 'auto'
        }
        self._data.append(trace)
        self._layout['yaxis'].update({'title': 'Value'})

    def add_trace(self, trace: dict[str, Any]):
        """Generic method to add any raw Plotly trace.

        Args:
            trace: A Plotly trace dictionary (e.g., ``{'type': 'scatter', ...}``).
        """
        self._data.append(trace)

    def set_y_range(self, min_val: float, max_val: float):
        """Force Y-axis limits.

        Args:
            min_val: Lower bound of the Y-axis.
            max_val: Upper bound of the Y-axis.
        """
        self._layout.setdefault('yaxis', {})['range'] = [min_val, max_val]

    def set_layout_opt(self, key: str, value: Any):
        """Generic setter for advanced layout options.

        Args:
            key: Top-level Plotly layout key (e.g., ``'xaxis'``, ``'legend'``).
            value: Value to assign. If the key already holds a dict and ``value`` is also
                a dict, the existing dict is updated (merged) rather than replaced.
        """
        if isinstance(value, dict) and key in self._layout:
            self._layout[key].update(value)
        else:
            self._layout[key] = value

    def set_legend(self, orientation: str = 'v', x: float = 1.0, y: float = 1.0,
                   xanchor: str = 'left', yanchor: str = 'top'):
        """Configures the legend position and orientation.

        Args:
            orientation: Legend orientation. ``'v'`` for vertical, ``'h'`` for horizontal.
            x: Horizontal position of the legend anchor in paper coordinates (0–1).
            y: Vertical position of the legend anchor in paper coordinates (0–1).
            xanchor: Horizontal alignment of the legend box relative to ``x``.
                One of ``'left'``, ``'center'``, ``'right'``.
            yanchor: Vertical alignment of the legend box relative to ``y``.
                One of ``'top'``, ``'middle'``, ``'bottom'``.
        """
        self._layout['showlegend'] = True
        self._layout['legend'] = {
            'orientation': orientation,
            'x': x,
            'y': y,
            'xanchor': xanchor,
            'yanchor': yanchor,
            'bgcolor': THEME['bg_paper'],
            'bordercolor': THEME['edge'],
            'borderwidth': 1
        }

    def to_json(self) -> str:
        """Serializes the panel to the format the Frontend expects.

        Returns:
            JSON string with ``'data'`` and ``'layout'`` keys ready for Plotly.
        """
        return json.dumps({'data': self._data, 'layout': self._layout})


class DefaultBaseDash:
    """
    A generic 2x2 Grid Dashboard for the base Stats class.
    Forces #111111 background to match the WStats styling.
    """

    def __init__(self, title: str = "Network Overview"):
        """Initializes the 2x2 grid dashboard with default dark-theme layout.

        Args:
            title: Main title displayed at the top of the dashboard.
        """
        self.traces = []
        self.layout = {
            "title": title,
            "height": 800,
            "template": "plotly_dark",
            "paper_bgcolor": THEME['bg_paper'],
            "grid": {"rows": 2, "columns": 2, "pattern": "independent"},

            # --- ROW 1 ---
            # Top Left (Graph)
            "xaxis1": {"domain": [0, 0.48]},
            "yaxis1": {"domain": [0.56, 1]},
            # "xaxis1": {"domain": [0, 0.48], "visible": False}, 
            # "yaxis1": {"domain": [0.58, 1], "visible": False},
            # Top Right (Timeseries)
            "xaxis2": {"domain": [0.52, 1]},
            "yaxis2": {"domain": [0.56, 1]},

            # --- ROW 2 ---
            # Bot Left (Bar)
            "xaxis3": {"domain": [0, 0.48]},
            "yaxis3": {"domain": [0, 0.44]},
            # Bot Right (Bar)
            "xaxis4": {"domain": [0.52, 1]},
            "yaxis4": {"domain": [0, 0.44]},

            "showlegend": True,
            "legend": {
                "orientation": "h",
                "y": 0.55,
                "x": 0.55,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": THEME['text_main']}
            },
            "margin": {"l": 50, "r": 50, "t": 80, "b": 50}
        }
        self._map = {
            "top_left": ("xaxis1", "yaxis1"),
            "top_right": ("xaxis2", "yaxis2"),
            "bot_left": ("xaxis3", "yaxis3"),
            "bot_right": ("xaxis4", "yaxis4")
        }

    def add_panel(self, ui_plot: UIPlot, position: str):
        """Merges a UIPlot into one of the four fixed grid positions.

        Args:
            ui_plot: The UIPlot instance whose traces and layout settings are merged in.
            position: Grid cell to target. One of ``'top_left'``, ``'top_right'``,
                ``'bot_left'``, ``'bot_right'``.
        """
        if position not in self._map:
            return

        xa, ya = self._map[position]
        self.layout: dict[str, dict[str, list[float]]]
        x_dom: list[float] = self.layout[xa]["domain"]
        y_dom: list[float] = self.layout[ya]["domain"]

        # Merge Traces
        for t in ui_plot._data:
            nt = t.copy()
            if nt.get("type") == "table":
                nt["domain"] = {"x": x_dom, "y": y_dom}
            else:
                # Cartesian plots use axis references
                nt["xaxis"] = xa.replace("xaxis", "x")
                nt["yaxis"] = ya.replace("yaxis", "y")
            self.traces.append(nt)

        # Merge Layout
        src_l = ui_plot._layout
        dest_x = self.layout.setdefault(xa, {})
        dest_y = self.layout.setdefault(ya, {})
        if "xaxis" in src_l:
            dest_x.update({k: v for k, v in src_l["xaxis"].items() if k != "domain"})
        if "yaxis" in src_l:
            dest_y.update({k: v for k, v in src_l["yaxis"].items() if k != "domain"})

        # Add Title via Annotation
        if src_l.get("title"):
            self.layout: dict[str, list]
            self.layout.setdefault("annotations", []).append({
                "text": f"<b>{src_l['title']}</b>",
                "x": (x_dom[0] + x_dom[1]) / 2,
                "y": y_dom[1] + 0.02,
                "xref": "paper", "yref": "paper",
                "showarrow": False, "xanchor": "center", "yanchor": "bottom",
                "font": {"size": 14, "color": THEME['text_main']}
            })

    def to_json(self) -> str:
        """Serializes the dashboard to the format the Frontend expects.

        Returns:
            JSON string with ``'data'`` and ``'layout'`` keys ready for Plotly.
        """
        return json.dumps({"data": self.traces, "layout": self.layout})


class Stats:
    """
    Encapsulates all logic for managing, storing, and persisting agent/world
    statistics. This class provides a clean API to the rest of the application
    and hides the implementation details of data structures and persistence.
    
    Design Principles:
      1.  Typed Schema: Class-level definitions (e.g., CORE_..._SCHEMA) are
            sets of tuples: {("stat_name", type), ...}
      2.  Unified API: All stat updates are handled by two methods:
            - store_static(stat_name, value, peer_id)
            - store_dynamic(stat_name, value, peer_id, timestamp)
      3.  Smart Branching: The store_... methods internally branch
            (if self.is_world: ...) to handle their specific roles:
              - Agent: Buffers for network, de-duplicates statics.
              - World: Updates hot cache, buffers for DB.
      4.  Persistence (SQLite):
            - A single SQLite DB file ('world_stats.db') stores all data.
            - Static Stats: Saved in a 'static_stats' table (key-value).
            - Dynamic Stats: Saved in a 'dynamic_stats' table (time-series).
      5.  Hot Cache (_stats):
            - Static Stats: Stored as their latest value.
            - Dynamic Stats: Stored in a sortedcontainers.SortedDict
              keyed by timestamp.
    """
    # These are all the keys in the local _stats dictionary collected by the world
    CORE_WORLD_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {
        'graph': (dict, {'nodes': {}, 'edges': {}})
    }
    CORE_WORLD_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {
        'world_masters': (int, 0),
        'world_agents': (int, 0),
        'human_agents': (int, 0),
        'artificial_agents': (int, 0)
    }

    # These are all the keys in the local _stats dictionary collected by the agent
    CORE_AGENT_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {
        'connected_peers': (list, []),
        'state': (str, None),
        'action': (str, None),
        'last_action': (str, None)
    }
    CORE_AGENT_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {}

    # Then we have the stats collected on behalf of other peers (by the agent or the world)
    CORE_OUTER_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CORE_OUTER_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {}

    # We also add class variables to extend these sets
    CUSTOM_WORLD_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CUSTOM_WORLD_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CUSTOM_AGENT_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CUSTOM_AGENT_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CUSTOM_OUTER_STATS_STATIC_SCHEMA: dict[str, tuple[type, Any]] = {}
    CUSTOM_OUTER_STATS_DYNAMIC_SCHEMA: dict[str, tuple[type, Any]] = {}

    # Whether to avoid storing repeated values in dynamic stats
    STORE_DYNAMIC_IF_CHANGED = False

    # Key for grouping stats in the _stats dictionary (both world and agent)
    GROUP_KEY = 'peer_stats'  # grouped stats are stored under this key

    # Deprecated
    DEBUG = False

    def __init__(self, is_world: bool,
                 db_path: str | None = None,  # only needed by the world
                 cache_window_hours: float = 2.0):  # only needed by the world
        """Initializes the Stats engine in either World or Agent mode.

        Args:
            is_world: When ``True``, initializes the full World-side engine including the
                SQLite database, hot cache, and persistence buffers. When ``False``,
                initializes a lightweight Agent-side buffer.
            db_path: Filesystem path for the SQLite database file. Required when
                ``is_world`` is ``True``; ignored otherwise.
            cache_window_hours: Duration (in hours) of the in-memory rolling window kept
                by the World hot cache. Older dynamic data is evicted from RAM while
                remaining on disk.
        """
        self.is_world: bool = is_world
        self.max_seen_timestamp: int = 0

        # --- Integrate custom statistics ---
        self.WORLD_STATS_STATIC_SCHEMA = self.CORE_WORLD_STATS_STATIC_SCHEMA | self.CUSTOM_WORLD_STATS_STATIC_SCHEMA
        self.WORLD_STATS_DYNAMIC_SCHEMA = self.CORE_WORLD_STATS_DYNAMIC_SCHEMA | self.CUSTOM_WORLD_STATS_DYNAMIC_SCHEMA
        self.AGENT_STATS_STATIC_SCHEMA = self.CORE_AGENT_STATS_STATIC_SCHEMA | self.CUSTOM_AGENT_STATS_STATIC_SCHEMA
        self.AGENT_STATS_DYNAMIC_SCHEMA = self.CORE_AGENT_STATS_DYNAMIC_SCHEMA | self.CUSTOM_AGENT_STATS_DYNAMIC_SCHEMA
        self.OUTER_STATS_STATIC_SCHEMA = self.CORE_OUTER_STATS_STATIC_SCHEMA | self.CUSTOM_OUTER_STATS_STATIC_SCHEMA
        self.OUTER_STATS_DYNAMIC_SCHEMA = self.CORE_OUTER_STATS_DYNAMIC_SCHEMA | self.CUSTOM_OUTER_STATS_DYNAMIC_SCHEMA

        # --- Master key sets for easier lookup ---
        self.all_static_keys: set[str] = set()
        self.all_dynamic_keys: set[str] = set()
        self.all_keys: set[str] = set()
        self.world_grouped_keys: set[str] = set()
        self.world_ungrouped_keys: set[str] = set()
        self.agent_grouped_keys: set[str] = set()
        self.agent_ungrouped_keys: set[str] = set()
        self.stat_types = {}
        self._stat_defaults: dict[str, Any] = {}
        self._initialize_key_sets()

        if self.is_world:
            # --- World Configuration ---
            self._stats: dict[str, Any] = {self.GROUP_KEY: {}}
            self.min_window_duration = timedelta(hours=cache_window_hours)
            self.db_path = db_path
            self._db_conn: sqlite3.Connection | None = None
            self._static_db_buffer = []
            self._dynamic_db_buffer = []

            # --- World Initialization ---
            self._init_db()  # Connect and create tables
            self._initialize_cache_structure()  # Ensures all keys exist
            self._load_existing_stats()  # Hydrates _stats from disk
        else:
            # --- Agent Initialization (Simple Buffer) ---
            self._world_view: dict[str, Any] = {}
            self.min_window_duration = timedelta(hours=3.0)  # cache for the _world_view
            self._update_batch: list[dict[str, Any]] = []

    def _initialize_key_sets(self):
        """Populates the master key sets and the type for later use."""
        # Combine all schema definitions
        all_static_schemas = {
            **self.WORLD_STATS_STATIC_SCHEMA,
            **self.AGENT_STATS_STATIC_SCHEMA,
            **self.OUTER_STATS_STATIC_SCHEMA
        }

        all_dynamic_schemas = {
            **self.WORLD_STATS_DYNAMIC_SCHEMA,
            **self.AGENT_STATS_DYNAMIC_SCHEMA,
            **self.OUTER_STATS_DYNAMIC_SCHEMA
        }

        # Build the key sets AND the type map
        self.all_static_keys = set()
        for name, (type_obj, default) in all_static_schemas.items():
            self.all_static_keys.add(name)
            self.stat_types[name] = type_obj
            self._stat_defaults[name] = default

        self.all_dynamic_keys = set()
        for name, (type_obj, default) in all_dynamic_schemas.items():
            self.all_dynamic_keys.add(name)
            self.stat_types[name] = type_obj
            self._stat_defaults[name] = default

        self.all_keys = self.all_static_keys | self.all_dynamic_keys
        # World perspective
        self.world_ungrouped_keys = {name for name in self.WORLD_STATS_STATIC_SCHEMA | self.WORLD_STATS_DYNAMIC_SCHEMA}
        self.world_grouped_keys = {name for name in (self.AGENT_STATS_STATIC_SCHEMA | self.AGENT_STATS_DYNAMIC_SCHEMA |
                                                     self.OUTER_STATS_STATIC_SCHEMA | self.OUTER_STATS_DYNAMIC_SCHEMA)}
        self.agent_ungrouped_keys = {name for name in self.AGENT_STATS_STATIC_SCHEMA | self.AGENT_STATS_DYNAMIC_SCHEMA}
        self.agent_grouped_keys = {name for name in self.OUTER_STATS_STATIC_SCHEMA | self.OUTER_STATS_DYNAMIC_SCHEMA}

    def _init_db(self):
        """(World-only) Connects to SQLite and creates tables if they don't exist."""
        if not self.is_world:
            return

        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            self._db_conn = sqlite3.connect(self.db_path)
            self._db_conn.execute('PRAGMA journal_mode=WAL;')
            self._db_conn.execute('PRAGMA synchronous=NORMAL;')

            self._db_conn.executescript("""
                CREATE TABLE IF NOT EXISTS dynamic_stats (
                    timestamp INTEGER,
                    peer_id TEXT,
                    stat_name TEXT,
                    val_num REAL,
                    val_str TEXT,
                    val_json TEXT,
                    PRIMARY KEY (peer_id, stat_name, timestamp)
                );
                CREATE INDEX IF NOT EXISTS idx_stats_num ON dynamic_stats (stat_name, val_num);
                CREATE INDEX IF NOT EXISTS idx_stats_str ON dynamic_stats (stat_name, val_str);
                CREATE INDEX IF NOT EXISTS idx_stats_time ON dynamic_stats (timestamp);

                CREATE TABLE IF NOT EXISTS static_stats (
                    peer_id TEXT,
                    stat_name TEXT,
                    val_json TEXT,
                    timestamp INTEGER,
                    PRIMARY KEY (peer_id, stat_name)
                );
            """)
            self._db_conn.commit()
            print(f'SQLite DB initialized at {self.db_path}')
        except Exception as e:
            print(f'CRITICAL: Failed to initialize SQLite DB: {e}')
            self._db_conn = None

    def _initialize_cache_structure(self):
        """(World-only) Ensures the _stats dict has the correct structure (SortedDicts/dicts)."""
        if not self.is_world:
            return

        self._stats.setdefault(self.GROUP_KEY, {})
        for key in self.world_ungrouped_keys:
            if key in self.all_dynamic_keys:
                self._stats.setdefault(key, SortedDict())
            else:
                self._stats.setdefault(key, self._stat_defaults[key])  # e.g., 'graph'

        # Grouped keys are initialized on-demand by _get_group_stat_cache
        # But we must ensure existing loaded peers have their structures
        for _, peer_data in self._stats[self.GROUP_KEY].items():
            for key in self.world_grouped_keys:
                if key in self.all_dynamic_keys:
                    # If loaded from DB, it's not a SortedDict yet.
                    # It will be populated by _hydrate_dynamic_caches_from_db
                    peer_data.setdefault(key, SortedDict())

    def _get_group_stat_cache(self, group_key: str, stat_name: str) -> SortedDict | dict | None:
        """(World-only) Helper to get or create the cache structure for a grouped stat on demand.

        Args:
            group_key: Group key whose cache entry is retrieved or created.
            stat_name: Name of the statistic to look up within the group's cache.

        Returns:
            A ``SortedDict`` for dynamic stats, a plain ``dict`` (or scalar default) for
            static stats, or ``None`` if this instance is not the World.
        """
        if not self.is_world:
            return

        peer_cache = self._stats[self.GROUP_KEY].setdefault(group_key, {})
        if stat_name not in peer_cache:
            if stat_name in self.all_dynamic_keys:
                peer_cache[stat_name] = SortedDict()
            elif stat_name in self.all_static_keys:
                peer_cache[stat_name] = self._stat_defaults[stat_name]

        return peer_cache.get(stat_name)

    # --- SHARED API ---
    def store_stat(self, stat_name: str, value: Any, group_key: str, timestamp: int):
        """Unified API to store a stat. Dispatches to static or dynamic storage.

        Args:
            stat_name: Name of the statistic as defined in one of the schema dicts.
            value: Value to store. Type-validated and cast according to the schema.
            group_key: Grouping key for this stat (e.g. the sender's peer id, or any
                arbitrary identifier used to bucket related records together).
            timestamp: Millisecond Unix timestamp associated with the measurement.
        """
        if stat_name not in self.all_keys:
            log.error(f'Stat "{stat_name}" is not defined')
            return

        # disambiguate between static and dynamic stats
        if stat_name in self.all_static_keys:
            self._store_static(stat_name, value, group_key, timestamp)
        else:
            self._store_dynamic(stat_name, value, group_key, timestamp)

    def _validate_type(self, stat_name: str, value: Any) -> Any:
        """Validates and casts ``value`` to the type declared in the schema.

        If ``value`` is already the correct type it is returned unchanged, otherwise a
        cast is attempted. On failure the value is coerced to ``str`` and an error is
        logged.

        Args:
            stat_name: Name of the statistic whose schema type is looked up.
            value: Raw value to validate.

        Returns:
            The value cast to the schema-declared type, or a ``str`` fallback.

        Raises:
            KeyError: If ``stat_name`` is not present in ``stat_types``.
        """
        if stat_name not in self.stat_types:
            raise KeyError(f'Statistic "{stat_name}" is not defined in the stat_types schema.')

        schema_type = self.stat_types.get(stat_name)  # no default to str because it's a silent fail
        if isinstance(value, schema_type):
            return value
        else:
            try:
                # Try to safely cast it
                return schema_type(value)
            except (ValueError, TypeError, AttributeError):
                log.error(f'Type mismatch for {stat_name}: '
                          f'Expected {schema_type} but got {type(value)}. '
                          f'Value: "{value}". Storing as string.')
                return str(value)  # Fallback

    def _make_json_serializable(self, value: Any) -> Any:
        """Recursively converts non-serializable types (like sets) to lists.

        Args:
            value: Arbitrary Python value that may contain sets, dicts, lists, or scalars.

        Returns:
            A JSON-serializable equivalent of ``value``, with sets converted to lists
            and nested structures recursed into.
        """
        if isinstance(value, set):
            return list(value)
        if isinstance(value, dict):
            # Recurse on values
            return {k: self._make_json_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            # Recurse on items
            return [self._make_json_serializable(item) for item in value]

        # Add other types here if needed (e.g., numpy arrays -> lists)

        # Base case: value is fine as-is
        return value

    def _store_static(self, stat_name: str, value: Any, group_key: str, timestamp: int):
        """Stores a static (single-value) stat.

        On Agent: de-duplicates and appends to the network send buffer.
        On World: updates the hot cache and appends to the DB write buffer.

        Args:
            stat_name: Name of the static statistic.
            value: New value for the stat. Type-validated before storage.
            group_key: Grouping key for this stat.
            timestamp: Millisecond Unix timestamp of the measurement.
        """
        value = self._validate_type(stat_name, value)
        if self.is_world:
            # --- WORLD LOGIC ---
            if timestamp > self.max_seen_timestamp:
                self.max_seen_timestamp = timestamp
            # 1. Update hot cache
            if stat_name in self.world_ungrouped_keys:
                self._stats[stat_name] = value
            else:
                peer_cache = self._stats[self.GROUP_KEY].setdefault(group_key, {})
                peer_cache[stat_name] = value

            # 2. Add to DB buffer (key, value_json)
            serializable_value = self._make_json_serializable(value)
            self._static_db_buffer.append((group_key, stat_name, json.dumps(serializable_value), timestamp))
        else:
            # --- AGENT LOGIC ---
            # De-duplicate logic: remove previous static value for this group_key/stat
            self._update_batch = [u for u in self._update_batch
                                  if not (u['group_key'] == group_key and u['stat_name'] == stat_name)]

            # 2. Add to batch
            self._update_batch.append({
                'group_key': group_key,
                'stat_name': stat_name,
                'timestamp': timestamp,
                'value': value
            })

    def _store_dynamic(self, stat_name: str, value: Any, group_key: str, timestamp: int):
        """Stores a dynamic (time-series) stat.

        On Agent: appends to the network send buffer.
        On World: inserts into the rolling hot-cache ``SortedDict``, prunes old entries,
        and appends to the DB write buffer.

        Args:
            stat_name: Name of the dynamic statistic.
            value: New measurement value. Type-validated before storage.
            group_key: Grouping key for this stat.
            timestamp: Millisecond Unix timestamp used as the time-series key.
        """
        value = self._validate_type(stat_name, value)
        last_value = self.get_last_value(stat_name)
        if self.STORE_DYNAMIC_IF_CHANGED and last_value == value:
            return

        if self.is_world:
            # --- WORLD LOGIC ---
            if timestamp > self.max_seen_timestamp:
                self.max_seen_timestamp = timestamp

            # 1. Update hot cache
            if stat_name in self.world_ungrouped_keys:
                cache = self._stats.get(stat_name)
            else:
                cache = self._get_group_stat_cache(group_key, stat_name)

            # Verify we have a valid SortedDict to work with
            if isinstance(cache, SortedDict):
                # Insert new value and prune outdated ones
                cache[timestamp] = value
                cutoff = timestamp - int(self.min_window_duration.total_seconds() * 1000)
                while cache and cache.peekitem(0)[0] < cutoff:
                    cache.popitem(0)

            # 2. Add to DB buffer depending on the type (value was already cast to the type defined in the schema)
            val_num = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            val_str = value if isinstance(value, str) else None
            # always create the json-serialized as fallback
            serializable_value = self._make_json_serializable(value)
            val_json = json.dumps(serializable_value)
            self._dynamic_db_buffer.append((timestamp, group_key, stat_name, val_num, val_str, val_json))
        else:
            # --- AGENT LOGIC ---
            self._update_batch.append({
                'group_key': group_key,
                'stat_name': stat_name,
                'timestamp': timestamp,
                'value': value
            })

    # --- AGENT API ---
    def update_view(self, view_data: dict[str, Any] | None = None, overwrite: bool = False):
        """(Agent-side) Merges a snapshot received from the World into the local view.

        This is 'dumb' storage: the data is kept as-is for later plotting without further
        parsing. Dynamic stats are extended (appended) rather than replaced unless
        ``overwrite`` is set.

        Expected ``view_data`` structure::

            {
                "world": { "stat_name": value_or_timeseries },
                "peers": { "peer_id": { "stat_name": value_or_timeseries } }
            }

        For dynamic stats, each ``value_or_timeseries`` is a list of ``[timestamp, value]``
        pairs for efficient JSON/Plotly usage.

        Args:
            view_data: The snapshot received from the world.
            overwrite: If ``True``, discards the current view before merging the new data.
        """
        if self.is_world:
            return

        # Initialize empty structure if needed
        if not self._world_view or overwrite:
            self._world_view = {'world': {}, 'peers': {}}

        def _update_max_ts(ts):
            """Helper to update the max seen timestamp from a time-series."""
            # Dynamic stats come as [[ts, val], [ts, val]...]
            if isinstance(ts, list) and len(ts) > 0 and isinstance(ts[0], list):
                # The last item is usually the newest in sorted time-series
                last_ts = ts[-1][0]
                if last_ts > self.max_seen_timestamp:
                    self.max_seen_timestamp = int(last_ts)

        def _merge_dict(target: dict, source: dict):
            """
            Helper to merge source into target with special handling for dynamic stats.
            Copies a source dict { "stat_name": value_or_timeseries } into target.
            """
            for stat_name, val_or_ts in source.items():
                if stat_name in self.all_dynamic_keys:
                    _update_max_ts(val_or_ts)
                    if stat_name not in target:
                        target[stat_name] = []
                    target[stat_name].extend(val_or_ts)
                else:
                    target[stat_name] = val_or_ts

        # 1. Merge World (Ungrouped) Stats
        if 'world' in view_data:
            _merge_dict(self._world_view.setdefault('world', {}), view_data['world'])

        # 2. Merge Peer (Grouped) Stats
        if 'peers' in view_data:
            target_peers = self._world_view.setdefault('peers', {})
            for peer_id, peer_data in view_data['peers'].items():
                target_peer = target_peers.setdefault(peer_id, {})
                _merge_dict(target_peer, peer_data)

    def _get_last_val_from_view(self, view: dict[str, Any], name: str) -> str:
        """Extracts the most recent scalar value for a world-level stat from the view snapshot.

        For dynamic stats (stored as ``[[timestamp, value], ...]``), the last pair's value
        is returned. For static stats, the value itself is returned. Only world-level
        (ungrouped) stats are searched.

        Args:
            view: View snapshot dictionary as returned by ``get_view`` or stored in
                ``_world_view``.
            name: Stat name to look up.

        Returns:
            A string representation of the latest value, ``"-"`` if the stat is absent.
            Float values are formatted to three decimal places.
        """
        val = None
        # Try World (Ungrouped)
        if name in view.get('world', {}):
            data = view['world'][name]
            # If dynamic (list of lists), get last value. If static, get value.
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                val = data[-1][1]
            else:
                val = data

        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val) if val is not None else "-"

    def get_stats(self) -> dict[str, Any]:
        """Returns the raw internal stats dictionary (World-only).

        Returns:
            The ``_stats`` hot-cache dictionary containing ungrouped world stats and
            the peer-grouped stats keyed by ``GROUP_KEY``.
        """
        return self._stats

    def get_payload_for_world(self, clear_buffer: bool = True) -> list[dict[str, Any]]:
        """(Agent-only) Gathers, returns, and clears all pending stats to be sent to the world.

        Args:
            clear_buffer: Whether to clear the stats buffer after the get operation (default True).

        Returns:
            List of stat update dicts, each with ``'peer_id'``, ``'stat_name'``,
            ``'timestamp'``, and ``'value'`` keys. Returns an empty list on the World side.
        """
        if self.is_world:
            return []

        # self._update_agent_static()  # Ensure static stats are fresh in the batch
        payload = self._update_batch
        if clear_buffer:
            self._update_batch = []  # Clear after getting
        return payload

    # --- WORLD API ---
    def get_view(self, since_timestamp: int = 0) -> dict[str, Any]:
        """(World-side) Returns a JSON-serializable snapshot of the current in-memory cache.

        Used for initial handshakes or lightweight polling. Dynamic stats are filtered to
        only include entries newer than ``since_timestamp``.

        Args:
            since_timestamp: Millisecond Unix timestamp. Only dynamic data points with a
                timestamp strictly greater than this value are included. Pass ``0`` to
                include all cached data.

        Returns:
            Dictionary with the structure::

                {
                    "world": { "stat_name": value_or_timeseries },
                    "peers": { "peer_id": { "stat_name": value_or_timeseries } }
                }

            For dynamic stats, ``value_or_timeseries`` is a list of ``[timestamp, value]``
            pairs. Returns an empty dict when called on the Agent side.
        """
        if not self.is_world:
            return {}
        snapshot = {'world': {}, 'peers': {}}

        # 1. Process World (Ungrouped) Stats
        for stat_name in self.world_ungrouped_keys:
            val = self._stats.get(stat_name)
            if val is not None:
                snapshot['world'][stat_name] = self._serialize_value(val, since_timestamp)

        # 2. Process Peer (Grouped) Stats
        peer_groups = self._stats.get(self.GROUP_KEY, {})

        for pid in peer_groups.keys():
            peer_data = {}
            for stat_name, val in peer_groups[pid].items():
                serialized = self._serialize_value(val, since_timestamp)
                # Optimize: Don't send empty lists if polling
                if isinstance(serialized, list) and len(serialized) == 0:
                    continue
                peer_data[stat_name] = serialized

            if peer_data:
                snapshot['peers'][pid] = peer_data

        return snapshot

    def _serialize_value(self, value: Any, since_timestamp: int) -> Any:
        """Converts a cached value to a JSON-serializable form.

        ``SortedDict`` caches are sliced from ``since_timestamp`` and converted to
        ``[[timestamp, value], ...]`` lists. All other values are passed through
        ``_make_json_serializable``.

        Args:
            value: A ``SortedDict`` time-series cache, or any static value.
            since_timestamp: Lower-bound millisecond timestamp for slicing dynamic data.

        Returns:
            A list of ``[timestamp, value]`` pairs for dynamic stats, or the JSON-safe
            equivalent of ``value`` for static stats.
        """
        if isinstance(value, SortedDict):
            idx = value.bisect_left(since_timestamp)
            sliced_items = value.items()[idx:]
            # Convert to list of [timestamp, value] for Plotly readiness
            return [[k, self._make_json_serializable(v)] for k, v in sliced_items]
        else:
            # Static value: return as is (assuming it's serializable)
            return self._make_json_serializable(value)

    def get_last_value(self, stat_name: str, group_key: str | None = None) -> Any | None:
        """Returns the most recent value of any stat, whether static or dynamic.

        Args:
            stat_name: Name of the statistic to retrieve.
            group_key: When ``None``, looks up an ungrouped (world-level) stat. When
                provided, looks up the grouped stat under that key.

        Returns:
            The most recent value, or ``None`` if the stat is unknown or has no data.
        """
        if stat_name in self.all_static_keys:
            return self._get_last_static_value(stat_name, group_key)
        elif stat_name in self.all_dynamic_keys:
            return self._get_last_dynamic_value(stat_name, group_key)
        else:
            log.error(f'get_last_value: Unknown stat_name "{stat_name}"')
            return None

    def _get_last_dynamic_value(self, stat_name: str, group_key: str | None = None) -> Any | None:
        """Returns the most recent value of a dynamic stat from the hot cache.

        Args:
            stat_name: Name of the dynamic statistic.
            group_key: When ``None``, searches for an ungrouped (world-level) stat. When
                provided, searches for the grouped stat under that key.

        Returns:
            The most recently recorded value, or ``None`` if the stat is not found, the
            cache is empty, or this instance is not the World.
        """
        if not self.is_world:
            # Agents look in their local cache
            for _update_batch in reversed(self._update_batch):
                if _update_batch['stat_name'] == stat_name and _update_batch['group_key'] == group_key:
                    return _update_batch['value']
            return None

        cache: SortedDict | None = None

        if group_key is None:
            # --- This is an ungrouped (world) stat ---
            if stat_name in self.world_ungrouped_keys:
                cache = self._stats.get(stat_name)
        else:
            # --- This is a grouped stat ---
            if stat_name in self.world_grouped_keys:
                peer_cache = self._stats.get(self.GROUP_KEY, {}).get(group_key)
                if peer_cache:
                    cache = peer_cache.get(stat_name)

        # Check if we found a valid SortedDict cache and it's not empty
        if isinstance(cache, SortedDict) and cache:
            return cache.peekitem(-1)[1]  # Return the last value

        return None  # Stat not found or no values

    def _get_last_static_value(self, stat_name: str, group_key: str | None = None) -> Any | None:
        """Returns the current value of a static stat from the hot cache.

        Args:
            stat_name: Name of the static statistic.
            group_key: When ``None``, searches for an ungrouped (world-level) stat. When
                provided, searches for the grouped stat under that key.

        Returns:
            The current cached value, or ``None`` if the stat is not found or this
            instance is not the World.
        """
        if not self.is_world:
            return None  # Agents don't have this cache

        value: Any | None = None
        if group_key is None:
            # --- This is an ungrouped (world) stat ---
            if stat_name in self.world_ungrouped_keys:
                value = self._stats.get(stat_name)
        else:
            # --- This is a grouped stat ---
            if stat_name in self.world_grouped_keys:
                peer_cache = self._stats.get(self.GROUP_KEY, {}).get(group_key)
                if peer_cache:
                    value = peer_cache.get(stat_name)
        return value

    # --- WORLD API (PERSISTENCE) ---
    def save_to_disk(self):
        """(World-only) Saves the static snapshot and dynamic buffer to SQLite."""
        if not self.is_world or not self._db_conn:
            return
        log.debug(f'Saving world stats to DB...')
        try:
            self._save_static_to_db()
            self._save_dynamic_to_db()
            self._prune_cache()
            self._prune_db()

            self._db_conn.commit()
            log.debug(f'Save complete.')
        except Exception as e:
            log.error(f'CRITICAL: Save_to_disk failed: {e}')
            if self._db_conn:
                self._db_conn.rollback()

    def _save_static_to_db(self):
        """(World-only) Dumps all static stats from hot cache to DB."""
        if not self._static_db_buffer or not self._db_conn:
            return

        self._db_conn.executemany("""
            INSERT INTO static_stats (peer_id, stat_name, val_json, timestamp)
            VALUES (?, ?, ?, ?) ON CONFLICT(peer_id, stat_name) DO UPDATE
            SET val_json = excluded.val_json, timestamp = excluded.timestamp
        """, self._static_db_buffer)

        self._static_db_buffer = []  # Clear buffer

    def _save_dynamic_to_db(self):
        """(World-only) Writes the in-memory dynamic buffer to SQLite."""
        if not self._dynamic_db_buffer or not self._db_conn:
            return

        self._db_conn.executemany("""
            INSERT OR IGNORE INTO dynamic_stats 
            (timestamp, peer_id, stat_name, val_num, val_str, val_json) 
            VALUES (?, ?, ?, ?, ?, ?)
        """, self._dynamic_db_buffer)

        log.debug(f'Wrote {len(self._dynamic_db_buffer)} dynamic stats to SQLite.')
        self._dynamic_db_buffer = []  # Clear buffer

    def _prune_db(self):
        """(World-only) Add here the logic to prune the db (e.g., when a peer leaves the world)."""
        if not self._db_conn:
            return
        pass

    def _prune_cache(self):
        """
        (World-only) Periodic maintenance to clean up 'stale' peers.
        
        The 'prune-on-write' logic in _store_dynamic handles active peers efficiently.
        This method handles peers that have disconnected or stopped sending data,
        preventing their old data from haunting the RAM forever.
        """
        if not self.is_world:
            return

        # Calculate cutoff based on latest timestamp
        window_ms = int(self.min_window_duration.total_seconds() * 1000)
        cutoff = self.max_seen_timestamp - window_ms

        # 1. Prune Ungrouped Stats (World Stats)
        for key in self.world_ungrouped_keys:
            cache = self._stats.get(key)
            if isinstance(cache, SortedDict):  # only true for dynamic stats
                # Remove items older than cutoff
                while cache and cache.peekitem(0)[0] < cutoff:
                    cache.popitem(0)

        # 2. Prune Grouped Stats (Peer Stats)
        peer_groups = self._stats.get(self.GROUP_KEY, {})

        # We might need to remove empty peers entirely, so we collect keys to delete
        peers_to_remove = []

        for peer_id, peer_cache in peer_groups.items():
            all_stats_were_empty = True
            for _, stat_data in peer_cache.items():
                if isinstance(stat_data, SortedDict):  # only true for dynamic stats
                    # Prune the time series
                    while stat_data and stat_data.peekitem(0)[0] < cutoff:
                        stat_data.popitem(0)
                    # after pruning, check if the stat dict is empty
                    all_stats_were_empty &= len(stat_data) == 0

            if all_stats_were_empty:
                peers_to_remove.append(peer_id)

        # Remove completely dead peers from memory
        for peer_id in peers_to_remove:
            del peer_groups[peer_id]
            log.debug(f'Pruned stale peer {peer_id} from cache.')

    # --- WORLD API (LOADING) ---
    def _load_existing_stats(self):
        """(World-only) Loads existing stats from disk to hydrate the cache."""
        if not self.is_world or not self._db_conn:
            return
        print('Loading existing stats from disk...')
        self._load_static_from_db()
        self._hydrate_dynamic_caches_from_db()
        print('Finished loading stats.')

    def _load_static_from_db(self):
        """(World-only) Loads the static_stats table into the _stats hot cache."""
        # There are no default static stats that are meaningful to load at startup (graph, state...)
        pass

    def _hydrate_dynamic_caches_from_db(self):
        """(World-only) Queries SQLite for 'hot' data to fill dynamic caches."""
        if not self._db_conn:
            return
        try:
            max_ts_cursor = self._db_conn.execute('SELECT MAX(timestamp) FROM dynamic_stats')
            max_ts_result = max_ts_cursor.fetchone()

            if max_ts_result is None or max_ts_result[0] is None:
                log.debug('No dynamic stats found in DB. Hydration skipped.')
                return  # No data in DB, nothing to load
            self.max_seen_timestamp = int(max_ts_result[0])
            cutoff_t_ms = self.max_seen_timestamp - int(self.min_window_duration.total_seconds() * 1000)

            cursor = self._db_conn.execute("""
                SELECT timestamp, peer_id, stat_name, val_num, val_str, val_json 
                FROM dynamic_stats 
                WHERE timestamp > ?
                ORDER BY timestamp
            """, (cutoff_t_ms,))

            count = 0
            for ts, group_key, stat_name, _, _, val_json in cursor:
                ts = int(ts)
                # we just need the val_json that will be cast to the exact type by _validate_type
                value = json.loads(val_json)
                self._store_dynamic(stat_name, value, group_key, ts)
                count += 1

            # Clear the buffer generated by hydrating
            self._dynamic_db_buffer = []

            if count > 0:
                print(f'Hydrated cache with {count} recent dynamic stats.')
            else:
                print('No recent dynamic stats found in DB.')

        except Exception as e:
            print(f'Failed to hydrate dynamic caches from DB: {e}')

    # --- WORLD API (QUERYING) ---
    def query_history(self,
                      stat_names: list[str] | None = None,
                      group_keys: list[str] | None = None,
                      time_range: tuple[int, int] | int | None = None,
                      value_range: tuple[float, float] | None = None,
                      limit: int | None = None) -> dict[str, Any]:
        """(World-only) Queries the SQLite DB for historical stats with optional filters.

        Automatically flushes the current memory buffers to the DB before querying to
        ensure "read-your-writes" consistency. Returns the same structure as
        ``get_view()``, so the Agent can ingest it seamlessly.

        Args:
            stat_names: Restrict results to these stat names. Pass ``None`` or ``[]``
                to include all stats.
            group_keys: Restrict results to these group keys. Pass ``None`` or ``[]``
                to include all groups.
            time_range: Time filter for dynamic stats. An ``int`` is treated as a
                "since X" lower bound; a ``(start, end)`` tuple filters to the closed
                interval ``[start, end]``. Pass ``None`` for no time filter.
            value_range: ``(min, max)`` numeric range filter applied to ``val_num``.
                Rows without a numeric value are excluded when this filter is active.
            limit: Maximum number of dynamic-stat rows to return. Defaults to 5000.

        Returns:
            Dictionary with the same structure as ``get_view()``. Returns an empty dict
            when called on the Agent side or when the DB connection is unavailable.
        """
        if stat_names is None:
            stat_names = []
        if group_keys is None:
            group_keys = []
        if not self.is_world or not self._db_conn:
            return {}

        # Flush the cached upadtes to db before querying
        self._save_static_to_db()
        self._save_dynamic_to_db()
        self._db_conn.commit()

        snapshot = {'world': {}, 'peers': {}}

        # A. Query the static stats
        query_static = ['SELECT peer_id, stat_name, val_json FROM static_stats']
        params_static = []

        where_added = False
        if stat_names:
            query_static.append("WHERE")
            where_added = True
            query_static.append(f"stat_name IN ({','.join(['?'] * len(stat_names))})")
            params_static.extend(stat_names)
        if group_keys:
            if not where_added:
                query_static.append("WHERE")
            else:
                query_static.append(f"AND")
            query_static.append(f"peer_id IN ({','.join(['?'] * len(group_keys))})")
            params_static.extend(group_keys)

        try:
            cursor = self._db_conn.execute(' '.join(query_static), params_static)
            for pid, sname, vjson in cursor:
                val = self._validate_type(sname, json.loads(vjson))
                # Handle special Graph reconstruction if needed (legacy format support)
                if sname == 'graph':
                    # Handle both legacy format (just edges) and new format (nodes+edges) safely
                    if isinstance(val, dict) and 'edges' in val:
                        # Convert the edge lists back to sets
                        val['edges'] = {k: set(v) for k, v in val['edges'].items()}
                        # Ensure nodes dict exists
                        if 'nodes' not in val:
                            val['nodes'] = {}
                    else:
                        val: dict
                        # Convert entire dict to sets (as it was before)
                        edges_set = {k: set(v) for k, v in val.items()}
                        # Migrate to new structure on the fly
                        val = {'nodes': {}, 'edges': edges_set}

                # Static stats format: value (direct)
                if pid in (None, 'None', ''):
                    snapshot['world'][sname] = val
                else:
                    snapshot['peers'].setdefault(pid, {})[sname] = val
        except Exception as e:
            log.error(f'Query history (static) failed: {e}')

        # B. Query the dynamic stats
        query_dyn = ['SELECT timestamp, peer_id, stat_name, val_num, val_str, val_json FROM dynamic_stats']
        params_dyn = []

        # 1. Stat Names
        where_added = False
        if stat_names:
            query_static.append("WHERE")
            where_added = True
            query_dyn.append(f"stat_name IN ({','.join(['?'] * len(stat_names))})")
            params_dyn.extend(stat_names)

        # 2. Group Keys
        if group_keys:
            if not where_added:
                query_static.append("WHERE")
            else:
                query_static.append(f"AND")
            query_dyn.append(f"peer_id IN ({','.join(['?'] * len(group_keys))})")
            params_dyn.extend(group_keys)

        if time_range is not None:
            if isinstance(time_range, int):
                # Treated as "Since X"
                if not where_added:
                    query_static.append("WHERE")
                else:
                    query_static.append(f"AND")
                query_dyn.append("timestamp >= ?")
                params_dyn.append(time_range)
            elif isinstance(time_range, (tuple, list)) and len(time_range) == 2:
                # Treated as "Between X and Y"
                if not where_added:
                    query_static.append("WHERE")
                else:
                    query_static.append(f"AND")
                query_dyn.append("timestamp >= ? AND timestamp <= ?")
                params_dyn.extend([time_range[0], time_range[1]])

        # 4. Value Range (The logic requested by user)
        if value_range:
            if not where_added:
                query_static.append("WHERE")
            else:
                query_static.append(f"AND")
            query_dyn.append("val_num IS NOT NULL AND val_num >= ? AND val_num <= ?")
            params_dyn.extend([value_range[0], value_range[1]])

        query_dyn.append("ORDER BY timestamp ASC")

        # add the limit
        query_dyn.append("LIMIT 5000" if limit is None else f"LIMIT {limit}")

        try:
            cursor = self._db_conn.execute(' '.join(query_dyn), params_dyn)
            for ts, pid, sname, vnum, vstr, vjson in cursor:
                ts = int(ts)
                val = vnum if vnum is not None else (vstr if vstr is not None else json.loads(vjson))
                val = self._validate_type(sname, val)

                # Structure construction
                if pid in (None, 'None', ''):  # Handling world stats
                    target_ts = snapshot['world'].setdefault(sname, [])
                else:  # Handling peer stats
                    target_ts = snapshot['peers'].setdefault(pid, {}).setdefault(sname, [])
                target_ts.append([ts, val])

        except Exception as e:
            log.error(f'Query history failed: {e}')

        return snapshot

    def _aggregate_time_indexed_stats_over_peers(self, stats: dict[str, Any]) -> tuple[dict, dict]:
        """(World-only) Aggregates time-indexed peer stats (mean/std) from the cache.

        For each numeric stat present in at least one peer's ``SortedDict`` cache, builds
        a time-aligned mean and standard-deviation series across all peers.

        Args:
            stats: The raw ``_stats`` dict (as returned by ``get_stats()``), containing
                ungrouped world stats and the ``GROUP_KEY`` sub-dict of peer caches.

        Returns:
            A two-element tuple ``(mean_dict, std_dict)`` where each element maps
            ``stat_name -> {timestamp: value}``. Values may be ``None`` for timestamps
            with no valid peer data.
        """
        mean_dict = {}
        std_dict = {}
        peer_stats = stats.get(self.GROUP_KEY, {})

        number_stats = {name for name, type_obj in self.stat_types.items()
                        if type_obj in (int, float)}

        for stat_name in number_stats:
            peer_series = []
            for _, peer_data in peer_stats.items():
                if stat_name in peer_data:
                    tv_dict: SortedDict = peer_data[stat_name]
                    if tv_dict:
                        peer_series.append(tv_dict)

            if not peer_series:
                continue

            all_times = sorted({t for series in peer_series for t in series.keys()})
            if not all_times:
                continue

            aligned_values = []
            for series in peer_series:
                if not series:
                    continue
                filled = []
                series_times = list(series.keys())
                series_vals = list(series.values())

                last_val = series_vals[0]
                series_idx = 0

                for t in all_times:
                    while series_idx < len(series_times) and series_times[series_idx] <= t:
                        last_val = series_vals[series_idx]
                        series_idx += 1
                    filled.append(last_val)
                aligned_values.append(filled)

            if not aligned_values:
                continue

            mean_dict[stat_name] = {}
            std_dict[stat_name] = {}
            for i, t in enumerate(all_times):
                vals = [peer_vals[i] for peer_vals in aligned_values if peer_vals[i] is not None]
                if vals:
                    mean_val = sum(vals) / float(len(vals))
                    var = sum((x - mean_val) ** 2 for x in vals) / len(vals)
                    std_val = math.sqrt(var)
                else:
                    mean_val = None
                    std_val = None

                mean_dict[stat_name][t] = mean_val
                std_dict[stat_name][t] = std_val

        return mean_dict, std_dict

    def shutdown(self) -> None:
        """Flushes pending stats and closes the SQLite connection.

        Call this explicitly when the application is shutting down to ensure all
        buffered data is persisted. On the Agent side this is a no-op.
        """
        if self.is_world and self._db_conn:
            log.debug('Shutdown: Saving final stats...')
            try:
                self.save_to_disk()
            except Exception as e:
                log.error(f'Shutdown save failed: {e}')
            self._db_conn.close()
            self._db_conn = None
            log.debug('SQLite connection closed.')

    def __del__(self):
        if self.is_world and self._db_conn:
            try:
                # Final save on exit, if any buffer
                self.save_to_disk()
            except Exception:
                pass  # Don't raise in destructor
            self._db_conn.close()
            print('SQLite connection closed.')

    # --- PLOTTING INTERFACE ---
    def plot(self, since_timestamp: int = 0) -> str | None:
        """Builds and returns the default 2x2 dashboard as a self-contained HTML document.

        Visualizes core stats: network topology, agent-count history, state distribution,
        and last-action distribution. Plotly.js is loaded from CDN; Python never imports
        the plotly package.

        Args:
            since_timestamp: Millisecond Unix timestamp. Only dynamic data points newer
                than this value are included in the time-series panels. Pass ``0`` to
                include all cached data.

        Returns:
            HTML string (``<!DOCTYPE html>`` document), or ``None`` if no view data
            is available.
        """
        from .stats_html_renderer import render_plotly_html

        # 1. Get Data view
        view = self.get_view(since_timestamp) if self.is_world else self._world_view
        if not view:
            return None

        dash = DefaultBaseDash("World Overview")

        # --- Panel 1: Network Topology (Top Left) ---
        p1 = UIPlot(title="World Topology")
        self._populate_graph(p1, view, "graph")
        clean_axis = {'showgrid': False, 'showticklabels': False, 'zeroline': False}
        p1.set_layout_opt('xaxis', clean_axis)
        p1.set_layout_opt('yaxis', clean_axis)
        dash.add_panel(p1, "top_left")

        # --- Panel 2: System Counters (Table) ---
        p2 = UIPlot(title="World Agents History")
        metrics = [
            ("world_masters", "World Masters", THEME['peers'][0]),
            ("world_agents", "World Agents", THEME['peers'][1]),
            ("human_agents", "Human Agents", THEME['peers'][2]),
            ("artificial_agents", "Artificial Agents", THEME['peers'][3]),
        ]
        for stat_key, label, color in metrics:
            self._populate_time_series(
                panel=p2,
                view=view,
                stat_name=stat_key,
                color_override=color,
                title_override=label
            )
        p2.set_layout_opt('xaxis', {'title': None, 'showticklabels': False})
        p2.set_layout_opt('yaxis', {'title': None})
        dash.add_panel(p2, "top_right")

        # --- Panel 3: State Distribution (Bar) ---
        p3 = UIPlot(title="State Distribution")
        self._populate_graph(p3, view, "graph")
        clean_axis = {'showgrid': False, 'showticklabels': False, 'zeroline': False}
        p3.set_layout_opt('xaxis', clean_axis)
        p3.set_layout_opt('yaxis', clean_axis)
        p3.set_layout_opt("xaxis", {"title": None})
        dash.add_panel(p3, "bot_left")

        # --- Panel 4: Action Distribution (Bar) ---
        p4 = UIPlot(title="Last Action Distribution")
        self._populate_distribution(p4, view, "last_action")
        p4.set_layout_opt("xaxis", {"title": None})
        dash.add_panel(p4, "bot_right")

        return render_plotly_html(dash.to_json())

    def _populate_time_series(self, panel: 'UIPlot', view: dict[str, Any], stat_name: str,
                              peer_ids: list[str] | None = None, color_override: str | None = None,
                              show_legend: bool = True, title_override: str | None = None):
        """Extracts ``[[t, v], ...]`` time-series data and adds line traces to a panel.

        Args:
            panel: Target ``UIPlot`` instance to receive the line traces.
            view: View snapshot dictionary as produced by ``get_view`` or ``_world_view``.
            stat_name: Name of the dynamic stat to plot.
            peer_ids: Subset of peer IDs to include. Pass ``None`` to include all peers
                present in the view.
            color_override: CSS color string applied to all traces. Defaults to
                per-peer consistent colors or the primary theme color for world data.
            show_legend: Whether to show these traces in the legend.
            title_override: Custom label for the world-level trace. Defaults to
                ``"World"``.
        """

        def get_xy(raw):
            if isinstance(raw, list) and raw and isinstance(raw[0], list):
                return [r[0] for r in raw], [r[1] for r in raw]
            return [], []

        # World
        w_data = view.get('world', {}).get(stat_name)
        if w_data:
            x, y = get_xy(w_data)
            if x:
                label = title_override if title_override else "World"
                color = color_override if color_override else THEME['main']
                panel.add_line(x, y, name=label, color=color,
                               legend_group=label, show_legend=show_legend)

        # Peers
        peers_dict = view.get('peers', {})
        targets = peer_ids if peer_ids else peers_dict.keys()
        for pid in targets:
            p_data = peers_dict.get(pid, {}).get(stat_name)
            if p_data:
                x, y = get_xy(p_data)
                if x:
                    c = color_override or self._get_consistent_color(pid)
                    panel.add_line(x, y, name=f'{pid[-6:]}', color=c,
                                   legend_group=pid, show_legend=show_legend)

    def _populate_indicator(self, panel: 'UIPlot', view: dict[str, Any], stat_name: str,
                            peer_ids: list[str] | None = None):
        """Extracts a scalar value and adds a big-number indicator trace to the panel.

        Args:
            panel: Target ``UIPlot`` instance to receive the indicator.
            view: View snapshot dictionary.
            stat_name: Stat name whose value is displayed.
            peer_ids: Peer IDs to search when the stat is not found at world level.
                The first available peer's value is used. Pass ``None`` to search all peers.
        """
        val = None
        if 'world' in view and stat_name in view['world']:
            val = view['world'][stat_name]
        elif 'peers' in view:
            # Just grab the first available peer's value if not specified
            targets = peer_ids if peer_ids else list(view['peers'].keys())
            if targets:
                val = view['peers'][targets[0]].get(stat_name)

        panel.add_indicator(val, title=stat_name)

    def _populate_table(self, panel: 'UIPlot', view: dict[str, Any], stat_name: str,
                        peer_ids: list[str] | None = None):
        """Builds a two-column ``Entity / Value`` table and adds it to the panel.

        Args:
            panel: Target ``UIPlot`` instance to receive the table trace.
            view: View snapshot dictionary.
            stat_name: Stat name to look up for each entity.
            peer_ids: Subset of peer IDs to include. Pass ``None`` to include all peers.
        """
        headers = ['Entity', 'Value']
        col_ent = []
        col_val = []

        # World
        if 'world' in view and stat_name in view['world']:
            col_ent.append('World')
            col_val.append(str(view['world'][stat_name]))

        # Peers
        peers_dict = view.get('peers', {})
        targets = peer_ids if peer_ids else peers_dict.keys()
        for pid in targets:
            val = peers_dict.get(pid, {}).get(stat_name)
            if val is not None:
                col_ent.append(pid[-6:])
                col_val.append(str(val))  # Simple stringification

        panel.add_table(headers, [col_ent, col_val])

    def _populate_graph(self, panel: 'UIPlot', view: dict[str, Any], stat_name: str):
        """Renders a network graph (nodes + edges) onto the panel using a circular layout.

        Args:
            panel: Target ``UIPlot`` instance to receive the edge and node traces.
            view: View snapshot dictionary containing the world-level graph stat.
            stat_name: Name of the stat that holds the graph data (typically ``'graph'``).
        """

        # 1. Fetch Data
        raw_graph = view.get('world', {}).get(stat_name, {})
        if not raw_graph:
            return

        # Handle both legacy format (just edges) and new format (nodes+edges) safely
        if 'edges' in raw_graph and 'nodes' in raw_graph:
            edges_data = raw_graph['edges']
            nodes_data = raw_graph['nodes']
        else:
            # Fallback for simple graphs without node details
            edges_data = raw_graph
            nodes_data = {}

        if edges_data is None:
            edges_data = {}
        if nodes_data is None:
            nodes_data = {}

        # 2. Calculate Layout (Circular)
        # We use edges_data keys for positioning, but we might have nodes in nodes_data
        # that have no edges yet, so we merge them.
        all_pids = set(edges_data.keys()).union(*edges_data.values()) | set(nodes_data.keys())
        pids = list(all_pids)
        pos = {}
        if pids:
            radius = 10
            angle_step = (2 * math.pi) / len(pids)
            for i, pid in enumerate(pids):
                pos[pid] = (
                    radius * math.cos(i * angle_step),
                    radius * math.sin(i * angle_step)
                )

        # 3. Create Edge Trace
        edge_x, edge_y = [], []
        for source, targets in edges_data.items():
            if source not in pos:
                continue
            x0, y0 = pos[source]
            # targets might be a list (from JSON) or set (from local cache)
            target_iter = targets if isinstance(targets, (list, set)) else []
            for target in target_iter:
                if target in pos:
                    x1, y1 = pos[target]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

        panel.add_trace({
            'type': 'scatter', 'mode': 'lines',
            'x': edge_x, 'y': edge_y,
            'line': {'width': 0.5, 'color': THEME['edge']},
            'hoverinfo': 'none', 'showlegend': False
        })

        # 4. Create Node Trace
        node_x, node_y, node_text, node_color, node_labels = [], [], [], [], []
        for pid in pids:
            if pid not in pos:
                continue
            x, y = pos[pid]
            node_x.append(x)
            node_y.append(y)

            # Node labels
            node_labels.append(pid[-6:])
            # Build hover text
            if nodes_data:
                hover_text = ''
                for key, val in nodes_data.get(pid, {}).items():
                    hover_text += f'{key}: {val}<br>'
            else:
                hover_text = f'Peer ID: {pid}'
            node_text.append(hover_text)

            # Determine Color
            # You can customize this mapping based on your NodeProfile types
            node_color.append(self._get_consistent_color(pid))

        panel.add_trace({
            'type': 'scatter',
            'mode': 'markers+text',
            'x': node_x, 'y': node_y,
            'text': node_labels,
            'hovertext': node_text,
            'hoverinfo': 'text',
            'textposition': 'top center',
            'showlegend': False,
            'marker': {
                'color': node_color,
                'size': 12,
                'line': {'width': 2, 'color': THEME['edge']}
            }
        })

        # 5. Layout overrides
        # panel.set_layout_opt('xaxis', {'visible': False})
        # panel.set_layout_opt('yaxis', {'visible': False})

    def _populate_distribution(self, panel: 'UIPlot', view: dict[str, Any], stat_name: str):
        """Aggregates peer values into a frequency-count bar chart.

        Counts how many peers hold each distinct value for the given stat and adds a
        sorted bar chart (e.g., ``{"IDLE": 3, "RUNNING": 5}``).

        Args:
            panel: Target ``UIPlot`` instance to receive the bar trace.
            view: View snapshot dictionary.
            stat_name: Name of the static stat to aggregate across peers.
        """
        peers_dict = view.get('peers', {})
        counts = {}

        # 1. Aggregate
        for data in peers_dict.values():
            # Handle uninitialized or None values
            val_str = str(data.get(stat_name, 'Unknown'))
            counts[val_str] = counts.get(val_str, 0) + 1

        # 2. Sort for consistency (e.g., alphabetically by state name)
        sorted_keys = sorted(counts.keys())
        sorted_vals = [counts[k] for k in sorted_keys]
        colors = [self._get_consistent_color(k) for k in sorted_keys]

        # 3. Plot
        panel.add_bar(xs=sorted_keys, ys=sorted_vals, names=sorted_vals, colors=colors)

    def _get_consistent_color(self, unique_str: str) -> str:
        """Returns a deterministic color from the theme palette for a given string.

        Uses an Adler-32 checksum of the string to select a palette index, ensuring
        the same string always maps to the same color across renders.

        Args:
            unique_str: Any string (e.g., a peer ID or state name) used as the hash key.

        Returns:
            A CSS hex color string from the ``THEME['peers']`` palette. Returns
            ``'#ffffff'`` for empty strings.
        """
        if not unique_str:
            return '#ffffff'
        idx = zlib.adler32(str(unique_str).encode()) % len(THEME['peers'])
        return THEME['peers'][idx]

    # ==================================================================================================================
    # BEGIN OF DEPRECATED METHODS
    # ==================================================================================================================
    @deprecated("Use the new logger")
    def _out(self, msg: str):
        """DEPRECATED: Logs a stats-level message.

        Args:
            msg: Message text to log.
        """
        log.stats(msg)

    @deprecated("Use the new logger")
    def _err(self, msg: str):
        """DEPRECATED: Logs an error message.

        Args:
            msg: Error message text to log.
        """
        log.error(msg)

    @deprecated("Use the new logger")
    def _deb(self, msg: str):
        """DEPRECATED: Logs a debug message.

        Args:
            msg: Debug message text to log.
        """
        log.debug(msg)
    # ==================================================================================================================
    # END OF DEPRECATED METHODS
    # ==================================================================================================================
