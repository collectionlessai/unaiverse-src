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
from datetime import timedelta
from sortedcontainers import SortedDict
from typing import Any, Set, List, Dict, Tuple, Optional, Union


# A fixed palette for consistent coloring
THEME = {
    # Main structural colors (Dark Mode optimized)
    'bg_paper': 'rgba(0,0,0,0)',    # Transparent to blend with container
    'bg_plot': 'rgba(0,0,0,0)',     # Transparent plot area
    'text_main': '#7e7e7e',         # Primary text color
    'text_light': '#7e7e7e',        # Secondary/Axis text color
    
    # UI Element specific
    'grid': '#333333',              # Grid lines
    'edge': '#666666',              # Graph edges
    'node_border': '#ffffff',       # Node borders
    
    # Main Accents
    'main': '#636EFA',              # Primary accent (Blue)
    'main_light': '#aab1ff',        # Lighter shade of primary
    
    # Table Styling
    'table': {
        'header_bg': '#2c2c2c',
        'header_txt': '#ffffff',
        'cell_bg': '#1a1a1a',
        'cell_txt': '#dddddd',
        'line': '#444444'
    },

    # Data Categorical Palette (Plotly default set)
    'peers': [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
    ]
}


class UIPlot:
    """
    A Python abstraction for a UI Panel (specifically a Plotly chart).
    Allows users to build plots using Python methods instead of dicts/JSON.
    """
    def __init__(self, title: str = '', height: int = 400):
        self._data: List[Dict[str, Any]] = []
        
        # Define the standard axis style for a "boxed" look
        axis_style = {
            'gridcolor': THEME['grid'],
            'gridwidth': 1,
            'griddash': 'dot',
            'color': THEME['text_light'],
            'showline': True,           # Draw the axis line
            'mirror': True,             # Mirror it on top/right (creates the box)
            'linewidth': 2,             # Width of the box border
            'linecolor': THEME['grid'],  # Color of the box border
            'zeroline': False,          # Prevents double-thick borderlines at 0
            'layer': 'below traces'     # Key fix: puts grid BEHIND the box border
        }
        
        self._layout: Dict[str, Any] = {
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

    def add_line(self, x: List[Any], y: List[Any], name: str, color: str = THEME['main'],
                 legend_group: str = None, show_legend: bool = True):
        """Adds a standard time-series line."""
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

    def add_area(self, x: List[Any], y: List[Any], name: str, color: str = THEME['main']):
        """Adds a filled area chart."""
        trace = {
            'x': x, 'y': y, 'name': name,
            'type': 'scatter', 'fill': 'tozeroy',
            'line': {'color': color}
        }
        self._data.append(trace)

    def add_indicator(self, value: Any, title: str):
        """Adds a big number indicator."""
        self._data.append({
            'type': 'indicator',
            'mode': 'number',
            'value': value,
            'title': {'text': title}
        })
        self._layout['height'] = 300  # Indicators usually need less height

    def add_table(self, headers: List[str] | None, columns: List[List[Any]]):
        """Adds a data table."""
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
    
    def add_bar(self, xs: List[Any], ys: List[Any], names: List[str], colors: Union[str, List[str]] = THEME['main']):
        """Adds a bar chart trace."""
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
    
    def add_trace(self, trace: Dict[str, Any]):
        """Generic method to add any raw Plotly trace."""
        self._data.append(trace)

    def set_y_range(self, min_val: float, max_val: float):
        """Force Y-axis limits."""
        self._layout.setdefault('yaxis', {})['range'] = [min_val, max_val]

    def set_layout_opt(self, key: str, value: Any):
        """Generic setter for advanced layout options."""
        if isinstance(value, dict) and key in self._layout:
            self._layout[key].update(value)
        else:
            self._layout[key] = value
    
    def set_legend(self, orientation: str = 'v', x: float = 1.0, y: float = 1.0,
                   xanchor: str = 'left', yanchor: str = 'top'):
        """
        Configures the legend position and orientation.
        orientation: 'v' (vertical) or 'h' (horizontal)
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
        """Serializes the panel to the format the Frontend expects."""
        return json.dumps({'data': self._data, 'layout': self._layout})


class DefaultBaseDash:
    """
    A generic 2x2 Grid Dashboard for the base Stats class.
    Forces #111111 background to match the WStats styling.
    """
    def __init__(self, title="Network Overview"):
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
        if position not in self._map:
            return

        xa, ya = self._map[position]
        x_dom = self.layout[xa]["domain"]
        y_dom = self.layout[ya]["domain"]

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
            self.layout.setdefault("annotations", []).append({
                "text": f"<b>{src_l['title']}</b>",
                "x": (x_dom[0] + x_dom[1]) / 2, 
                "y": y_dom[1] + 0.02,
                "xref": "paper", "yref": "paper", 
                "showarrow": False, "xanchor": "center", "yanchor": "bottom",
                "font": {"size": 14, "color": THEME['text_main']} 
            })

    def to_json(self):
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
    DEBUG = True  # Turns on/off extra logging
    
    # These are all the keys in the local _stats dictionary collected by the world
    CORE_WORLD_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {
        'graph': (dict, {'nodes': {}, 'edges': {}})
    }
    CORE_WORLD_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {
        'world_masters': (int, 0),
        'world_agents': (int, 0),
        'human_agents': (int, 0),
        'artificial_agents': (int, 0)
    }

    # These are all the keys in the local _stats dictionary collected by the agent
    CORE_AGENT_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {
        'connected_peers': (list, []),
        'state': (str, None),
        'action': (str, None),
        'last_action': (str, None)
    }
    CORE_AGENT_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    
    # Then we have the stats collected on behalf of other peers (by the agent or the world)
    CORE_OUTER_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CORE_OUTER_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    
    # We also add class variables to extend these sets
    CUSTOM_WORLD_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CUSTOM_WORLD_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CUSTOM_AGENT_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CUSTOM_AGENT_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CUSTOM_OUTER_STATS_STATIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    CUSTOM_OUTER_STATS_DYNAMIC_SCHEMA: Dict[str, Tuple[type, Any]] = {}
    
    # Key for grouping stats in the _stats dictionary (both world and agent)
    GROUP_KEY = 'peer_stats'  # _BY_PEER stats are grouped under this key

    def __init__(self, is_world: bool,
                 db_path: str | None = None,  # only needed by the world
                 cache_window_hours: float = 2.0):  # only needed by the world
        
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
        self.all_static_keys: Set[str] = set()
        self.all_dynamic_keys: Set[str] = set()
        self.all_keys: Set[str] = set()
        self.world_grouped_keys: Set[str] = set()
        self.world_ungrouped_keys: Set[str] = set()
        self.agent_grouped_keys: Set[str] = set()
        self.agent_ungrouped_keys: Set[str] = set()
        self.stat_types: Dict[str, str] = {}
        self._stat_defaults: Dict[str, Any] = {}
        self._initialize_key_sets()

        if self.is_world:
            # --- World Configuration ---
            self._stats: Dict[str, Any] = {self.GROUP_KEY: {}}
            self.min_window_duration = timedelta(hours=cache_window_hours)
            self.db_path = db_path
            self._db_conn: Optional[sqlite3.Connection] = None
            self._static_db_buffer: List[Tuple[str, str]] = []
            self._dynamic_db_buffer: List[Tuple[float, str, str, str]] = []

            # --- World Initialization ---
            self._init_db()  # Connect and create tables
            self._initialize_cache_structure()  # Ensures all keys exist
            self._load_existing_stats()  # Hydrates _stats from disk
        else:
            # --- Agent Initialization (Simple Buffer) ---
            self._world_view: Dict[str, Any] = {}
            self.min_window_duration = timedelta(hours=3.0)  # cache for the _world_view
            self._update_batch: List[Dict[str, Any]] = []
    
    def _out(self, msg: str):
        """Prints a message using the node's out function."""
        print(msg)

    def _err(self, msg: str):
        """Prints an error message."""
        self._out('<ERROR> [Stats] ' + msg) 

    def _deb(self, msg: str):
        """Prints a debug message if enabled."""
        if self.DEBUG:
            prefix = '[DEBUG ' + ('WORLD' if self.is_world else 'AGENT') + ']'
            self._out(f'{prefix} [Stats] {msg}')
    
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
            self._deb(f'SQLite DB initialized at {self.db_path}')
        except Exception as e:
            self._err(f'CRITICAL: Failed to initialize SQLite DB: {e}')
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

        # Grouped keys are initialized on-demand by _get_peer_stat_cache
        # But we must ensure existing loaded peers have their structures
        for _, peer_data in self._stats[self.GROUP_KEY].items():
            for key in self.world_grouped_keys:
                if key in self.all_dynamic_keys:
                    # If loaded from DB, it's not a SortedDict yet.
                    # It will be populated by _hydrate_dynamic_caches_from_db
                    peer_data.setdefault(key, SortedDict())

    def _get_peer_stat_cache(self, peer_id: str, stat_name: str) -> Union[SortedDict, dict, None]:
        """(World-only) Helper to get or create the cache structure for a peer stat on demand."""
        if not self.is_world:
            return
        
        peer_cache = self._stats[self.GROUP_KEY].setdefault(peer_id, {})
        if stat_name not in peer_cache:
            if stat_name in self.all_dynamic_keys:
                peer_cache[stat_name] = SortedDict()
            elif stat_name in self.all_static_keys:
                peer_cache[stat_name] = self._stat_defaults[stat_name]
        
        return peer_cache.get(stat_name)

    # --- SHARED API ---
    def store_stat(self, stat_name: str, value: Any, peer_id: str, timestamp: int):
        """Unified API to store a stat. It then calls private methods to
        differentiate between static and dynamic stats.
        """
        if stat_name not in self.all_keys:
            self._err(f'Stat "{stat_name}" is not defined.')
        
        # disambiguate between static and dynamic stats
        if stat_name in self.all_static_keys:
            self._store_static(stat_name, value, peer_id, timestamp)
        else:
            self._store_dynamic(stat_name, value, peer_id, timestamp)
    
    def _validate_type(self, stat_name, value):
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
                self._err(f'Type mismatch for {stat_name}: '
                          f'Expected {schema_type} but got {type(value)}. '
                          f'Value: "{value}". Storing as string.')
                return str(value)  # Fallback

    def _make_json_serializable(self, value: Any) -> Any:
        """Recursively converts non-serializable types (like sets) to lists."""
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

    def _store_static(self, stat_name: str, value: Any, peer_id: str, timestamp: int):
        """
        Unified API to store a static (single-value) stat.
        - On Agent: Adds to the network send buffer.
        - On World: Updates the hot cache and adds to the DB buffer.
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
                peer_cache = self._stats[self.GROUP_KEY].setdefault(peer_id, {})
                peer_cache[stat_name] = value
            
            # 2. Add to DB buffer (key, value_json)
            serializable_value = self._make_json_serializable(value)
            self._static_db_buffer.append((peer_id, stat_name, json.dumps(serializable_value), timestamp))
        else:
            # --- AGENT LOGIC ---
            # De-duplicate logic: remove previous static value for this peer/stat
            self._update_batch = [u for u in self._update_batch
                                  if not (u['peer_id'] == peer_id and u['stat_name'] == stat_name)]
            
            # 2. Add to batch
            self._update_batch.append({
                'peer_id': peer_id,
                'stat_name': stat_name,
                'timestamp': timestamp,
                'value': value
            })

    def _store_dynamic(self, stat_name: str, value: Any, peer_id: str, timestamp: float):
        """
        Unified API to store a dynamic (time-series) stat.
        - On Agent: Gets current time, adds to network send buffer.
        - On World: Uses provided timestamp, updates hot cache, adds to DB buffer.
        """
        value = self._validate_type(stat_name, value)
        if self.is_world:
            # --- WORLD LOGIC ---
            if timestamp > self.max_seen_timestamp:
                self.max_seen_timestamp = timestamp

            # 1. Update hot cache
            if stat_name in self.world_ungrouped_keys:
                cache = self._stats.get(stat_name)
            else:
                cache = self._get_peer_stat_cache(peer_id, stat_name)
            
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
            self._dynamic_db_buffer.append((timestamp, peer_id, stat_name, val_num, val_str, val_json))
        else:
            # --- AGENT LOGIC ---
            self._update_batch.append({
                'peer_id': peer_id,
                'stat_name': stat_name,
                'timestamp': timestamp,
                'value': value
            })
    
    # --- AGENT API ---
    def update_view(self, view_data: Dict[str, Any] = None, overwrite: bool = False):
        """
        (Agent-side) Replaces the local view with data received from World.
        This is 'dumb' storage: we don't parse it, we just store it for plotting.
        
        The view has this structure:
        {
            "world": { "stat_name": value_or_timeseries },
            "peers": { "peer_id": { "stat_name": value_or_timeseries } }
        }
        For Dynamic stats, returns a list of lists: [[timestamp, value], ...] for efficient JSON/Plotly usage.
        
        Args:
            view_data: The snapshot received from the world.
            overwrite: If True, replaces the entire current view instead of merging.
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
        
        def _merge_dict(target: Dict, source: Dict):
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
    
    def _get_last_val_from_view(self, view: Dict, name: str) -> str:
        """Helper to extract a scalar value safely from the view snapshot.
        View structure:
                {
                "world": { "stat_name": value_or_timeseries },
                "peers": { "peer_id": { "stat_name": value_or_timeseries } }
                }
            For Dynamic stats we have a list of lists: [[timestamp, value], ...]"""
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

    def get_stats(self):
        return self._stats

    def get_payload_for_world(self) -> List[Dict[str, Any]]:
        """(Agent-only) Gathers, returns, and clears all stats to be sent to the world."""
        if self.is_world:
            return []
        
        # self._update_agent_static()  # Ensure static stats are fresh in the batch
        payload = self._update_batch
        self._update_batch = []  # Clear after getting
        return payload

    # --- WORLD API ---
    def get_view(self, since_timestamp: int = 0) -> Dict[str, Any]:
        """
        (World-side) Returns a clean, JSON-serializable dictionary of the CURRENT in-memory cache.
        Used for initial handshake or lightweight polling.
        
        Structure returned:
        {
            "world": { "stat_name": value_or_timeseries },
            "peers": { "peer_id": { "stat_name": value_or_timeseries } }
        }
        For Dynamic stats, returns a list of lists: [[timestamp, value], ...] for efficient JSON/Plotly usage.
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
        """Helper to convert SortedDicts to [[t, v], ...] and clean other types."""
        if isinstance(value, SortedDict):
            idx = value.bisect_left(since_timestamp)
            sliced_items = value.items()[idx:]
            # Convert to list of [timestamp, value] for Plotly readiness
            return [[k, self._make_json_serializable(v)] for k, v in sliced_items]
        else:
            # Static value: return as is (assuming it's serializable)
            return self._make_json_serializable(value)
    
    def get_last_value(self, stat_name: str, peer_id: str | None = None) -> Any | None:
        """Public API to get the most recent value of any stat, whether static or dynamic.
        - If peer_id is None, it searches for an ungrouped (world-level) stat.
        - If peer_id is provided, it searches for a grouped stat for that peer.
        Returns the last value, or None if not found.
        """
        if stat_name in self.all_static_keys:
            return self._get_last_static_value(stat_name, peer_id)
        elif stat_name in self.all_dynamic_keys:
            return self._get_last_dynamic_value(stat_name, peer_id)
        else:
            self._err(f'get_last_value: Unknown stat_name "{stat_name}"')
            return None
        
    def _get_last_dynamic_value(self, stat_name: str, peer_id: str | None = None) -> Any | None:
        """
        Returns the most recent value of a dynamic stat from the hot cache.
        - If peer_id is None, it searches for an ungrouped (world-level) stat.
        - If peer_id is provided, it searches for a grouped stat for that peer.
        Returns None if the stat is not found or has no entries.
        """
        if not self.is_world:
            return None  # Agents don't have this cache
            
        cache: Optional[SortedDict] = None
        
        if peer_id is None:
            # --- This is an ungrouped (world) stat ---
            if stat_name in self.world_ungrouped_keys:
                cache = self._stats.get(stat_name)
        else:
            # --- This is a grouped (peer) stat ---
            if stat_name in self.world_grouped_keys:
                peer_cache = self._stats.get(self.GROUP_KEY, {}).get(peer_id)
                if peer_cache:
                    cache = peer_cache.get(stat_name)
        
        # Check if we found a valid SortedDict cache and it's not empty
        if isinstance(cache, SortedDict) and cache:
            return cache.peekitem(-1)[1]  # Return the last value
        
        return None  # Stat not found or no values
    
    def _get_last_static_value(self, stat_name: str, peer_id: str | None = None) -> Any | None:
        """
        Returns the current value of a static stat from the hot cache.
        - If peer_id is None, it searches for an ungrouped (world-level) stat.
        - If peer_id is provided, it searches for a grouped stat for that peer.
        Returns None if the stat is not found.
        """
        if not self.is_world:
            return None  # Agents don't have this cache
        
        value: Any | None = None
        if peer_id is None:
            # --- This is an ungrouped (world) stat ---
            if stat_name in self.world_ungrouped_keys:
                value = self._stats.get(stat_name)
        else:
            # --- This is a grouped (peer) stat ---
            if stat_name in self.world_grouped_keys:
                peer_cache = self._stats.get(self.GROUP_KEY, {}).get(peer_id)
                if peer_cache:
                    value = peer_cache.get(stat_name)
        return value
    
    # --- WORLD API (PERSISTENCE) ---
    def save_to_disk(self):
        """(World-only) Saves the static snapshot and dynamic buffer to SQLite."""
        if not self.is_world or not self._db_conn:
            return
        self._deb(f'Saving world stats to DB...')
        try:
            self._save_static_to_db()
            self._save_dynamic_to_db()
            self._prune_cache()
            self._prune_db()
            
            self._db_conn.commit()
            self._deb(f'Save complete.')
        except Exception as e:
            self._err(f'CRITICAL: Save_to_disk failed: {e}')
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
        
        self._deb(f'Wrote {len(self._dynamic_db_buffer)} dynamic stats to SQLite.')
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
            self._deb(f'Pruned stale peer {peer_id} from cache.')

    # --- WORLD API (LOADING) ---
    def _load_existing_stats(self):
        """(World-only) Loads existing stats from disk to hydrate the cache."""
        if not self.is_world or not self._db_conn:
            return
        self._deb('Loading existing stats from disk...')
        self._load_static_from_db()
        self._hydrate_dynamic_caches_from_db()
        self._deb('Finished loading stats.')

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
                self._deb('No dynamic stats found in DB. Hydration skipped.')
                return  # No data in DB, nothing to load
            self.max_seen_timestamp = int(max_ts_result[0])
            cutoff_t_ms = self.max_seen_timestamp - int(self.min_window_duration.total_seconds() * 1000)

            cursor = self._db_conn.execute("""
                SELECT timestamp, peer_id, stat_name, val_num, val_str, val_json 
                FROM dynamic_stats 
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            """, (cutoff_t_ms,))
            
            count = 0
            for ts, peer_id, stat_name, _, _, val_json in cursor:
                ts = int(ts)
                # we just need the val_json that will be cast to the exact type by _validate_type
                value = json.loads(val_json)
                self._store_dynamic(stat_name, value, peer_id, ts)
                count += 1
            
            # Clear the buffer generated by hydrating
            self._dynamic_db_buffer = []
            
            if count > 0:
                self._deb(f'Hydrated cache with {count} recent dynamic stats.')
            else:
                self._deb('No recent dynamic stats found in DB.')

        except Exception as e:
            self._err(f'Failed to hydrate dynamic caches from DB: {e}')

    # --- WORLD API (QUERYING) ---
    def query_history(self,
                      stat_names: List[str] = [],
                      peer_ids: List[str] = [],
                      time_range: Union[Tuple[int, int], int, None] = None,
                      value_range: Tuple[float, float] | None = None,
                      limit: int | None = None) -> Dict[str, Any]:
        """
        (World-only) Queries the SQLite DB for specific stats, potentially filtering by VALUE.
        Returns the same structure as get_view(), allowing the agent to ingest it seamlessly.
        Automatically flushes the current memory buffer to DB before querying
        to ensure "read-your-writes" consistency.
        
        Args:
            value_range: (min, max) - Only returns rows where val_num is within range.
        """
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
            query_static.append(f"stat_name IN ({','.join(['?']*len(stat_names))})")
            params_static.extend(stat_names)
        if peer_ids:
            if not where_added:
                query_static.append("WHERE")
            else:
                query_static.append(f"AND")
            query_static.append(f"peer_id IN ({','.join(['?']*len(peer_ids))})")
            params_static.extend(peer_ids)

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
            self._err(f'Query history (static) failed: {e}')
        
        # B. Query the dynamic stats
        query_dyn = ['SELECT timestamp, peer_id, stat_name, val_num, val_str, val_json FROM dynamic_stats']
        params_dyn = []

        # 1. Stat Names
        where_added = False
        if stat_names:
            query_static.append("WHERE")
            where_added = True
            query_dyn.append(f"stat_name IN ({','.join(['?']*len(stat_names))})")
            params_dyn.extend(stat_names)
        
        # 2. Peer IDs
        if peer_ids:
            if not where_added:
                query_static.append("WHERE")
            else:
                query_static.append(f"AND")
            query_dyn.append(f"peer_id IN ({','.join(['?']*len(peer_ids))})")
            params_dyn.extend(peer_ids)
            
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
            self._err(f'Query history failed: {e}')
            
        return snapshot

    def _aggregate_time_indexed_stats_over_peers(self, stats: dict) -> tuple[dict, dict]:
        """(World-only) Aggregates time-indexed peer stats (mean/std) from CACHE."""
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
                series_times = series.keys()
                series_vals = series.values()
                
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
    
    def shutdown(self):
        """Call this explicitly when your application is closing."""
        if self.is_world and self._db_conn:
            self._deb('Shutdown: Saving final stats...')
            try:
                self.save_to_disk()
            except Exception as e:
                self._err(f'Shutdown save failed: {e}')
            self._db_conn.close()
            self._db_conn = None
            self._deb('SQLite connection closed.')
    
    def __del__(self):
        if self.is_world and self._db_conn:
            try:
                # Final save on exit, if any buffer
                self.save_to_disk()
            except Exception:
                pass  # Don't raise in destructor
            self._db_conn.close()
            self._deb('SQLite connection closed.')
    
    # --- PLOTTING INTERFACE ---
    def plot(self, since_timestamp: int = 0) -> str | None:
        """
        Default dashboard implementation. 
        Visualizes Core Stats: Topology, Agent Counts, States, and Actions.
        """
        # 1. Get Data view
        view = self.get_view(since_timestamp) if self.is_world else self._world_view
        if not view:
            return None
            
        dash = DefaultBaseDash("World Overview")

        # --- Panel 1: Network Topology (Top Left) ---
        p1 = UIPlot(title="World Topology")
        self._populate_graph(p1, view, "graph")
        # p1.set_layout_opt('xaxis', {'visible': False})
        # p1.set_layout_opt('yaxis', {'visible': False})
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
        # p2.set_layout_opt('xaxis', {'title': None, 'visible': False})
        p2.set_layout_opt('xaxis', {'title': None, 'showticklabels': False}) 
        p2.set_layout_opt('yaxis', {'title': None})
        dash.add_panel(p2, "top_right")

        # --- Panel 3: State Distribution (Bar) ---
        p3 = UIPlot(title="State Distribution")
        # self._populate_distribution(p3, view, "state")
        self._populate_graph(p3, view, "graph")
        # p1.set_layout_opt('xaxis', {'visible': False})
        # p1.set_layout_opt('yaxis', {'visible': False})
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

        return dash.to_json()
    
    def _populate_time_series(self, panel: UIPlot, view: Dict, stat_name: str,
                              peer_ids: List[str] | None = None, color_override: str = None,
                              show_legend: bool = True, title_override: str = None):
        """Extracts [[t,v],...] lists and adds lines to panel. Supports custom titles and colors."""
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
    
    def _populate_indicator(self, panel: UIPlot, view: Dict, stat_name: str, peer_ids: List[str] | None = None):
        """Extracts a scalar value and adds indicator."""
        val = None
        if 'world' in view and stat_name in view['world']:
            val = view['world'][stat_name]
        elif 'peers' in view:
            # Just grab the first available peer's value if not specified
            targets = peer_ids if peer_ids else list(view['peers'].keys())
            if targets:
                val = view['peers'][targets[0]].get(stat_name)

        panel.add_indicator(val, title=stat_name)

    def _populate_table(self, panel: UIPlot, view: Dict, stat_name: str, peer_ids: List[str] | None = None):
        """Extracts data for a table."""
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
    
    def _populate_graph(self, panel: UIPlot, view: Dict, stat_name: str):
        """Calculates layout and adds graph traces to the panel."""
        
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
    
    def _populate_distribution(self, panel: UIPlot, view: Dict, stat_name: str):
        """
        Aggregates peer values into a frequency count (Bar Chart).
        e.g., {"IDLE": 3, "RUNNING": 5}
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
        """Deterministic color generation based on string hash."""
        if not unique_str:
            return '#ffffff'
        idx = zlib.adler32(str(unique_str).encode()) % len(THEME['peers'])
        return THEME['peers'][idx]
