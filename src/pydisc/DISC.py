
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# import ruptures as rpt
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
import torch
# from kneed import KneeLocator
import time
import zarr

from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QSplitter, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QLineEdit, QSpinBox, QSizePolicy, QComboBox, QCheckBox, QToolBar, QToolButton, QMenu, QWidgetAction, QProgressDialog, QApplication, QPushButton, QFileDialog, QAction
import pyqtgraph as pg
import qtawesome as qta


class DISC_Sequence:

    def __init__(self, data: np.ndarray):

        # the time series trace
        self.data: np.ndarray = data

        # metadata tags
        self.tags: str = ''

        # for comparison to known simulated noiseless data
        self.noiseless_data: np.ndarray = None

        # parameters
        self.div_criterion = "BIC"
        self.agg_criterion = "BIC"
        self.n_required_levels: int = None
        self.level_func = np.median
        self.n_div_attempts: int = None
        self.hmm_algorithm: str = 'viterbi' # 'viterbi' or 'baum-welch'
        self.hmm_optimize_states: bool = True
        self.n_viterbi_repeats: int = 2
        self.hmm_scan: bool = False
        self.final_baum_welch_optimization: bool = False

        # idealization
        self.idealized_data: np.ndarray = None
        self.idealized_metric: float = None
        
        # intermediate results
        self.intermediate_results: dict = None
    
    def run(self, auto: bool = False, return_intermediate_results: bool = False, verbose: bool = False):
        func = auto_DISC if auto else run_DISC
        results = func(
            self.data,
            div_criterion = self.div_criterion,
            agg_criterion = self.agg_criterion,
            n_required_levels = self.n_required_levels,
            level_func = self.level_func,
            n_div_attempts = self.n_div_attempts,
            hmm_algorithm = self.hmm_algorithm,
            hmm_optimize_states = self.hmm_optimize_states,
            n_viterbi_repeats = self.n_viterbi_repeats,
            hmm_scan = self.hmm_scan,
            final_baum_welch_optimization = self.final_baum_welch_optimization,
            return_intermediate_results = return_intermediate_results,
            verbose = verbose
            )
        if auto:
            results, criterion = results
            self.div_criterion = criterion
            self.agg_criterion = criterion
        if return_intermediate_results:
            self.idealized_data, self.idealized_metric, self.intermediate_results = results
        else:
            self.idealized_data, self.idealized_metric = results
        self.n_idealized_levels = len(np.unique(self.idealized_data))
    
    def add_level(self, verbose: bool = False):
        if self.idealized_data is None:
            return
        if self.n_required_levels is not None:
            self.n_required_levels += 1
        else:
            try:
                agg_index = self.intermediate_results['agg_index']
                self.n_required_levels = agg_index + 2
            except:
                self.n_idealized_levels = len(np.unique(self.idealized_data))
                self.n_required_levels = self.n_idealized_levels + 1
        self.run(auto=False, return_intermediate_results=False, verbose=verbose)
    
    def remove_level(self, verbose: bool = False):
        if self.idealized_data is None:
            return
        if self.n_required_levels is not None:
            n_required_levels = self.n_required_levels - 1
        else:
            try:
                agg_index = self.intermediate_results['agg_index']
                n_required_levels = agg_index
                if n_required_levels < 1:
                    self.n_idealized_levels = len(np.unique(self.idealized_data))
                    n_required_levels = self.n_idealized_levels - 1
            except:
                self.n_idealized_levels = len(np.unique(self.idealized_data))
                n_required_levels = self.n_idealized_levels - 1
        if n_required_levels < 1:
            return
        self.n_required_levels = n_required_levels
        self.run(auto=False, return_intermediate_results=False, verbose=verbose)
    
    def merge_nearest_levels(self):
        if self.idealized_data is None:
            return
        self.n_idealized_levels = len(np.unique(self.idealized_data))
        if self.n_idealized_levels < 2:
            return
        merge_nearest_levels(self.data, self.idealized_data)
        self.n_idealized_levels = len(np.unique(self.idealized_data))
        self.idealized_metric = information_criterion(self.data, self.idealized_data, self.agg_criterion)
    
    def baum_welch_optimization(self, level_means: np.ndarray = None, level_stdevs: np.ndarray = None, optimize_level_means=False, optimize_level_stdevs=False):
        self.idealized_data, hmm = hmm_idealization_refinement(self.data, self.idealized_data, level_means, level_stdevs, optimize_level_means=optimize_level_means, optimize_level_stdevs=optimize_level_stdevs, algorithm='baum-welch')
        if not isinstance(self.intermediate_results, dict):
            self.intermediate_results = {}
        self.intermediate_results['final_hmm'] = hmm


class DISCO(QWidget):

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setWindowTitle("DISC")

        self._traces: list[DISC_Sequence] = []

        # plots
        self._trace_plot = pg.PlotWidget(
            pen=pg.mkPen(QColor.fromRgbF(0.15, 0.15, 0.15), width=1)
        )
        for axis in ['left', 'bottom', 'right', 'top']:
            axis_item = self._trace_plot.getAxis(axis)
            if axis_item is not None:
                axis_item.setTextPen(QColor.fromRgbF(0.15, 0.15, 0.15))
        self._trace_plot.setBackground(QColor(240, 240, 240))

        self._criterion_plot = pg.PlotWidget(
            pen=pg.mkPen(QColor.fromRgbF(0.15, 0.15, 0.15), width=1)
        )
        for axis in ['left', 'bottom', 'right', 'top']:
            axis_item = self._criterion_plot.getAxis(axis)
            if axis_item is not None:
                axis_item.setTextPen(QColor.fromRgbF(0.15, 0.15, 0.15))
        self._criterion_plot.setBackground(QColor(240, 240, 240))
        self._criterion_plot.getAxis('left').setLabel('Criterion')
        self._criterion_plot.getAxis('bottom').setLabel('# Levels')
        self._criterion_plot.getAxis('bottom').setTickSpacing(5, 1)
        
        # trace iteration
        self._trace_selector = QSpinBox()
        self._trace_selector.setRange(0, 0)
        self._trace_selector.setPrefix("Trace ")
        self._trace_selector.setSuffix(" of 0")
        self._trace_selector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._trace_selector.valueChanged.connect(lambda value: self.set_trace(value - 1))
        self._last_trace_index = None

        # buttons
        self._load_data_button = QToolButton()
        self._load_data_button.setIcon(qta.icon('fa.folder-open', opacity=0.75))
        self._load_data_button.setToolTip("Load data from Zarr store")
        self._load_data_button.pressed.connect(self.load_zarr)

        self._save_data_button = QToolButton()
        self._save_data_button.setIcon(qta.icon('fa.save', opacity=0.75))
        self._save_data_button.setToolTip("Save data to Zarr store")
        self._save_data_button.pressed.connect(self.save_zarr)

        self._simulation_button = QToolButton()
        self._simulation_button.setIcon(qta.icon('mdi.waveform', opacity=0.75))
        self._simulation_button.setToolTip("Simulate data")
        self._simulation_button.pressed.connect(self.simulate_data)

        self._idealize_button = QToolButton()
        self._idealize_button.setIcon(qta.icon('msc.run'))
        self._idealize_button.setToolTip("Run DISC on selected trace")
        self._idealize_button.pressed.connect(self._idealize_selected_trace)

        self._idealize_all_button = QToolButton()
        self._idealize_all_button.setIcon(qta.icon('msc.run-all'))
        self._idealize_all_button.setToolTip("Run DISC on all traces")
        self._idealize_all_button.pressed.connect(self._idealize_all_traces)

        self._add_level_button = QToolButton()
        self._add_level_button.setIcon(qta.icon('mdi.stairs-up', opacity=0.75))
        self._add_level_button.setToolTip("Add level")
        self._add_level_button.pressed.connect(self.add_level)

        self._remove_level_button = QToolButton()
        self._remove_level_button.setIcon(qta.icon('mdi.stairs-down', opacity=0.75))
        self._remove_level_button.setToolTip("Remove level")
        self._remove_level_button.pressed.connect(self.remove_level)

        self._merge_nearest_levels_button = QToolButton()
        self._merge_nearest_levels_button.setIcon(qta.icon('mdi.merge', opacity=0.75))
        self._merge_nearest_levels_button.setToolTip("Merge nearest levels")
        self._merge_nearest_levels_button.pressed.connect(self.merge_nearest_levels)

        self._hmm_optimization_button = QToolButton()
        self._hmm_optimization_button.setText("HMM")
        self._hmm_optimization_button.setToolTip("HMM Baum-Welch optimization of idealization")
        self._hmm_optimization_button.pressed.connect(self.baum_welch_optimization)

        # DISC controls
        self._num_levels_edit = QLineEdit()
        self._num_levels_edit.setToolTip("Number of idealized levels")
        self._num_levels_edit.setPlaceholderText("auto")
        self._num_levels_edit.textEdited.connect(self._on_num_levels_changed)

        self._div_criterion_selector = QComboBox()
        self._div_criterion_selector.addItems(["RSS", "AIC", "BIC", "HQC"])
        self._div_criterion_selector.setCurrentText("BIC")
        self._div_criterion_selector.setToolTip("Divisive criterion")

        self._agg_criterion_selector = QComboBox()
        self._agg_criterion_selector.addItems(["RSS", "AIC", "BIC", "HQC"])
        self._agg_criterion_selector.setCurrentText("BIC")
        self._agg_criterion_selector.setToolTip("Agglomerative criterion")

        self._auto_criterion_toggle = QCheckBox()
        self._auto_criterion_toggle.setChecked(False)
        self._auto_criterion_toggle.setToolTip("Auto-select criterion (see Bandyopadhyay and Goldschen-Ohm, 2021)")
        self._auto_criterion_toggle.stateChanged.connect(self._on_auto_disc_toggled)

        self._hmm_algorithm_selector = QComboBox()
        self._hmm_algorithm_selector.addItems(["Viterbi", "Baum-Welch"])
        self._hmm_algorithm_selector.setCurrentText("Viterbi")
        self._hmm_algorithm_selector.setToolTip("HMM algorithm (Viterbi: faster, no level optim; Baum-Welch: slower, level optim)")
        self._hmm_algorithm_selector.currentTextChanged.connect(self._on_hmm_algorithm_changed)

        self._num_viterbi_repeats = QLineEdit("2")
        self._num_viterbi_repeats.setToolTip("Number of Viterbi repetitions (levels reoptimized each time)")
        self._num_viterbi_repeats.setPlaceholderText("2")

        self._hmm_scan_toggle = QCheckBox()
        self._hmm_scan_toggle.setChecked(False)
        self._hmm_scan_toggle.setToolTip("Apply HMM to multiple agglomerations. Slow, but can help find optimal levels.")

        self._final_baum_welch_optimization_toggle = QCheckBox()
        self._final_baum_welch_optimization_toggle.setChecked(False)
        self._final_baum_welch_optimization_toggle.setToolTip("Use Baum-Welch to optimize final HMM levels.")

        self._verbose_toggle = QCheckBox()
        self._verbose_toggle.setChecked(False)
        self._verbose_toggle.setToolTip("Print messages to stdout.")

        self._fastest_settings_button = QPushButton("Faster")
        self._fastest_settings_button.pressed.connect(self._set_to_fastest_settings)

        self._most_accurate_settings_button = QPushButton("Most Accurate")
        self._most_accurate_settings_button.pressed.connect(self._set_to_most_accurate_settings)

        self._on_auto_disc_toggled()
        self._on_num_levels_changed()
        self._on_hmm_algorithm_changed()

        self._disc_controls = QWidget()
        form = QFormLayout(self._disc_controls)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow("# Levels", self._num_levels_edit)
        form.addRow("Divisive Criterion", self._div_criterion_selector)
        form.addRow("Agglomerative Criterion", self._agg_criterion_selector)
        form.addRow("AutoDISC", self._auto_criterion_toggle)
        form.addRow("HMM algorithm", self._hmm_algorithm_selector)
        form.addRow("# Viterbi Repeats", self._num_viterbi_repeats)
        form.addRow("HMM scan", self._hmm_scan_toggle)
        form.addRow("Final Baum-Welch optimization", self._final_baum_welch_optimization_toggle)
        form.addRow("Verbose", self._verbose_toggle)
        tmp = QWidget()
        hbox = self._side_by_side_button_layout(self._fastest_settings_button, self._most_accurate_settings_button)
        tmp.setLayout(hbox)
        form.addRow(tmp)

        self._disc_contols_menu = QMenu()
        action = QWidgetAction(self._disc_contols_menu)
        action.setDefaultWidget(self._disc_controls)
        self._disc_contols_menu.addAction(action)
        self._idealize_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._idealize_button.setMenu(self._disc_contols_menu)

        # simulation controls
        self.sim_n_traces_edit = QLineEdit("1")
        self.sim_n_traces_edit.setToolTip("Number of traces")
        self.sim_n_traces_edit.setPlaceholderText("1")

        self.sim_n_samples_edit = QLineEdit()
        self.sim_n_samples_edit.setToolTip("Number of samples per trace")
        self.sim_n_samples_edit.setPlaceholderText("random")

        self.sim_levels_edit = QLineEdit()
        self.sim_levels_edit.setToolTip("Array of level means (e.g., [0, 1, 2])")
        self.sim_levels_edit.setPlaceholderText("random")

        self.sim_noise_edit = QLineEdit()
        self.sim_noise_edit.setToolTip("Array of level stdevs (e.g., [0.5, 0.5, 0.5])")
        self.sim_noise_edit.setPlaceholderText("random")

        self.sim_trans_edit = QLineEdit()
        self.sim_trans_edit.setToolTip("2-D array of transition probabilities between each level (e.g., [[0.9, 0.1], [0.1, 0.9]])")
        self.sim_trans_edit.setPlaceholderText("random")

        self._simulation_controls = QWidget()
        form = QFormLayout(self._simulation_controls)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow("# Traces", self.sim_n_traces_edit)
        form.addRow("# Samples", self.sim_n_samples_edit)
        form.addRow("Level Means", self.sim_levels_edit)
        form.addRow("Level Stdevs", self.sim_noise_edit)
        form.addRow("Transitions Proba", self.sim_trans_edit)

        self._simulation_contols_menu = QMenu()
        action = QWidgetAction(self._simulation_contols_menu)
        action.setDefaultWidget(self._simulation_controls)
        self._simulation_contols_menu.addAction(action)
        self._simulation_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._simulation_button.setMenu(self._simulation_contols_menu)

        # HMM controls
        self.hmm_levels_edit = QLineEdit()
        self.hmm_levels_edit.setToolTip("Array of level means (e.g., [0, 1, 2])")
        self.hmm_levels_edit.setPlaceholderText("from idealization")

        self.hmm_noise_edit = QLineEdit()
        self.hmm_noise_edit.setToolTip("Array of level stdevs (e.g., [0.5, 0.5, 0.5])")
        self.hmm_noise_edit.setPlaceholderText("auto")

        self._hmm_fix_means_toggle = QCheckBox()
        self._hmm_fix_means_toggle.setChecked(False)
        self._hmm_fix_means_toggle.setToolTip("Fix level means")

        self._hmm_fix_stdevs_toggle = QCheckBox()
        self._hmm_fix_stdevs_toggle.setChecked(False)
        self._hmm_fix_stdevs_toggle.setToolTip("Fix level stdevs")

        self._hmm_controls = QWidget()
        form = QFormLayout(self._hmm_controls)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow("Level means", self.hmm_levels_edit)
        form.addRow("Level stdevs", self.hmm_noise_edit)
        form.addRow("Fix means", self._hmm_fix_means_toggle)
        form.addRow("Fix stdevs", self._hmm_fix_stdevs_toggle)

        self._hmm_contols_menu = QMenu()
        action = QWidgetAction(self._hmm_contols_menu)
        action.setDefaultWidget(self._hmm_controls)
        self._hmm_contols_menu.addAction(action)
        self._hmm_optimization_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._hmm_optimization_button.setMenu(self._hmm_contols_menu)

        # tags
        self._tags_icon_action = QAction()
        self._tags_icon_action.setIcon(qta.icon('fa.tag', opacity=0.75))
        self._tags_edit = QLineEdit()
        self._tags_edit.textEdited.connect(self._on_tags_edited)
        self._tags_filter_icon_action = QAction()
        self._tags_filter_icon_action.setIcon(qta.icon('mdi6.filter-multiple-outline', opacity=0.75))
        self._tags_filter_edit = QLineEdit()
        self._tags_filter_edit.textEdited.connect(self._on_tags_filter_edited)

        # layout
        self._toolbar = QToolBar(orientation=Qt.Orientation.Horizontal)
        self._toolbar.addWidget(self._load_data_button)
        self._toolbar.addWidget(self._save_data_button)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._simulation_button)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._trace_selector)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._idealize_button)
        self._toolbar.addWidget(self._idealize_all_button)
        self._toolbar.addSeparator()
        self._toolbar.addWidget(self._add_level_button)
        self._toolbar.addWidget(self._remove_level_button)
        self._toolbar.addWidget(self._merge_nearest_levels_button)
        self._toolbar.addWidget(self._hmm_optimization_button)
        self._toolbar.addAction(self._tags_icon_action)
        self._toolbar.addWidget(self._tags_edit)
        self._toolbar.addAction(self._tags_filter_icon_action)
        self._toolbar.addWidget(self._tags_filter_edit)
        self._toolbar.setIconSize(QSize(24, 24))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._trace_plot)
        splitter.addWidget(self._criterion_plot)

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self._toolbar)
        vbox.addWidget(splitter)
    
    @property
    def data(self):
        return self._traces
    
    @data.setter
    def data(self, traces):
        for i, trace in enumerate(traces):
            if not isinstance(trace, DISC_Sequence):
                if isinstance(trace, np.ndarray):
                    traces[i] = DISC_Sequence(trace)
                elif isinstance(trace, list):
                    traces[i] = DISC_Sequence(np.array(trace))
                else:
                    raise ValueError(f"traces[{i}] is not a DISC_Sequence, numpy array, or list.")
        self._traces = traces
        self._trace_selector.setRange(1, len(traces))
        self._trace_selector.setValue(1)
        self._trace_selector.setSuffix(f" of {len(traces)}")
        self.replot()
    
    def _side_by_side_button_layout(self, *buttons):
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        for button in buttons:
            hbox.addWidget(button)
        return hbox
    
    def load_zarr(self, filepath=''):
        if filepath == '':
            filepath = QFileDialog.getExistingDirectory(self, 'Open from Zarr store...')
            if filepath == '':
                return
        traces = []
        root = zarr.open_group(filepath, mode='r')
        for groupname, group in root.groups():
            trace = DISC_Sequence(None)
            for arrayname, array in group.arrays():
                setattr(trace, arrayname, array[:])
            if 'tags' in group.attrs:
                trace.tags = group.attrs['tags']
            traces.append(trace)
        self.data = traces

    def save_zarr(self, filepath=''):
        if filepath == '':
            filepath, _ = QFileDialog.getSaveFileName(self, 'Save to Zarr store...')
            if filepath == '':
                return
        root = zarr.open_group(filepath, mode='w')
        for i, trace in enumerate(self._traces):
            group = root.create_group(f"trace.{i}")
            group.create_dataset('data', data=trace.data)
            if trace.idealized_data is not None:
                group.create_dataset('idealized_data', data=trace.idealized_data)
            if trace.noiseless_data is not None:
                group.create_dataset('noiseless_data', data=trace.noiseless_data)
            if trace.tags:
                group.attrs['tags'] = trace.tags

    def simulate_data(self):
        try:
            n_traces = int(self.sim_n_traces_edit.text())
        except:
            n_traces = 1
        
        try:
            n_pts = int(self.sim_n_samples_edit.text())
        except:
            n_pts = None
        
        try:
            levels = self.sim_levels_edit.text().lstrip("[").rstrip("]")
            levels = np.array([float(x) for x in levels.split(",")])
        except:
            levels = None

        try:
            sigmas = self.sim_noise_edit.text().lstrip("[").rstrip("]")
            sigmas = np.array([float(x) for x in sigmas.split(",")])
        except:
            sigmas = None
        
        try:
            rows = self.sim_trans_edit.text().split(",")
            trans_proba = []
            for row in rows:
                trans_proba.append([float(x) for x in row.lstrip("[").rstrip("]").split(",")])
            trans_proba = np.array(trans_proba)
        except:
            trans_proba = None

        self._traces = []
        for _ in range(n_traces):
            data, noiseless, params = simulate_trace(n_pts, levels, sigmas, trans_proba)
            trace = DISC_Sequence(data)
            trace.noiseless_data = noiseless
            self._traces.append(trace)
        
        self._trace_selector.setRange(1, n_traces)
        self._trace_selector.setValue(1)
        self._trace_selector.setSuffix(f" of {n_traces}")
        
        self.replot()
    
    def idealize_data(self, trace_indices: list[int] | str = "selected"):
        if not self._traces:
            return
        if trace_indices == "selected":
            traces = [self._traces[self._trace_selector.value() - 1]]
        elif trace_indices == "all":
            traces = self._traces
        else:
            traces = [self._traces[i] for i in trace_indices]
        n_traces = len(traces)

        progress = QProgressDialog(f"Idealizing {n_traces} trace(s)...", None, 0, n_traces, self)
        progress.show()
        QApplication.processEvents()

        verbose = self._verbose_toggle.isChecked()
        for i, trace in enumerate(traces):
            if progress.wasCanceled():
                break
            if i > 0:
                progress.setValue(i)
                QApplication.processEvents()
        
            try:
                trace.n_required_levels = int(self._num_levels_edit.text())
            except:
                trace.n_required_levels = None
            trace.hmm_algorithm = self._hmm_algorithm_selector.currentText().lower()
            try:
                trace.n_viterbi_repeats = int(self._num_viterbi_repeats.text())
            except:
                trace.n_viterbi_repeats = 2
            trace.hmm_scan = self._hmm_scan_toggle.isChecked()
            trace.final_baum_welch_optimization = self._final_baum_welch_optimization_toggle.isChecked()
            auto = self._auto_criterion_toggle.isChecked()
            if not auto:
                trace.div_criterion = self._div_criterion_selector.currentText()
                trace.agg_criterion = self._agg_criterion_selector.currentText()
            
            trace.run(auto=auto, return_intermediate_results=True, verbose=verbose)
        
            # only keep some of the intermediate results
            trace.intermediate_results = {
                key: trace.intermediate_results[key]
                for key in ['agg_levels', 'agg_metrics', 'agg_index', 'hmm_levels', 'hmm_metrics', 'hmm_index', 'final_hmm']
            }
        
        self.replot()
        progress.close()
    
    def _idealize_selected_trace(self):
        self.idealize_data("selected")
    
    def _idealize_all_traces(self):
        self.idealize_data("all")
    
    def add_level(self):
        if not self._traces:
            return
        trace = self._traces[self._trace_selector.value() - 1]
        if trace.idealized_data is None:
            return

        progress = QProgressDialog("Adding level...", None, 0, 0, self)
        progress.show()
        QApplication.processEvents()

        verbose = self._verbose_toggle.isChecked()
        trace.add_level(verbose=verbose)
        self.replot()

        progress.close()
    
    def remove_level(self):
        if not self._traces:
            return
        trace = self._traces[self._trace_selector.value() - 1]
        if trace.idealized_data is None or trace.n_idealized_levels < 2:
            return

        progress = QProgressDialog("Removing level...", None, 0, 0, self)
        progress.show()
        QApplication.processEvents()

        verbose = self._verbose_toggle.isChecked()
        trace.remove_level(verbose=verbose)
        self.replot()

        progress.close()
    
    def merge_nearest_levels(self):
        if not self._traces:
            return
        trace = self._traces[self._trace_selector.value() - 1]
        if trace.idealized_data is None:
            return
        
        trace.merge_nearest_levels()
        self.replot()
    
    def baum_welch_optimization(self):
        if not self._traces:
            return
        trace = self._traces[self._trace_selector.value() - 1]
        
        try:
            level_means = self.hmm_levels_edit.text().lstrip("[").rstrip("]")
            level_means = np.array([float(x) for x in level_means.split(",")])
        except:
            level_means = None
        
        try:
            level_stdevs = self.hmm_noise_edit.text().lstrip("[").rstrip("]")
            level_stdevs = np.array([float(x) for x in level_stdevs.split(",")])
        except:
            level_stdevs = None
        
        optimize_level_means = not self._hmm_fix_means_toggle.isChecked()
        optimize_level_stdevs = not self._hmm_fix_stdevs_toggle.isChecked()

        progress = QProgressDialog("Baum-Welch optimization...", None, 0, 0, self)
        progress.show()
        QApplication.processEvents()

        trace.baum_welch_optimization(level_means=level_means, level_stdevs=level_stdevs, optimize_level_means=optimize_level_means, optimize_level_stdevs=optimize_level_stdevs)

        self.replot()

        progress.close()
    
    def set_trace(self, trace_index: int):
        if not self._traces:
            return

        if self._is_tags_filter_enabled():
            n_traces = len(self._traces)
            filter_tags = [tag.strip() for tag in self._tags_filter_edit.text().split(",")]
            foundit = False
            if (self._last_trace_index is None) or (self._last_trace_index <= trace_index):
                # check forward
                for i in range(trace_index, n_traces):
                    trace_tags = [tag.strip() for tag in self._traces[i].tags.split(",")]
                    if any(tag in trace_tags for tag in filter_tags):
                        trace_index = i
                        foundit = True
                        break
                if not foundit:
                    # check backward
                    for i in range(trace_index, -1, -1):
                        trace_tags = [tag.strip() for tag in self._traces[i].tags.split(",")]
                        if any(tag in trace_tags for tag in filter_tags):
                            trace_index = i
                            foundit = True
                            break
            else:
                # check backward
                for i in range(trace_index, -1, -1):
                    trace_tags = [tag.strip() for tag in self._traces[i].tags.split(",")]
                    if any(tag in trace_tags for tag in filter_tags):
                        trace_index = i
                        foundit = True
                        break
                if not foundit:
                    # check forward
                    for i in range(trace_index, n_traces):
                        trace_tags = [tag.strip() for tag in self._traces[i].tags.split(",")]
                        if any(tag in trace_tags for tag in filter_tags):
                            trace_index = i
                            foundit = True
                            break
        
        self._last_trace_index = trace_index

        if self._trace_selector.value() != trace_index + 1:
            self._trace_selector.blockSignals(True)
            self._trace_selector.setValue(trace_index + 1)
            self._trace_selector.blockSignals(False)
        
        self.replot()
    
    def replot(self):
        self._trace_plot.clear()
        self._criterion_plot.clear()
        if not self._traces:
            return
        
        trace_index = self._trace_selector.value() - 1
        trace = self._traces[trace_index]
        
        if trace.data is not None:
            self._trace_plot.plot(trace.data, pen=pg.mkPen(QColor.fromRgb(0, 114, 189), width=1))
        
        # # for debugging
        # if isinstance(trace.intermediate_results, dict):
        #     div_idealized_data = trace.intermediate_results.get('div_idealized_data', None)
        #     if div_idealized_data is not None:
        #         self._trace_plot.plot(div_idealized_data, pen=pg.mkPen(QColor.fromRgb(0, 155, 255), width=1))
        
        if trace.idealized_data is not None:
            self._trace_plot.plot(trace.idealized_data, pen=pg.mkPen(QColor.fromRgb(217,  83,  25), width=2))
        
        if isinstance(trace.intermediate_results, dict):
            agg_levels = trace.intermediate_results.get('agg_levels', None)
            agg_metrics = trace.intermediate_results.get('agg_metrics', None)
            if agg_metrics is not None:
                color = QColor.fromRgb(0, 114, 189)
                self._criterion_plot.plot(agg_levels, agg_metrics, pen=pg.mkPen(color, width=1), symbol='o', symbolPen=pg.mkPen(color, width=1), symbolBrush=color)
                self._criterion_plot.getAxis('left').setLabel(trace.agg_criterion)
            else:
                self._criterion_plot.getAxis('left').setLabel('Criterion')
            
            hmm_levels = trace.intermediate_results.get('hmm_levels', None)
            hmm_metrics = trace.intermediate_results.get('hmm_metrics', None)
            if hmm_metrics is not None:
                color = QColor.fromRgb(0,  255,  255)
                self._criterion_plot.plot(hmm_levels, hmm_metrics, pen=pg.mkPen(color, width=1), symbol='o', symbolPen=pg.mkPen(color, width=1), symbolBrush=color)
        else:
            self._criterion_plot.getAxis('left').setLabel('Criterion')
        
        if (trace.idealized_data is not None) and (trace.idealized_metric is not None):
            n_idealized_levels = len(np.unique(trace.idealized_data))
            color = QColor.fromRgb(217,  83,  25)
            self._criterion_plot.plot([n_idealized_levels], [trace.idealized_metric], pen=pg.mkPen(color, width=1), symbol='o', symbolPen=pg.mkPen(color, width=1), symbolBrush=color)
        
        # self._criterion_plot.autoRange()

        self._tags_edit.setText(trace.tags)
    
    def _set_to_fastest_settings(self):
        self._auto_criterion_toggle.setChecked(False)
        self._hmm_algorithm_selector.setCurrentText("Viterbi")
        self._hmm_scan_toggle.setChecked(False)
        self._final_baum_welch_optimization_toggle.setChecked(False)
    
    def _set_to_most_accurate_settings(self):
        self._auto_criterion_toggle.setChecked(True)
        self._hmm_algorithm_selector.setCurrentText("Baum-Welch")
        self._hmm_scan_toggle.setChecked(True)
        self._final_baum_welch_optimization_toggle.setChecked(True)
    
    def _on_auto_disc_toggled(self):
        self._div_criterion_selector.setEnabled(not self._auto_criterion_toggle.isChecked())
        self._agg_criterion_selector.setEnabled(not self._auto_criterion_toggle.isChecked())
    
    def _on_num_levels_changed(self):
        self._hmm_scan_toggle.setEnabled(self._num_levels_edit.text().strip() == "")
    
    def _on_hmm_algorithm_changed(self):
        self._final_baum_welch_optimization_toggle.setEnabled(self._hmm_algorithm_selector.currentText() != "Baum-Welch")
        self._num_viterbi_repeats.setEnabled(self._hmm_algorithm_selector.currentText() == "Viterbi")
    
    def _on_tags_edited(self):
        if not self._traces:
            return
        trace = self._traces[self._trace_selector.value() - 1]
        trace.tags = self._tags_edit.text()
    
    def _on_tags_filter_edited(self):
        pass

    def _is_tags_filter_enabled(self) -> bool:
        return self._tags_filter_edit.text().strip() != ""


class SegmentationTreeNode:

    def __init__(self, indices: np.ndarray, parent: SegmentationTreeNode = None):
        self.indices = indices
        self.parent: SegmentationTreeNode = parent
        self.children: list[SegmentationTreeNode] = []
    
    @property
    def root(self) -> SegmentationTreeNode:
        node = self
        while node.parent is not None:
            node = node.parent
        return node
    
    @property
    def first_child(self) -> SegmentationTreeNode | None:
        if self.children:
            return self.children[0]

    @property
    def last_child(self) -> SegmentationTreeNode | None:
        if self.children:
            return self.children[-1]
    
    @property
    def next_sibling(self) -> SegmentationTreeNode | None:
        if self.parent is not None:
            siblings = self.parent.children
            i: int = siblings.index(self)
            if i+1 < len(siblings):
                return siblings[i+1]

    @property
    def prev_sibling(self) -> SegmentationTreeNode | None:
        if self.parent is not None:
            siblings = self.parent.children
            i: int = siblings.index(self)
            if i-1 >= 0:
                return siblings[i-1]
    
    def depth(self, root: SegmentationTreeNode = None) -> int:
        depth: int = 0
        node: SegmentationTreeNode = self
        while (node.parent is not None) and (node is not root):
            depth += 1
            node = node.parent
        return depth
    
    def next_depth_first(self) -> SegmentationTreeNode | None:
        if self.children:
            return self.children[0]
        next_sibling = self.next_sibling
        if next_sibling is not None:
            return next_sibling
        node = self.parent
        while node is not None:
            next_sibling = node.next_sibling
            if next_sibling is not None:
                return next_sibling
            node = node.parent
    
    def has_ancestor(self, node: SegmentationTreeNode) -> bool:
        ancestor = self
        while ancestor is not None:
            if ancestor is node:
                return True
            ancestor = ancestor.parent
        return False
    
    def is_leaf(self) -> bool:
        return not self.children
    
    def leaves(self) -> list[SegmentationTreeNode]:
        leaves: list[SegmentationTreeNode] = []
        node = self
        while node is not None:
            if node.is_leaf() and node.has_ancestor(self):
                leaves.append(node)
            node = node.next_depth_first()
        return leaves


def simulate_trace(
    n_pts: int = None,
    levels: np.ndarray = None,
    sigmas: np.ndarray = None,
    transition_proba: np.ndarray = None,
    level_per_event_heterogeneity: float = 0
    ):

    if n_pts is None:
        n_pts = np.random.choice(np.arange(100, 3000, dtype=int))
    
    if levels is None:
        if transition_proba is not None:
            n_levels = len(transition_proba)
        elif sigmas is not None:
            n_levels = len(sigmas)
        else:
            n_levels = np.random.choice(np.arange(2, 5, dtype=int))
        levels = np.arange(n_levels) + np.random.uniform(-0.2, 0.2, size=n_levels)
    else:
        n_levels = len(levels)
    
    if sigmas is None:
        sigmas = np.random.uniform(0.25, 0.4, size=n_levels)
    
    if transition_proba is None:
        transition_proba = np.random.random([n_levels, n_levels])
        for i, row in enumerate(transition_proba):
            row[i] = np.random.uniform(0.8, 0.99)
            j = np.arange(n_levels, dtype=int) != i
            row[j] = (1 - row[i]) * row[j] / row[j].sum()
    
    state_seq = np.zeros(n_pts, dtype=int)
    state_seq[0] = np.random.choice(np.arange(n_levels))
    rvs = np.random.random(size=n_pts)
    transition_proba_cumsum = np.cumsum(transition_proba, axis=1)
    for t in range(1, n_pts):
        state_seq[t] = np.where(rvs[t] < transition_proba_cumsum[state_seq[t-1],:])[0][0]
    
    data = np.zeros(n_pts)
    noiseless = np.zeros(n_pts)
    for state in range(n_levels):
        is_state = state_seq == state
        data[is_state] = stats.norm.rvs(levels[state], sigmas[state], size=is_state.sum())
        noiseless[is_state] = levels[state]
    
    if level_per_event_heterogeneity > 0:
        starts, stops = find_piecewise_constant_segments(state_seq)
        n_events = len(starts)
        event_states = state_seq[starts]
        event_offsets = stats.norm.rvs(0, 1, size=n_events) * sigmas[event_states] * level_per_event_heterogeneity
        for start, stop, offset in zip(starts, stops, event_offsets):
            data[start:stop] += offset
    
    return data, noiseless, (levels, sigmas, transition_proba, level_per_event_heterogeneity)


def find_piecewise_constant_segments(data: np.ndarray):
    """ Return start and stop indices for all piecewise constant segments.

    For boolean array, only return start and stop indices for all True segments.
    """
    if data.dtype == bool:
        # start/stop for True events only
        ddata = np.diff(data.astype(int))
        starts = 1 + np.where(ddata == 1)[0]
        stops = 1 + np.where(ddata == -1)[0]
        if data[0]:
            starts = np.insert(starts, 0, 0)
        if data[-1]:
            stops = np.append(stops, len(data))
    else:
        # start/stop for all piecewise constant segments
        ddata = np.diff(data)
        starts = 1 + np.where(ddata != 0)[0]
        stops = starts.copy()
        starts = np.insert(starts, 0, 0)
        stops = np.append(stops, len(data))
    return starts, stops


def find_change_points(data: np.ndarray, alpha=0.05, min_pts_per_segment=1):
    """ Change point detection using recursive binary segmentation and student's T-test until critical value is met.
    """
    n_pts = len(data)
    critical_value = stats.t.ppf(1 - alpha / 2, n_pts)
    sigma_noise = gaussian_noise_estimate(data)

    # Find change points
    is_change_point = np.zeros(data.shape, dtype=bool)
    start = 0
    stop = n_pts
    is_change_point[0] = True
    while start + 1 < n_pts:
        if (stop - start) >= 2 * min_pts_per_segment:
            cp, t = find_change_point(data[start:stop], sigma_noise, min_pts_per_segment)
            if t > critical_value:
                # Accept the change-point
                stop = start + cp
                is_change_point[stop] = True
                continue
        # Advance to next segment
        start = stop
        if start + 1 < n_pts:
            try:
                stop = start + 1 + np.where(is_change_point[start+1:])[0][0]
            except:
                stop = n_pts
    # segment starts and stops
    starts = np.where(is_change_point)[0]
    stops = starts[1:].copy()
    stops = np.append(stops, n_pts)
    return starts, stops


def find_change_point(data: np.ndarray, sigma_noise: float = None, min_pts_per_segment=1):
    """ Change point detection using binary segmentation and student's T-test.

        Return change point index and t-value for split.
    """
    if sigma_noise is None:
        sigma_noise = gaussian_noise_estimate(data)
    n_pts = len(data)

    # Potential change points
    start = min_pts_per_segment - 1
    stop = n_pts - min_pts_per_segment
    pts = np.arange(start, stop)

    # Mean of segments split by change point for all possible splits.
    cum_sum = np.cumsum(data)
    total_sum = cum_sum[-1]
    mean1 = cum_sum[start:stop] / pts
    mean2 = (total_sum - cum_sum[start:stop]) / (n_pts - pts)

    # Compute t-value for each potential change point.
    tvalues = np.abs(mean2 - mean1) / (sigma_noise * np.sqrt(1.0 / pts + 1.0 / (n_pts - pts)))

    # Select change point with maximum t-value.
    i = np.argmax(tvalues)
    change_point = pts[i]

    return change_point, tvalues[i]


def idealize_change_points(data: np.ndarray, starts: np.ndarray, stops: np.ndarray, level_func=np.median):
    idealized_data = np.zeros(data.shape)
    for start, stop in zip(starts, stops):
        idealized_data[start:stop] = level_func(data[start:stop])
    return idealized_data


def gaussian_noise_estimate(data: np.ndarray):
    n_pts = len(data)
    sorted_wavelet = np.sort(np.abs(np.diff(data) / 1.4))
    sigma_noise = sorted_wavelet[int(np.round(0.682 * (n_pts - 1)))]
    return sigma_noise


def estimate_SNR(data: np.ndarray, idealized_data: np.ndarray):
    residuals = data - idealized_data
    _, sigma_noise = stats.norm.fit(residuals)
    starts, stops = find_piecewise_constant_segments(idealized_data)
    durs = stops - starts
    steps = idealized_data[starts[1:]] - idealized_data[starts[1:] - 1]
    repeats = durs[:-1] + durs[1:]
    signals = []
    for step, rep in zip(steps, repeats):
        signals += [step] * rep
    signals = np.abs(signals)
    not_noise = signals > 2 * sigma_noise
    if np.any(not_noise):
        estimated_signal = np.mean(signals[not_noise])
    elif len(signals) > 0:
        estimated_signal = np.mean(signals)
    else:
        estimated_signal = 0
    return estimated_signal / sigma_noise


def divisive_segmentation(data: np.ndarray, criterion="BIC", level_func=np.median, n_required_levels=None):
    """ Divisive segmentation of sequence with jumps between discrete levels.

    Note: The K-means splitting introduces randomness, so the results may vary between runs.
    """
    n_pts = len(data)
    idealized_data = np.ones(data.shape) * data.mean()

    kmeans_splitter = KMeans(n_clusters=2, n_init='auto')
    # gmm_splitter = GaussianMixture(n_components=2)

    # Start with all points in one cluster
    root = SegmentationTreeNode(indices=np.arange(n_pts))
    node = root
    forced_split = None
    while node is not None:
        # data in cluster 
        cluster_data = data[node.indices]
        
        # try splitting the cluster in two
        if len(np.unique(cluster_data)) >= 3:#2:
            # kmeans_splitter.init = np.quantile(np.unique(cluster_data), [[0.33], [0.66]]).reshape([-1,1])
            split_labels = kmeans_splitter.fit_predict(cluster_data.reshape([-1,1]))

            # gmm_splitter.means_init = np.quantile(np.unique(cluster_data), [[0.33], [0.66]]).reshape([-1,1])
            # split_labels = gmm_splitter.fit_predict(cluster_data.reshape([-1,1]))

            is_split0 = split_labels == 0
            is_split1 = split_labels == 1
            n_pts0 = is_split0.sum()
            n_pts1 = is_split1.sum()
            if (n_pts0 > 0) and (n_pts1 > 0):
                cluster_ideal_split = kmeans_splitter.cluster_centers_[split_labels].reshape(-1)
                # cluster_ideal_split = gmm_splitter.means_[split_labels].reshape(-1)
                if level_func is not None:
                    cluster_ideal_split[is_split0] = level_func(cluster_data[is_split0])
                    cluster_ideal_split[is_split1] = level_func(cluster_data[is_split1])
                do_split = node.depth() == 0
                if not do_split:
                    split_metric = information_criterion(cluster_data, cluster_ideal_split, criterion)
                    combined_metric = information_criterion(cluster_data, idealized_data[node.indices], criterion)
                    do_split = split_metric < combined_metric
                if not do_split and (n_pts < 1000):
                    # force one extra split because not splitting enough can lead to suboptimal solutions
                    # especially when the number of samples is small
                    ancestor = node.parent
                    while (ancestor is not None) and (forced_split is not ancestor):
                        ancestor = ancestor.parent
                    if ancestor is None:
                        do_split = True
                        forced_split = node
                if do_split:
                    idealized_data[node.indices] = cluster_ideal_split
                    child0 = SegmentationTreeNode(indices=node.indices[is_split0], parent=node)
                    child1 = SegmentationTreeNode(indices=node.indices[is_split1], parent=node)
                    node.children = [child0, child1]
                    # try to split first child
                    node = child0
                    continue
            
        # if we are done splitting this node, go to next node
        node = node.next_depth_first()
    
    if n_required_levels is not None:
        while len(np.unique(idealized_data)) < n_required_levels:
            # split leaves of tree until we have at least n_levels
            node = root
            did_split = False
            while node is not None:
                if not node.children:
                    # split leaf node
                    cluster_data = data[node.indices]
                    n_pts = len(cluster_data)
                    if n_pts >= 3:#2:
                        split_labels = kmeans_splitter.fit_predict(cluster_data.reshape([-1,1]))
                        is_split0 = split_labels == 0
                        is_split1 = split_labels == 1
                        n_pts0 = is_split0.sum()
                        n_pts1 = is_split1.sum()
                        if (n_pts0 > 0) and (n_pts1 > 0):
                            cluster_ideal_split = kmeans_splitter.cluster_centers_[split_labels].reshape(-1)
                            if level_func is not None:
                                cluster_ideal_split[is_split0] = level_func(cluster_data[is_split0])
                                cluster_ideal_split[is_split1] = level_func(cluster_data[is_split1])
                            idealized_data[node.indices] = cluster_ideal_split
                            child0 = SegmentationTreeNode(indices=node.indices[is_split0], parent=node)
                            child1 = SegmentationTreeNode(indices=node.indices[is_split1], parent=node)
                            node.children = [child0, child1]
                            node = child1
                            did_split = True
                node = node.next_depth_first()
            if not did_split:
                # cannot split any more nodes
                if len(np.unique(idealized_data)) < n_required_levels:
                    # Still don't have n_levels, so try using the raw data instead.
                    # This is a last resort, because it means the divisive segmentation was pointless.
                    idealized_data = data
                    if np.unique(idealized_data).size < n_required_levels:
                        raise RuntimeError("More levels requested than exist in data.")
                break
    
    return idealized_data, root


def agglomerative_clustering(data, idealized_data, level_func=np.median):
    """ Iteratively merge levels with minimal Ward distance.
    """
    levels = np.unique(idealized_data)
    level_npts = np.array([np.sum(idealized_data == level) for level in levels])
    tmp = idealized_data.copy()
    idealized_data = np.zeros([len(levels), len(data)])
    idealized_data[-1] = tmp
    for i in np.flip(np.arange(len(idealized_data) - 1)):
        idealized_data[i] = idealized_data[i+1].copy()

        # Ward's distance between each level
        # ward_dist = (2 * level_npts[:-1] * level_npts[1:] / (level_npts[:-1] + level_npts[1:]))**0.5 * (levels[:-1] - levels[1:])**2
        ward_merge_cost = level_npts[:-1] * level_npts[1:] / (level_npts[:-1] + level_npts[1:]) * (levels[:-1] - levels[1:])**2
        
        # merge levels with minimal Ward distance
        j = np.argmin(ward_merge_cost)
        k = j + 1
        is_j = np.isclose(idealized_data[i], levels[j])
        is_k = np.isclose(idealized_data[i], levels[k])
        is_merge = is_j | is_k
        levels[j] = level_func(data[is_merge])
        levels = np.delete(levels, k)
        level_npts[j] += level_npts[k]
        level_npts = np.delete(level_npts, k)
        idealized_data[i,is_merge] = levels[j]
    return idealized_data


def information_criterion(data: np.ndarray, idealized_data: np.ndarray, criterion: str) -> float:
    n_pts = len(data)

    if criterion == "RSS":
        n_change_points = (np.diff(idealized_data) != 0).sum()
        n_levels = len(np.unique(idealized_data))
        dof = n_change_points + n_levels
        RSS = np.sum((data - idealized_data)**2)
        goodness_of_fit = n_pts * np.log(RSS / n_pts)
        if np.isinf(goodness_of_fit) or np.isnan(goodness_of_fit):
            return np.log(n_pts) * dof
        else:
            return goodness_of_fit + np.log(n_pts) * dof
    
    # if criterion == "MDL":
    #     pass # TODO

    # if criterion == "Silhouette":
    #     states = np.unique(idealized_data)
    #     if len(states) == 1:
    #         return np.nan
    #     labels = np.zeros(idealized_data.shape, dtype=int)
    #     for i, state in enumerate(states):
    #         labels[idealized_data == state] = i
    #     return silhouette_score(data.reshape(-1,1), labels)

    # # ignore this, just playing around
    # if criterion == "Evidence":
    #     se = (data - ideal)**2
    #     starts = 1 + np.where(np.diff(idealized_data) != 0)[0]
    #     starts = np.insert(starts, 0, 0)
    #     stops = starts[1:].copy()
    #     stops = np.append(stops, n_pts)
    #     evidence = 0
    #     for start, stop in zip(starts, stops):
    #         evidence += (stop - start) / np.mean(se[start:stop])
    #     return evidence

    # Likelihood based on Gaussian Mixture Model
    levels = np.unique(idealized_data)
    n_levels = len(levels)
    means = np.zeros([n_levels,1])
    sigmas = np.zeros([n_levels,1])
    weights = np.zeros([n_levels,1])
    for i, level in enumerate(levels):
        level_data = data[idealized_data == level]
        means[i] = np.mean(level_data)
        sigmas[i] = np.std(level_data)
        weights[i] = float(len(level_data)) / n_pts
    sigma_is_zero = (sigmas == 0)
    if np.any(sigma_is_zero):
        if np.all(sigma_is_zero):
            sigmas[:] = gaussian_noise_estimate(data)
        else:
            sigmas[sigma_is_zero] = np.mean(sigmas[~sigma_is_zero])
    likelihood = (weights * stats.norm.pdf(data, means, sigmas)).sum(axis=0)
    log_likelihood = np.sum(np.log(likelihood))
    dof = 3 * n_levels - 1

    if criterion == "AIC":
        return -2 * log_likelihood + 2 * dof
    elif criterion == "BIC":
        return -2 * log_likelihood + np.log(n_pts) * dof
    elif criterion == "HQC":
        return -2 * log_likelihood + 2 * np.log(np.log(n_pts)) * dof


def gmm_idealization(
    data: np.ndarray,
    means: np.ndarray,
    stdevs: np.ndarray = None,
    optimize_means=False,
    optimize_stdevs=False
    ) -> tuple[np.ndarray, GeneralMixtureModel]:

    n_states = len(means)

    if stdevs is None:
        stdevs = np.ones(n_states) * gaussian_noise_estimate(data)
        optimize_stdevs = True

    states = []
    for i in range(n_states):
        state = Normal(means=torch.tensor([means[i]]).float(), covs=torch.tensor([[stdevs[i]**2]]).float())
        if not optimize_means:
            state.means.frozen = True
        if not optimize_stdevs:
            state.covs.frozen = True
        states.append(state)
    
    gmm = GeneralMixtureModel(states)
    X = torch.tensor(data.reshape(-1,1)).float()
    gmm.fit(X)
    state_sequence = gmm.predict(X).numpy().reshape(data.shape)
    
    idealized_data = np.zeros(data.shape)
    for i in range(n_states):
        in_state = state_sequence == i
        if not np.any(in_state):
            continue
        idealized_data[in_state] = gmm.distributions[i].means.item()
    return idealized_data, gmm


def hmm_idealization_refinement(
    data: np.ndarray, 
    idealized_data: np.ndarray = None, 
    level_means: np.ndarray = None,
    level_stdevs: np.ndarray = None,
    optimize_level_means=True,
    optimize_level_stdevs=True,
    algorithm='viterbi', 
    minimum_transition_proba=0.02
    ):

    if (idealized_data is None) and (level_means is None):
        raise ValueError("Either idealized_data or level_means must be provided.")

    n_pts = len(data)
    
    if level_means is not None:
        # idealized data from state distributions
        idealized_data, gmm = gmm_idealization(data, means=level_means, stdevs=level_stdevs, optimize_means=False, optimize_stdevs=False)
        if level_stdevs is None:
            level_stdevs = np.array([state.covs.item()**0.5 for state in gmm.distributions])
        n_levels = len(level_means)
    elif idealized_data is not None:
        # state distributions from idealized data
        levels = np.unique(idealized_data)
        n_levels = len(levels)
        level_means = np.zeros(n_levels)
        level_stdevs = np.zeros(n_levels)
        for i in range(n_levels):
            in_level = idealized_data == levels[i]
            level_data = data[in_level]
            level_means[i] = np.mean(level_data)
            if len(level_data) > 1:
                level_stdevs[i] = np.std(level_data)
        sigma_is_zero = (level_stdevs == 0)
        if np.any(sigma_is_zero):
            if np.all(sigma_is_zero):
                level_stdevs[:] = gaussian_noise_estimate(data)
            else:
                level_stdevs[sigma_is_zero] = np.mean(level_stdevs[~sigma_is_zero])
    
    # estimated state transition probabilities from idealized data
    if n_levels == 1:
        transition_proba = np.ones([1,1])
    else:
        transition_proba = np.zeros([n_levels, n_levels])
        for i in range(n_levels):
            in_level_i = np.isclose(idealized_data, level_means[i])
            for j in range(n_levels):
                in_level_j = np.isclose(idealized_data, level_means[j])
                num_i_to_j = (in_level_i[:-1] & in_level_j[1:]).sum()
                transition_proba[i,j] = float(num_i_to_j) / n_pts
        # normalize
        norm = transition_proba.sum(axis=1).reshape((-1,1))
        norm[norm == 0] = 1 / n_pts
        transition_proba /= norm
        if minimum_transition_proba > 0:
            # no completely disallowed transitions
            transition_proba[transition_proba < minimum_transition_proba] = minimum_transition_proba
            # normalize
            norm = transition_proba.sum(axis=1).reshape((-1,1))
            norm[norm == 0] = 1 / n_pts
            transition_proba /= norm
    
    # estimated state starting probabilities
    # TODO: set to equilibrium probabilities using transition matrix?
    start_proba = np.ones(n_levels) / n_levels

    algorithm = algorithm.lower()

    if algorithm == 'viterbi':
        emission_likelihood = np.zeros([n_levels, n_pts])
        for i in range(n_levels):
            emission_likelihood[i] = stats.norm.pdf(data, level_means[i], level_stdevs[i])

        state_sequence = viterbi_path(start_proba, transition_proba, emission_likelihood)

        hmm = None

        if optimize_level_means or optimize_level_stdevs:
            # optimize state means and stdevs
            for i in range(n_levels):
                in_state = state_sequence == i
                n_state = np.sum(in_state)
                if optimize_level_means and (n_state > 0):
                    level_means[i] = np.mean(data[in_state])
                if optimize_level_stdevs and (n_state > 1):
                    level_stdevs[i] = np.std(data[in_state])
    
    elif algorithm == 'baum-welch':
        states = []
        for i in range(n_levels):
            state = Normal(means=torch.tensor([level_means[i]]).float(), covs=torch.tensor([[level_stdevs[i]**2]]).float())
            if not optimize_level_means:
                state.means.frozen = True
            if not optimize_level_stdevs:
                state.covs.frozen = True
            states.append(state)
        
        transition_proba = torch.tensor(transition_proba).float()
        start_proba = torch.tensor(start_proba).float()
        
        hmm = DenseHMM(states, starts=start_proba, edges=transition_proba)

        # reshape data [number of sequences, sequence length, dimensions of each data point] and convert to tensor
        X = torch.tensor(data.reshape(1, n_pts, 1)).float()

        # optimize model parameters
        hmm.fit(X)

        level_means = np.array([state.means.item() for state in states])
        level_stdevs = np.array([state.covs.item()**0.5 for state in states])

        # optimal state sequence
        state_sequence = hmm.predict(X).numpy().reshape(data.shape)
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # idealized sequence
    for i in range(n_levels):
        in_state = state_sequence == i
        if not np.any(in_state):
            continue
        idealized_data[in_state] = level_means[i]
    
    return idealized_data, hmm


def viterbi_path(start_proba, transition_proba, emission_likelihood, scaled=True):
    """Finds the most-probable (Viterbi) path through the HMM state trellis

    At the moment, pomegranate 1.0 has not yet implemented Viterbi, so we'll roll it ourselves.

    TODO: Use log(probabilities) instead of raw probabilities (or is scaling good enough)???

    Based on https://stackoverflow.com/questions/9729968/python-implementation-of-viterbi-algorithm

    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        start_proba: np.array(num_hid)
            start_proba[i] := Pr(Q[0] == i)
        transition_proba: np.ndarray((num_hid,num_hid))
            trans_proba[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        emission_likelihood: np.ndarray((num_hid,num_obs))
            emission_lik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    """
    num_hid = emission_likelihood.shape[0] # number of hidden states
    num_obs = emission_likelihood.shape[1] # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = np.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid,num_obs), dtype=int) # int because its elements will be used as indicies
    path = np.zeros(num_obs, dtype=int) # int because its elements will be used as indicies

    trellis_prob[:,0] = start_proba * emission_likelihood[:,0] # element-wise mult
    if scaled:
        scale = np.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 / np.sum(trellis_prob[:,0])
        trellis_prob[:,0] *= scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            trans_probs = trellis_prob[:,t-1] * transition_proba[:,j] # element-wise mult
            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] *= emission_likelihood[j,t]
        if scaled:
            scale[t] = 1.0 / np.sum(trellis_prob[:,t])
            trellis_prob[:,t] *= scale[t]

    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    return path


def merge_nearest_levels(data, idealized_data, level_func=np.median):
    levels = np.unique(idealized_data)
    if len(levels) < 2:
        return
    i = np.argmin(np.diff(levels))
    j = i + 1
    is_i = np.isclose(idealized_data, levels[i])
    is_j = np.isclose(idealized_data, levels[j])
    is_merge = is_i | is_j
    idealized_data[is_merge] = level_func(data[is_merge])


def run_DISC(
    data: np.ndarray,
    div_criterion: str = 'BIC',
    agg_criterion: str = 'BIC',
    n_required_levels: int = None,
    level_func = np.median,
    n_div_attempts: int = None,
    hmm_algorithm: str = 'viterbi', # 'viterbi' or 'baum-welch'
    hmm_optimize_states: bool = True,
    n_viterbi_repeats: int = 2,
    hmm_scan: bool = False,
    final_baum_welch_optimization: bool = False,
    return_intermediate_results: bool = False,
    verbose: bool = False
    ):

    if verbose:
        tic = time.time()
        print('---------- DISC ----------')
        print(f'required levels: {n_required_levels}')
        print(f'div criterion: {div_criterion}')
        print(f'agg criterion: {agg_criterion}')
    
    n_pts = len(data)

    # divisive segmentation
    # the kmeans splitter introduces some randomness which can lead to different results
    # so optionally segment multiple times and choose the best result (after agglomeration)
    if n_div_attempts is None:
        # default is to try more segmentations for shorter traces, less for longer traces
        n_div_attempts = 3 if n_pts < 1000 else (2 if n_pts < 3000 else 1)
    div_idealized_data = np.empty((n_div_attempts, n_pts))
    div_tree = [None] * n_div_attempts
    for i in range(n_div_attempts):
        div_idealized_data[i], div_tree[i] = divisive_segmentation(data, criterion=div_criterion, level_func=level_func, n_required_levels=n_required_levels)
    
    if verbose:
        toc = time.time()
        print(f'div (x{n_div_attempts}): {toc - tic} sec')
        tic = toc
    
    # agglomerative clustering (for each divisive segmentation result)
    agg_idealized_data = [None] * n_div_attempts
    agg_metrics = [None] * n_div_attempts
    agg_index = [None] * n_div_attempts
    for i in range(n_div_attempts):
        agg_idealized_data[i] = agglomerative_clustering(data, div_idealized_data[i], level_func=level_func)
        
        # evaluate agglomeration metric
        n_agg_traces = len(agg_idealized_data[i])
        agg_metrics[i] = np.empty(n_agg_traces)
        agg_metrics[i][:] = np.nan
        n_bad_steps = 0
        total_bad_steps = 0
        min_metric = None
        second_to_min_metric = None
        for j in range(n_agg_traces):
            agg_metrics[i][j] = information_criterion(data, agg_idealized_data[i][j], agg_criterion)

            # early stopping based on metric
            if n_required_levels is None:
                if j > 0:
                    if agg_metrics[i][j] > agg_metrics[i][j-1]:
                        n_bad_steps += 1
                        total_bad_steps += 1
                    else:
                        n_bad_steps = 0
                    if n_bad_steps >= 5:
                        break
                if j == 1:
                    min_metric, second_to_min_metric = np.sort(agg_metrics[i][:2])
                elif j > 1:
                    min_metric, second_to_min_metric = np.sort([min_metric, second_to_min_metric, agg_metrics[i][j]])[:2]
                if (j >= 6) and (total_bad_steps >= 2):
                    if np.all(agg_metrics[i][j-3:j+1] > second_to_min_metric):
                        break
            else:
                if j + 1 >= n_required_levels:
                    break
    
        # choose agglomeration level
        if n_required_levels is not None:
            if n_required_levels > n_agg_traces:
                raise RuntimeError("Number of requested levels exceeds divisive segmentation.")
            agg_index[i] = n_required_levels - 1
        else:
            agg_index[i] = None
            
            # non_nan = ~np.isnan(agg_metrics[i])
            # ind = np.where(non_nan)[0]
            # metrics = agg_metrics[i][non_nan]
            # knee_locator = KneeLocator(np.arange(metrics), metrics, curve='convex', direction='decreasing')
            # agg_index[i] = knee_locator.knee
            # if agg_index[i] is not None:
            #     agg_index[i] = ind[agg_index[i]]

            if agg_index[i] is None:
                try:
                    # only choose a single level if the metric is decreasing from 2 levels to 1 level
                    # and the metric is larger for all other # of levels
                    agg_index[i] = 1 + np.nanargmin(agg_metrics[i][1:])
                    if agg_index[i] == 1:
                        agg_index[i] = np.nanargmin(agg_metrics[i][:2])
                except [IndexError, ValueError]:
                    agg_index[i] = 0
    
    # choose segmentation that resulted in the best selected agglomeration metric
    div_index = np.argmin([agg_metrics[i][index] for i, index in enumerate(agg_index)])
    div_idealized_data = div_idealized_data[div_index]
    div_tree = div_tree[div_index]
    agg_idealized_data = agg_idealized_data[div_index]
    n_agg_traces = len(agg_idealized_data)
    agg_levels = 1 + np.arange(n_agg_traces)
    agg_metrics = agg_metrics[div_index]
    agg_index = agg_index[div_index]

    ok_agg_metrics = ~np.isnan(agg_metrics)
    agg_levels = agg_levels[ok_agg_metrics]
    agg_metrics = agg_metrics[ok_agg_metrics]

    if verbose:
        toc = time.time()
        print(f'agg: {toc - tic} sec')
        tic = toc
        print(f'div levels: {len(np.unique(div_idealized_data))}')
        print(f'agg levels: {agg_levels[agg_index]}')
    
    # hidden Markov model
    hmm_kwargs = {
        'algorithm': hmm_algorithm,
        'optimize_level_means': hmm_optimize_states,
        'optimize_level_stdevs': hmm_optimize_states,
    }
    viterbi_repeat_kwargs = {
        'algorithm': 'viterbi',
        'optimize_level_means': True,
        'optimize_level_stdevs': True,
    }
    if hmm_scan and (n_required_levels is None):
        # use HMM to scan for optimal number of levels in vicinity of chosen agglomeration level
        hmm_idealized_data = np.empty(agg_idealized_data.shape)
        hmms = [None] * n_agg_traces
        hmm_metrics = np.empty(n_agg_traces)
        hmm_metrics[:] = np.nan
        i = agg_index
        hmm_idealized_data[i], hmms[i] = hmm_idealization_refinement(data, agg_idealized_data[i], **hmm_kwargs)
        if (hmm_algorithm == 'viterbi') and (n_viterbi_repeats > 1) and hmm_optimize_states:
            for _ in range(1, n_viterbi_repeats):
                hmm_idealized_data[i], hmms[i] = hmm_idealization_refinement(data, hmm_idealized_data[i], **viterbi_repeat_kwargs)
        hmm_metrics[i] = information_criterion(data, hmm_idealized_data[i], agg_criterion)
        # explore more levels
        for j in range(i + 1, n_agg_traces):
            hmm_idealized_data[j], hmms[j] = hmm_idealization_refinement(data, agg_idealized_data[j], **hmm_kwargs)
            if (hmm_algorithm == 'viterbi') and (n_viterbi_repeats > 1) and hmm_optimize_states:
                for _ in range(1, n_viterbi_repeats):
                    hmm_idealized_data[j], hmms[j] = hmm_idealization_refinement(data, hmm_idealized_data[j], **viterbi_repeat_kwargs)
            hmm_metrics[j] = information_criterion(data, hmm_idealized_data[j], agg_criterion)
            if (hmm_metrics[j] > hmm_metrics[j-1]) and (j - i >= 2):
                # early stopping
                break
        # explore fewer levels
        for j in reversed(list(range(0, i))):
            hmm_idealized_data[j], hmms[j] = hmm_idealization_refinement(data, agg_idealized_data[j], **hmm_kwargs)
            if (hmm_algorithm == 'viterbi') and (n_viterbi_repeats > 1) and hmm_optimize_states:
                for _ in range(1, n_viterbi_repeats):
                    hmm_idealized_data[j], hmms[j] = hmm_idealization_refinement(data, hmm_idealized_data[j], **viterbi_repeat_kwargs)
            hmm_metrics[j] = information_criterion(data, hmm_idealized_data[j], agg_criterion)
            if (hmm_metrics[j] > hmm_metrics[j+1]) and ((i - j >= 2) or (j <= 1)):
                # early stopping
                break
        ok_hmm_metrics = ~np.isnan(hmm_metrics)
        hmm_idealized_data = hmm_idealized_data[ok_hmm_metrics]
        hmms = [hmm for hmm, ok in zip(hmms, ok_hmm_metrics) if ok]
        hmm_levels = np.array([len(np.unique(ideal)) for ideal in hmm_idealized_data], dtype=int)
        hmm_metrics = hmm_metrics[ok_hmm_metrics]
        hmm_index = np.nanargmin(hmm_metrics)
    else:
        # only evaluate HMM for chosen agglomeration level
        hmm_idealized_data, hmm = hmm_idealization_refinement(data, agg_idealized_data[agg_index], **hmm_kwargs)
        if (hmm_algorithm == 'viterbi') and (n_viterbi_repeats > 1) and hmm_optimize_states:
            for _ in range(1, n_viterbi_repeats):
                hmm_idealized_data, hmm = hmm_idealization_refinement(data, hmm_idealized_data, **viterbi_repeat_kwargs)
        # for compatibilty with result of scan
        hmm_idealized_data = hmm_idealized_data.reshape([1, -1])
        hmms = [hmm]
        hmm_levels = np.array([len(np.unique(hmm_idealized_data[0]))], dtype=int)
        hmm_metrics = np.array([information_criterion(data, hmm_idealized_data[0], agg_criterion)])
        hmm_index = 0

    if verbose:
        toc = time.time()
        print(f'HMM: {toc - tic} sec')
        print(f'HMM levels: {hmm_levels[hmm_index]}')
        tic = toc
    
    # idealization
    idealized_data = hmm_idealized_data[hmm_index]
    n_idealized_levels = hmm_levels[hmm_index]
    idealized_metric = hmm_metrics[hmm_index]
    final_hmm = hmms[hmm_index]

    if n_required_levels is None:
        # try merging nearest levels until no improvement in metric.
        # sometimes this can help, and it's fast, so why not?
        did_merge = False
        while n_idealized_levels > 1:
            merged_idealized_data = idealized_data.copy()
            merge_nearest_levels(data, merged_idealized_data)
            merged_metric = information_criterion(data, merged_idealized_data, agg_criterion)
            if merged_metric > idealized_metric:
                break
            idealized_data = merged_idealized_data
            n_idealized_levels = len(np.unique(idealized_data))
            idealized_metric = merged_metric
            did_merge = True
        
        # if we did not accept any merge, try adding two levels and then merging.
        # this can sometimes identify levels with few data points.
        if not did_merge:
            if hmm_index + 2 < len(hmm_idealized_data):
                test_idealized_data = hmm_idealized_data[hmm_index + 2].copy()
                hmm = hmms[hmm_index + 2]
            elif agg_index + 2 < len(agg_idealized_data):
                test_idealized_data = agg_idealized_data[agg_index + 2].copy()
                test_idealized_data, hmm = hmm_idealization_refinement(data, test_idealized_data, algorithm=hmm_algorithm, optimize_level_means=hmm_optimize_states, optimize_level_stdevs=hmm_optimize_states)
            elif hmm_index + 1 < len(hmm_idealized_data):
                test_idealized_data = hmm_idealized_data[hmm_index + 1].copy()
                hmm = hmms[hmm_index + 1]
            elif agg_index + 1 < len(agg_idealized_data):
                test_idealized_data = agg_idealized_data[agg_index + 1].copy()
                test_idealized_data, hmm = hmm_idealization_refinement(data, test_idealized_data, algorithm=hmm_algorithm, optimize_level_means=hmm_optimize_states, optimize_level_stdevs=hmm_optimize_states)
            else:
                test_idealized_data = None
            if test_idealized_data is not None:
                n_test_levels = len(np.unique(test_idealized_data))
                test_metric = information_criterion(data, test_idealized_data, agg_criterion)
                while n_test_levels > 1:
                    merged_idealized_data = test_idealized_data.copy()
                    merge_nearest_levels(data, merged_idealized_data)
                    merged_metric = information_criterion(data, merged_idealized_data, agg_criterion)
                    if merged_metric > test_metric:
                        break
                    test_idealized_data = merged_idealized_data
                    n_test_levels = len(np.unique(test_idealized_data))
                    test_metric = merged_metric
                    did_merge = True
                if test_metric < idealized_metric:
                    idealized_data = test_idealized_data
                    n_idealized_levels = n_test_levels
                    idealized_metric = test_metric
                    final_hmm = hmm
        
        if did_merge:
            # need HMM refinement after merging
            idealized_data, final_hmm = hmm_idealization_refinement(data, idealized_data, algorithm=hmm_algorithm, optimize_level_means=hmm_optimize_states, optimize_level_stdevs=hmm_optimize_states)
            if (hmm_algorithm == 'viterbi') and (n_viterbi_repeats > 1) and hmm_optimize_states:
                for _ in range(1, n_viterbi_repeats):
                    idealized_data, final_hmm = hmm_idealization_refinement(data, idealized_data, algorithm='viterbi', optimize_level_means=True, optimize_level_stdevs=True)
            n_idealized_levels = len(np.unique(idealized_data))
            idealized_metric = information_criterion(data, idealized_data, agg_criterion)

        if verbose:
            toc = time.time()
            print(f'merging: {toc - tic} sec')
            tic = toc
    
    # # chosen agglomeration level
    # agg_idealized_data = agg_idealized_data[agg_index]

    # # chosed HMM from scan
    # hmm_idealized_data = hmm_idealized_data[hmm_index]

    # final HMM Baum-Welch optimization
    if final_baum_welch_optimization and (hmm_algorithm != 'baum-welch'):
        idealized_data, final_hmm = hmm_idealization_refinement(data, idealized_data, algorithm='baum-welch', optimize_level_means=True, optimize_level_stdevs=True)
        n_idealized_levels = len(np.unique(idealized_data))
        idealized_metric = information_criterion(data, idealized_data, agg_criterion)

        if verbose:
            toc = time.time()
            print(f'final HMM: {toc - tic} sec')
            tic = toc
    
    if verbose:
        print(f'levels: {n_idealized_levels}')
        print(f'metric: {idealized_metric}')
        print('--------------------------')
    
    if return_intermediate_results:
        intermediate_results = {
            'div_idealized_data': div_idealized_data,
            'agg_idealized_data': agg_idealized_data,
            'hmm_idealized_data': hmm_idealized_data,
            'div_tree': div_tree,
            'agg_levels': agg_levels,
            'agg_metrics': agg_metrics,
            'agg_index': agg_index,
            'hmms': hmms,
            'hmm_levels': hmm_levels,
            'hmm_metrics': hmm_metrics,
            'hmm_index': hmm_index,
            'final_hmm': final_hmm,
        }
        return idealized_data, idealized_metric, intermediate_results
    
    return idealized_data, idealized_metric


def auto_DISC(
    data: np.ndarray,
    div_criterion: str = 'BIC',
    agg_criterion: str = 'BIC',
    n_required_levels: int = None,
    level_func = np.median,
    n_div_attempts: int = None,
    hmm_algorithm: str = 'viterbi', # 'viterbi' or 'baum-welch'
    hmm_optimize_states: bool = True,
    n_viterbi_repeats: int = 2,
    hmm_scan: bool = False,
    final_baum_welch_optimization: bool = False,
    return_intermediate_results: bool = False,
    verbose: bool = False
    ):

    criterion = "RSS"
    results = run_DISC(
        data,
        div_criterion = criterion,
        agg_criterion = criterion,
        n_required_levels = n_required_levels,
        level_func = level_func,
        n_div_attempts = n_div_attempts,
        hmm_algorithm = hmm_algorithm,
        hmm_optimize_states = hmm_optimize_states,
        n_viterbi_repeats = n_viterbi_repeats,
        hmm_scan = hmm_scan,
        final_baum_welch_optimization = final_baum_welch_optimization,
        return_intermediate_results = return_intermediate_results,
        verbose = verbose
        )

    n_pts = len(data)
    idealized_data = results[0]
    estimated_SNR = estimate_SNR(data, idealized_data)

    # choose criterion based on n_pts and estimated_SNR
    # see Bandyopadhyay and Goldschen-Ohm (2021) in Biophysical Journal
    log10_boundary_samples = -0.49 * estimated_SNR + 4.69
    if verbose:
        print('auto (relative to boundary):', np.log10(n_pts) - log10_boundary_samples)
    if np.log10(n_pts) >= log10_boundary_samples:
        criterion = "AIC"
    else:
        criterion = "RSS"

    if criterion != "RSS":
        results = run_DISC(
            data,
            div_criterion = criterion,
            agg_criterion = criterion,
            n_required_levels = n_required_levels,
            level_func = level_func,
            n_div_attempts = n_div_attempts,
            hmm_algorithm = hmm_algorithm,
            hmm_optimize_states = hmm_optimize_states,
            n_viterbi_repeats = n_viterbi_repeats,
            hmm_scan = hmm_scan,
            final_baum_welch_optimization = final_baum_welch_optimization,
            return_intermediate_results = return_intermediate_results,
            verbose = verbose
            )
    
    return results, criterion


if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication
    app = QApplication()
    widget = DISCO()
    widget.show()
    app.exec()