# -*- coding: utf-8 -*-

"""Trace view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

import numpy as np

from phy.utils import Bunch
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Trace view
# -----------------------------------------------------------------------------

def select_traces(traces, interval, sample_rate=None):
    """Load traces in an interval (in seconds)."""
    start, end = interval
    i, j = round(sample_rate * start), round(sample_rate * end)
    i, j = int(i), int(j)
    traces = traces[i:j, :]
    # traces = traces - np.mean(traces, axis=0)
    return traces


class TraceView(ManualClusteringView):
    interval_duration = .25  # default duration of the interval
    shift_amount = .1
    scaling_coeff_x = 1.5
    scaling_coeff_y = 1.1
    default_trace_color = (.75, .75, .75, .75)
    default_shortcuts = {
        'go_left': 'alt+left',
        'go_right': 'alt+right',
        'decrease': 'alt+down',
        'increase': 'alt+up',
        'toggle_show_labels': 'alt+l',
        'widen': 'alt+-',
        'narrow': 'alt++',
    }

    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 duration=None,
                 n_channels=None,
                 channel_positions=None,
                 channel_order=None,
                 **kwargs):

        self.do_show_labels = False

        # traces is a function interval => [traces]
        # spikes is a function interval => [Bunch(...)]

        # Sample rate.
        assert sample_rate > 0
        self.sample_rate = float(sample_rate)
        self.dt = 1. / self.sample_rate

        # Traces and spikes.
        assert hasattr(traces, '__call__')
        self.traces = traces

        assert duration >= 0
        self.duration = duration

        assert n_channels >= 0
        self.n_channels = n_channels

        channel_positions = (channel_positions
                             if channel_positions is not None
                             else np.c_[np.arange(n_channels),
                                        np.zeros(n_channels)])
        assert channel_positions.shape == (n_channels, 2)
        self.channel_positions = channel_positions

        channel_order = (channel_order if channel_order is not None
                         else np.arange(n_channels))
        assert channel_order.shape == (n_channels,)
        self.channel_order = channel_order

        # Double argsort for inverse permutation.
        self.channel_vertical_order = \
            np.argsort(np.argsort(channel_positions[:, 1]))

        # Box and probe scaling.
        self._scaling = 1.
        self._origin = None

        # Initialize the view.
        super(TraceView, self).__init__(layout='stacked',
                                        origin=self.origin,
                                        n_plots=self.n_channels,
                                        **kwargs)

        # Make a copy of the initial box pos and size. We'll apply the scaling
        # to these quantities.
        self.box_size = np.array(self.stacked.box_size)
        self._update_boxes()

        # Initial interval.
        self._interval = None
        self.go_to(duration / 2.)

    # Internal methods
    # -------------------------------------------------------------------------

    def _plot_traces(self, traces, color=None):
        traces = traces.T
        n_samples = traces.shape[1]
        n_ch = self.n_channels
        assert traces.shape == (n_ch, n_samples)
        color = color or self.default_trace_color

        t = self._interval[0] + np.arange(n_samples) * self.dt
        t = self._normalize_time(t)
        t = np.tile(t, (n_ch, 1))
        # Display the channels in vertical order.
        order = self.channel_vertical_order
        box_index = np.repeat(order[:, np.newaxis],
                              n_samples,
                              axis=1)

        assert t.shape == (n_ch, n_samples)
        assert traces.shape == (n_ch, n_samples)
        assert box_index.shape == (n_ch, n_samples)

        self.plot(t, traces,
                  color=color,
                  data_bounds=None,
                  box_index=box_index,
                  uniform=True,
                  )

    def _plot_waveforms(self, waveforms=None,
                        channel_ids=None,
                        start_time=None,
                        color=None,
                        ):
        # The spike time corresponds to the first sample of the waveform.
        n_samples, n_channels = waveforms.shape
        c = self.channel_vertical_order
        if channel_ids is not None:
            assert len(channel_ids) == n_channels
            c = c[channel_ids]

        # Generate the x coordinates of the waveform.
        t = start_time + self.dt * np.arange(n_samples)
        t = self._normalize_time(t)
        t = np.tile(t, (n_channels, 1))  # (n_unmasked_channels, n_samples)

        # The box index depends on the channel.
        box_index = np.repeat(c[:, np.newaxis], n_samples, axis=0)
        self.plot(t, waveforms.T, color=color,
                  box_index=box_index,
                  data_bounds=None,
                  )

    def _plot_labels(self, traces):
        for ch in range(self.n_channels):
            ch_label = '%d' % self.channel_order[ch]
            och = self.channel_vertical_order[ch]
            self[och].text(pos=[-1., traces[0, ch]],
                           text=ch_label,
                           anchor=[+1., -.1],
                           data_bounds=None,
                           )

    def _restrict_interval(self, interval):
        start, end = interval
        # Round the times to full samples to avoid subsampling shifts
        # in the traces.
        start = int(round(start * self.sample_rate)) / self.sample_rate
        end = int(round(end * self.sample_rate)) / self.sample_rate
        # Restrict the interval to the boundaries of the traces.
        if start < 0:
            end += (-start)
            start = 0
        elif end >= self.duration:
            start -= (end - self.duration)
            end = self.duration
        start = np.clip(start, 0, end)
        end = np.clip(end, start, self.duration)
        assert 0 <= start < end <= self.duration
        return start, end

    # Public methods
    # -------------------------------------------------------------------------

    def set_interval(self, interval=None, change_status=True,
                     force_update=False):
        """Display the traces and spikes in a given interval."""
        if interval is None:
            interval = self._interval
        interval = self._restrict_interval(interval)
        if not force_update and interval == self._interval:
            return
        self._interval = interval
        start, end = interval
        self.clear()

        # OPTIM: normalize time manually into [-1.0, 1.0].
        def _normalize_time(t):
            return -1. + (2. / float(end - start)) * (t - start)
        self._normalize_time = _normalize_time

        # Set the status message.
        if change_status:
            self.set_status('Interval: {:.3f} s - {:.3f} s'.format(start, end))

        # Load the traces.
        traces = self.traces(interval)

        # Plot the traces.
        self._plot_traces(traces.data, color=traces.color)

        # Plot the spikes.
        waveforms = traces.waveforms
        assert isinstance(waveforms, list)
        for w in waveforms:
            self._plot_waveforms(waveforms=w.data,
                                 color=w.color,
                                 channel_ids=w.get('channel_ids', None),
                                 start_time=w.start_time,
                                 )

        # Plot the labels.
        if self.do_show_labels:
            self._plot_labels(traces.data)

        self.build()
        self.update()

    def on_select(self, cluster_ids=None):
        super(TraceView, self).on_select(cluster_ids)
        self.set_interval(self._interval, change_status=False)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(TraceView, self).attach(gui)
        self.actions.add(self.go_to, alias='tg')
        self.actions.add(self.shift, alias='ts')
        self.actions.add(self.go_right)
        self.actions.add(self.go_left)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.add(self.widen)
        self.actions.add(self.narrow)
        self.actions.add(self.toggle_show_labels)

    @property
    def state(self):
        return Bunch(scaling=self.scaling,
                     origin=self.origin,
                     interval=self._interval,
                     do_show_labels=self.do_show_labels,
                     )

    # Scaling
    # -------------------------------------------------------------------------

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
        self._update_boxes()

    # Origin
    # -------------------------------------------------------------------------

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value
        self._update_boxes()

    # Navigation
    # -------------------------------------------------------------------------

    @property
    def time(self):
        """Time at the center of the window."""
        return sum(self._interval) * .5

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        self.set_interval(value)

    @property
    def half_duration(self):
        """Half of the duration of the current interval."""
        if self._interval is not None:
            a, b = self._interval
            return (b - a) * .5
        else:
            return self.interval_duration * .5

    def go_to(self, time):
        """Go to a specific time (in seconds)."""
        half_dur = self.half_duration
        self.set_interval((time - half_dur, time + half_dur))

    def shift(self, delay):
        """Shift the interval by a given delay (in seconds)."""
        self.go_to(self.time + delay)

    def go_right(self):
        """Go to right."""
        start, end = self._interval
        delay = (end - start) * .2
        self.shift(delay)

    def go_left(self):
        """Go to left."""
        start, end = self._interval
        delay = (end - start) * .2
        self.shift(-delay)

    def widen(self):
        """Increase the interval size."""
        t, h = self.time, self.half_duration
        h *= self.scaling_coeff_x
        self.set_interval((t - h, t + h))

    def narrow(self):
        """Decrease the interval size."""
        t, h = self.time, self.half_duration
        h /= self.scaling_coeff_x
        self.set_interval((t - h, t + h))

    def toggle_show_labels(self):
        self.do_show_labels = not self.do_show_labels
        self.set_interval(force_update=True)

    # Channel scaling
    # -------------------------------------------------------------------------

    def _update_boxes(self):
        self.stacked.box_size = self.box_size * self.scaling

    def increase(self):
        """Increase the scaling of the traces."""
        self.scaling *= self.scaling_coeff_y
        self._update_boxes()

    def decrease(self):
        """Decrease the scaling of the traces."""
        self.scaling /= self.scaling_coeff_y
        self._update_boxes()
