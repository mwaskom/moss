from __future__ import division
import os
import re
import subprocess
import tempfile
import shutil
import mimetypes
import copy

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


class EyeData(object):

    def __init__(self, fname):

        self.settings = dict(PRESCALER=None,
                             VPRESCALER=None,
                             PUPIL=None,
                             EVENTS=None,
                             SAMPLES=None)

        self.messages = pd.Series(index=pd.Int64Index([], name="timestamp"))

        self.samples = []
        self.fixations = []
        self.saccades = []
        self.blinks = []

        # Obtain eye data in ASCII format
        type, encoding = mimetypes.guess_type(fname)
        if type == "text/plain":
            temp_dir = None
            asc_file = fname
        else:
            temp_dir = tempfile.mkdtemp()
            asc_file = self._edf_to_asc(fname, temp_dir)

        # Process the eye data file
        self._parse_asc_file(asc_file)

        # Convert to better representations of the data
        samples = pd.DataFrame(self.samples,
                               columns=["timestamp", "x", "y", "pupil"])
        self.samples = (samples.replace({".": np.nan})
                               .apply(pd.to_numeric)
                               .set_index("timestamp"))

        fix_columns = ["start", "end", "duration", "x", "y", "pupil"]
        fixations = pd.DataFrame(self.fixations, columns=fix_columns)
        self.fixations = fixations.replace({".": np.nan}).apply(pd.to_numeric)

        sacc_columns = ["start", "end", "duration",
                        "start_x", "start_y", "end_x", "end_y",
                        "amplitude", "peak_velocity"]
        saccades = pd.DataFrame(self.saccades, columns=sacc_columns)
        self.saccades = saccades.replace({".": np.nan}).apply(pd.to_numeric)

        blink_columns = ["start", "end", "duration"]
        blinks = pd.DataFrame(self.blinks, columns=blink_columns)
        self.blinks = blinks.replace({".": np.nan}).apply(pd.to_numeric)

        # Parse some settings
        sample_settings = self.settings["SAMPLES"]
        self.units = sample_settings.split()[0]
        m = re.search("RATE (\d+\.00)", sample_settings)
        self.sampling_rate = float(m.group(1))

        # Clean up
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

    def _edf_to_asc(self, edf_file, temp_dir):

        subprocess.call(["edf2asc",
                         "-p", temp_dir,
                         edf_file])

        self._temp_dir = temp_dir

        edf_basename = os.path.basename(edf_file)
        asc_basename = edf_basename[:-3] + "asc"
        asc_file = os.path.join(temp_dir, asc_basename)

        return asc_file

    def _parse_asc_file(self, asc_file):

        with open(asc_file) as fid:
            for line in fid:
                self._parse_line(line)

    def _parse_line(self, line):

        if not line[0].strip():
            return

        if line.startswith("*"):
            return

        fields = line.split()

        if fields[0] in self.settings:
            self.settings[fields[0]] = " ".join(fields[1:])

        if fields[0] == "MSG":
            timestamp = int(fields[1])
            self.messages.loc[timestamp] = " ".join(fields[2:])

        if fields[0] in ["SFIX", "SSACC", "SBLINK"]:
            return

        # Note that we are not reading the eye field for events, assuming
        # that we are in monocular mode (as we always should be).
        # This makes it simpler to convert data to numeric after parsing.
        if fields[0] in ["EFIX"]:
            self.fixations.append(fields[2:])

        if fields[0] in ["ESACC"]:
            self.saccades.append(fields[2:])

        if fields[0] in ["EBLINK"]:
            self.blinks.append(fields[2:])

        try:
            timestamp = int(fields[0])
        except ValueError:
            return

        self.samples.append(fields[:4])

    def convert_to_degrees(self, width, distance, resolution):
        """Convert eye position data from pixels to degrees.

        Also changes the origin from the upper right hand corner to the center
        of the screen.

        Modifies the data inplace and returns self for easy chaining.

        """
        def recenter_x_data(x):
            x -= resolution[0] / 2

        def recenter_y_data(y):
            y -= resolution[1] / 2
            y *= -1

        def pix_to_deg(data):
            data *= (width / resolution[0])
            data /= (distance * 0.017455)

        for field in ["samples", "fixations"]:
            data = getattr(self, field)
            recenter_x_data(data["x"])
            recenter_y_data(data["y"])
            pix_to_deg(data["x"])
            pix_to_deg(data["y"])

        for point in ["start", "end"]:
            recenter_x_data(self.saccades[point + "_x"])
            recenter_y_data(self.saccades[point + "_y"])
            pix_to_deg(self.saccades[point + "_x"])
            pix_to_deg(self.saccades[point + "_y"])

        return self

    def filter_blinks(self):
        """Remove blinks from saccade and sample data.

        Sample data is set to null between the start and end of saccades
        that include a blink, and then those saccades are removed from the
        saccades database.

        Modifies the data inplace and returns self for easy chaining.

        """
        true_saccades = []
        for i, s in self.saccades.iterrows():
            blink = ((self.blinks.start > s.start)
                     & (self.blinks.end < s.end)
                     ).any()

            if blink:
                self.samples.loc[s.start:s.end, ["x", "y"]] = np.nan
            else:
                true_saccades.append(i)

        self.saccades = self.saccades.loc[true_saccades].reset_index(drop=True)

        return self

    def reindex_to_experiment_clock(self, start_message="SYNCTIME"):
        """Convert timing data to seconds from experiment onset.

        Modifies the data inplace and returns self for easy chaining.

        Parameters
        ----------
        start_message : str
            Message text indicating the timepoint of the experiment onset.

        """
        start_time = self.messages[self.messages == "SYNCTIME"].index.item()

        self.samples.index = (self.samples.index - start_time) / 1000

        def reindex_events(df):
            cols = ["start", "end"]
            df.loc[:, cols] = (df[cols] - start_time) / self.sampling_rate
            df.loc[:, "duration"] /= 1000

        for event_type in ["blinks", "saccades", "fixations"]:
            reindex_events(getattr(self, event_type))

        return self

    def detect_saccades(self, kernel_sigma=.003,
                        start_thresh=(20, .005),
                        end_thresh=(7, .005)):
        """Detect saccade events, replacing the Eyelink data.

        Assumes that the timestamps have been converted to second resolution
        and the samples have been converted to degrees.

        This replaces the ``saccades`` attribute that is originally populated
        with the results of the Eyelink saccade detection algorithm; the Eyelink
        data is copied to the ``eyelink_saccades`` attribute in this method.

        Parameters
        ----------
        kernel_sigma : float
            Standard deviation of the smoothing kernel, in milliseconds. Samples
            are smoothed before eye movement velocity is computed.
        start_thresh, end_thresh : float, float pairs
            Each pair gives the velocity threshold and required duration for the
            respective identification of saccade onsets and offsets.

        Note
        ----
        This method currently does not alter any information in the
        ``fixations`` field , which retains the fixation onset and offset timing
        assigned by the Eyelink algorithm. As a result, samples might end up
        being tagged as both a "fixation" and a "saccade".

        """
        # Save a copy of the original eyelink saccades
        if not hasattr(self, "eyelink_saccades"):
            self.eyelink_saccades = self.saccades.copy()

        # Compute smoothing kernel size in sample bin units
        dt = 1 / self.sampling_rate
        kernel_width = kernel_sigma / dt

        # Extract gaze position
        xy = self.samples[["x", "y"]]

        # Smooth the gaze position data
        xy_s = xy.apply(gaussian_filter1d, args=(kernel_width,))

        # Compute velocity
        v = (xy_s.diff()
                 .apply(np.square)
                 .sum(axis=1, skipna=False)
                 .apply(np.sqrt)
                 .divide(dt)
                 .fillna(np.inf))

        # Identify saccade onsets
        start_window = int(start_thresh[1] / dt)
        sthr = (v > start_thresh[0]).rolling(start_window).min()
        starts = xy.where(sthr.diff() == 1).dropna()

        # Identify saccade offsets
        end_window = int(end_thresh[1] / dt)
        ethr = (v < end_thresh[0]).rolling(end_window).min()
        ends = xy.where(ethr.diff() == 1).dropna()

        # -- Parse each detected onset to identify the corresponding end
        saccades = []
        last_end_time = 0
        for start_time, start_pos in starts.iterrows():

            if start_time < last_end_time:
                # This is an acceration within a prior saccade that has not
                # yet ended; skip
                continue

            ends = ends.loc[start_time:]

            # Check if the dataset is ending in the middle of a saccade
            if ends.size:
                end = ends.iloc[0]
            else:
                break

            last_end_time = end.name

            saccades.append([start_time,
                             start_pos["x"],
                             start_pos["y"],
                             end.name,
                             end["x"],
                             end["y"]])

        # Package the saccades into a dataframe
        columns = ["start", "start_x", "start_y", "end", "end_x", "end_y"]
        saccades = pd.DataFrame(saccades, columns=columns)

        # -- Compute duration, amplitude, velocity, and angle

        dx = saccades["end_x"] - saccades["start_x"]
        dy = saccades["end_y"] - saccades["start_y"]
        saccades["amplitude"] = np.sqrt(np.square(dx) + np.square(dy))

        saccades["duration"] = saccades["end"] - saccades["start"]
        saccades["velocity"] = saccades["amplitude"] / saccades["duration"]

        saccades["angle"] = np.rad2deg(np.arctan2(dy, dx)) % 360

        # Overwrite the saccade data structure with the new results
        self.saccades = saccades

        return self

    @property
    def events(self):

        event_types = ["fixations", "saccades", "blinks"]
        events = pd.DataFrame(False,
                              index=self.samples.index,
                              columns=event_types)

        for event in event_types:
            event_data = getattr(self, event)
            for _, ev in event_data.iterrows():
                events.loc[ev.start:ev.end, event] = True

        return events
