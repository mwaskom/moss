import os
import subprocess
import tempfile
import shutil

import numpy as np
import pandas as pd


class EyeData(object):

    def __init__(self, edf_file=None, asc_file=None):

        if edf_file is None and asc_file is None:
            raise ValueError("Must pass either EDF or ASCII file")

        self.settings = dict(PRESCALER=None,
                             VPRESCALER=None,
                             PUPIL=None,
                             EVENTS=None,
                             SAMPLES=None)

        self.messages = pd.Series(index=pd.Int64Index([], name="timestamp"))

        self.eye_data = []
        self.fixations = []
        self.saccades = []
        self.blinks = []

        # Obtain eye data in ASCII format
        if asc_file is None:
            temp_dir = tempfile.mkdtemp()
            asc_file = self.edf_to_asc(edf_file, temp_dir)
        else:
            temp_dir = None

        # Process the eye data file
        self.parse_asc_file(asc_file)

        # Convert to better representations of the data
        eye_data = pd.DataFrame(self.eye_data,
                                columns=["timestamp", "x", "y", "pupil"])
        self.eye_data = (eye_data.replace({".": np.nan})
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

        # Clean up
        if temp_dir is not None:
            shutil.rmtree(temp_dir)

    def edf_to_asc(self, edf_file, temp_dir):

        subprocess.call(["edf2asc",
                         "-p", temp_dir,
                         edf_file])

        self._temp_dir = temp_dir

        edf_basename = os.path.basename(edf_file)
        asc_basename = edf_basename[:-3] + "asc"
        asc_file = os.path.join(temp_dir, asc_basename)

        return asc_file

    def parse_asc_file(self, asc_file):

        with open(asc_file) as fid:
            for line in fid:
                self.parse_line(line)

    def parse_line(self, line):

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

        self.eye_data.append(fields[:4])
