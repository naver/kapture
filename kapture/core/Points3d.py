# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

import numpy as np


class Points3d(np.ndarray):
    """
    3D points. This can be used like this:

    - 3-D points is a Nx6 numpy array: 1 point per row x 6 cols (x,y,z,r,g,b) with color
                 or a NX3 numpy array: 1 point per row X 3 cols (x,y,z)    without color
    - color (rgb) is coded from 0 to 255
    - optional per point timestamps (int64, one per row, in the time unit of the kapture records, usually
      microseconds) are stored in the attribute `timestamps` (None when absent). They are not carried over by numpy
      operations (slices, masks, stacks...): pass them explicitly to the new Points3d.
    """
    XYZ_ONLY = 3
    XYZ_RGB = 6
    COLUMN_TYPE = np.float64

    def __new__(cls, input_array=None, size=XYZ_RGB, timestamps=None):
        """
        Creates the ndarray instance of our type, given the usual ndarray input arguments.
        This will call the standard ndarray constructor, but return an object of our type.

        :param input_array: array of values
        :param size: size of default constructor
        :param timestamps: optional per point timestamps (N integers, same order as the points).
                           Defaults to the ones of input_array if it is a Points3d.
        """
        if input_array is None:
            # default constructor, empty shape
            empty_shape = (0, size)
            obj = super(Points3d, cls).__new__(cls, empty_shape).view(cls)
        elif isinstance(input_array, np.ndarray):
            # Input array is an already formed ndarray instance
            # We first cast to be our class type
            obj = np.asarray(input_array, dtype=Points3d.COLUMN_TYPE).view(cls)
        elif isinstance(input_array, list):
            # convert to numpy array and cast to our class type
            obj = np.array(input_array, dtype=Points3d.COLUMN_TYPE).view(cls)
        else:
            raise TypeError('Unknown type to convert 3-D points from')

        if timestamps is None and isinstance(input_array, Points3d):
            timestamps = input_array.timestamps  # Points3d(points3d) keeps the timestamps
        if obj is input_array:  # asarray(...).view() may return the input itself: do not alter it
            obj = obj.view(cls)
        obj.timestamps = timestamps
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # timestamps are never inherited: a view, slice or mask may not have the same points
        self._timestamps = None
        if (obj is None
                or (self.shape and len(self.shape) == 2
                    and (self.shape[1] == Points3d.XYZ_ONLY or self.shape[1] == Points3d.XYZ_RGB))
                or (self.shape and len(self.shape) == 1
                    and (self.shape[0] == Points3d.XYZ_ONLY or self.shape[0] == Points3d.XYZ_RGB))):
            return
        # from casting or view, check the shape is correct
        if (not self.shape
                or len(self.shape) < 2
                or not (self.shape[1] == Points3d.XYZ_ONLY or self.shape[1] == Points3d.XYZ_RGB)):
            raise ValueError('3D points are expected to be Nx3 or Nx6 shape ({})'.format(self.shape))

    def __getitem__(self, items):
        # with __array_finalize__ strict shape checking, getting a slice of Points3d can be an issue e.g p[:, 0:4] fails
        # this function tries to slice the Points3d normally.
        # If __array_finalize__ fails: e.g shape is not correct then it will return a slice of a normal numpy array
        try:
            return super(Points3d, self).__getitem__(items)
        except ValueError:
            return self.as_array().__getitem__(items)

    def __bool__(self):
        """
        Checks for emptiness

        :return: True if not empty. """
        return self.shape[0] > 0

    def has_colors(self):
        """
        Check for the presence of the colors columns
        """
        return self.shape[1] == Points3d.XYZ_RGB

    def as_array(self):
        """ cast the Points3d to numpy array. """
        return self.view(dtype=Points3d.COLUMN_TYPE, type=np.ndarray)

    @property
    def timestamps(self):
        """ per point timestamps (int64 array of N values) or None """
        return getattr(self, '_timestamps', None)

    @timestamps.setter
    def timestamps(self, timestamps):
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if timestamps.dtype.kind not in 'iu':
                raise TypeError(f'3D points timestamps are expected to be integers ({timestamps.dtype})')
            timestamps = timestamps.astype(np.int64, copy=False).reshape(-1)
            nb_points = self.shape[0] if len(self.shape) == 2 else 1
            if len(timestamps) != nb_points:
                raise ValueError(f'3D points timestamps: {len(timestamps)} values for {nb_points} points')
        self._timestamps = timestamps

    def has_timestamps(self) -> bool:
        """
        Check for the presence of per point timestamps
        """
        return self.timestamps is not None
